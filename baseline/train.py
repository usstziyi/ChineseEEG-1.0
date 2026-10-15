"""baseline 的对比学习训练循环（单被试 + 单小说）。

与 ``char_token/train.py`` 同构，但数据组织不同：

- 一次只跑一个 ``(subject, novel)``，对应一个 ``ReadingEEGDataset``；
- ``ReadingEEGDataset`` 默认按 ``sentence`` 去重，且每个实例内每句只剩一条样本，
  所以**没有同句多视图**——不使用分组采样器，直接随机 shuffle；
- 正样本就是"自己"：一批里每句只出现一次，损失即标准 CLIP 式单正样本 InfoNCE
  （``char_token`` 里 ``--n-repeats 1`` 的退化情形）。

数据
----
句子来自 ``datasets/preprocessed/novels/{subject}_{novel}.csv``，EEG 来自
``datasets/ChineseEEG/derivatives/preproc/filtered_0.5_30`` 下的 BrainVision 文件。
train / val 按 **句子** 切分（不是按样本），保证两侧句子集合完全不相交。

损失
----
对称 InfoNCE：

    logits = eeg_embedding @ text_embedding.T / temperature
    loss   = 0.5 * (CE(logits, 对角线) + CE(logits.T, 对角线))

温度系数默认 0.2；同 batch 内的其他句子构成 in-batch 负样本。

评测
----
- ``eeg2text``：候选 = val 里的唯一句子，命中 = 前 k 名。
- ``text2eeg``：候选 = 全部 val EEG 样本，命中 = 前 k 名里出现同句。
  （本设定下两句候选池等价于对角线检索。）

``val_loss`` 越小越好，其余指标越大越好；best checkpoint 默认按 ``val_loss`` 选。

文本侧用冻结的 LaBSE，所有句子的文本 embedding 在训练前一次性算好、做成查表，
训练时只做索引。

用法::

    uv run python baseline/train.py
    uv run python baseline/train.py --subject sub-04 --novel LittlePrince
    uv run python baseline/train.py --subject sub-04 --novel GarnettDream --epochs 20 --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# 支持 `python baseline/train.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.dataset import ReadingEEGDataset, pad_collate
from baseline.model import ReadingEEGEncoder
from baseline.text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "baseline"

# train_epochs.csv 每轮追加一行，字段顺序即表头顺序
EPOCH_LOG_FIELDS = [
    "epoch", "lr", "train_loss",
    "val_loss", "n_val", "n_unique_sentences", "n_text_candidates",
    "eeg2text_top1", "eeg2text_top5", "text2eeg_top1", "text2eeg_top5",
    "is_best", "temperature",
]

# train_steps.csv 每个训练 step 追加一行：第 0 列 epoch、第 1 列 step
TRAIN_STEP_FIELDS = ["epoch", "step", "loss", "lr", "batch_size"]


def append_csv_row(path: Path, fields: list, row: dict) -> None:
    """把一行追加到 CSV；文件不存在时先写表头。"""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def dump_args_json(args, out_dir: Path, filename: str = "args.json") -> Path:
    """把本次运行的所有命令行参数（key/value）以 JSON 写到时间戳输出目录。"""
    path = Path(out_dir) / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_sentence(dataset: ReadingEEGDataset, val_ratio: float, seed: int):
    """按句子切分 train / val，返回 ``(train_subset, val_subset, 统计信息)``。

    以句子为单位而非样本，避免同一句子的不同样本同时出现在两侧（本数据集已按
    句子去重，故等价于按样本随机切分，但仍以句子口径实现以防将来关闭去重）。
    """
    sentences = [row["sentence"] for row in dataset.rows]
    unique = sorted(set(sentences))
    if len(unique) < 2:
        raise ValueError(f"只有 {len(unique)} 个唯一句子，无法切分")

    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_ratio))
    n_val = min(max(n_val, 1), len(shuffled) - 1)
    val_sentences = set(shuffled[:n_val])

    val_idx = [i for i, s in enumerate(sentences) if s in val_sentences]
    train_idx = [i for i, s in enumerate(sentences) if s not in val_sentences]

    # 泄漏检查：按句子切分后，train / val 的句子集合必须完全不相交
    train_sentences = {sentences[i] for i in train_idx}
    info = {
        "n_unique_train": len(unique) - n_val,
        "n_unique_val": n_val,
        "sentence_overlap": len(train_sentences & val_sentences),
    }
    return Subset(dataset, train_idx), Subset(dataset, val_idx), info


def build_text_lookup(encoder, sentences, device, chunk: int = 64):
    """把所有唯一句子编码成 embedding，返回 ``({sentence: (D,)}, embedding_dim)``。"""
    unique = sorted(set(sentences))
    vectors = []
    with torch.no_grad():
        for start in range(0, len(unique), chunk):
            vectors.append(encoder(unique[start : start + chunk]))
    table = torch.cat(vectors, dim=0).to(device)
    embedding_dim = table.shape[1]
    return dict(zip(unique, table)), embedding_dim


def gather_text(sentences, lookup, device) -> torch.Tensor:
    """按句子取回文本 embedding，堆成 (B, D)。"""
    return torch.stack([lookup[s] for s in sentences]).to(device)


def sentence_group_ids(sentences, device) -> torch.Tensor:
    """把句子列表映射成组 id：同句得到同一个 id（去重后每句唯一，即 0..B-1）。"""
    mapping: dict[str, int] = {}
    ids = []
    for sentence in sentences:
        if sentence not in mapping:
            mapping[sentence] = len(mapping)
        ids.append(mapping[sentence])
    return torch.tensor(ids, dtype=torch.long, device=device)


def contrastive_loss(eeg: torch.Tensor, text: torch.Tensor, temperature: float) -> torch.Tensor:
    """对称单正样本 InfoNCE：EEG->文本 与 文本->EEG 两个方向取平均。

    一批里每句只出现一次（去重后），正样本即对角线；同 batch 的其他句子构成
    in-batch 负样本。
    """
    logits = eeg @ text.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )


def train_one_epoch(
    model, loader, lookup, optimizer, device, temperature, epoch, step_log_path=None
):
    """跑一个 epoch，返回平均 loss。

    传了 ``step_log_path`` 就把每个 step 追加到 CSV：第 0 列 epoch、第 1 列 step。
    """
    model.train()
    total_loss, total_n = 0.0, 0
    lr = optimizer.param_groups[0]["lr"]
    for step, (tokens, mask, sentences) in enumerate(loader, start=1):
        tokens = tokens.to(device)  # (B, n, C, T)
        mask = mask.to(device)      # (B, n)
        text = gather_text(sentences, lookup, device)  # (B, D)

        eeg = model(tokens, mask)
        loss = contrastive_loss(eeg, text, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch = len(sentences)
        if step_log_path is not None:
            append_csv_row(step_log_path, TRAIN_STEP_FIELDS, {
                "epoch": epoch,
                "step": step,
                "loss": f"{loss.item():.6f}",
                "lr": f"{lr:.6f}",
                "batch_size": batch,
            })

        total_loss += loss.item() * batch
        total_n += batch
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, lookup, device, temperature) -> dict:
    """在整个 val 集上算双向检索指标与全库对比损失。"""
    model.eval()
    eeg_parts, sentences = [], []
    for tokens, mask, batch_sentences in loader:
        eeg_parts.append(model(tokens.to(device), mask.to(device)))
        sentences.extend(batch_sentences)

    eeg = torch.cat(eeg_parts, dim=0)
    text = gather_text(sentences, lookup, device)
    n = eeg.size(0)

    metrics = {
        "val_loss": float(contrastive_loss(eeg, text, temperature)),
        "n_val": n,
        "n_unique_sentences": len(set(sentences)),
    }

    # ---- eeg -> text：候选池 = val 里的唯一句子 ----
    unique_sentences = sorted(set(sentences))
    sentence_to_col = {s: i for i, s in enumerate(unique_sentences)}
    text_unique = gather_text(unique_sentences, lookup, device)
    target = torch.tensor(
        [sentence_to_col[s] for s in sentences], device=device, dtype=torch.long
    )  # 每条样本的正确答案列号
    ranked = (eeg @ text_unique.t()).argsort(dim=1, descending=True)
    hit = ranked.eq(target[:, None])
    metrics["n_text_candidates"] = len(unique_sentences)
    metrics["eeg2text_top1"] = float(hit[:, :1].any(dim=1).float().mean())
    metrics["eeg2text_top5"] = float(hit[:, :5].any(dim=1).float().mean())

    # ---- text -> eeg：候选池 = 全部 val EEG 样本，命中 = 前 k 名里出现同句 ----
    groups = sentence_group_ids(sentences, device)
    positive = groups[:, None] == groups[None, :]
    ranked = (text @ eeg.t()).argsort(dim=1, descending=True)
    hit = positive[torch.arange(n, device=device)[:, None], ranked]
    metrics["text2eeg_top1"] = float(hit[:, :1].any(dim=1).float().mean())
    metrics["text2eeg_top5"] = float(hit[:, :5].any(dim=1).float().mean())

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="baseline：单被试单小说的 EEG–文本对比学习。")
    parser.add_argument("--subject", default="sub-04", help="被试编号，如 sub-04（默认 sub-04）")
    parser.add_argument("--novel", default="LittlePrince", help="小说名，如 LittlePrince（默认 LittlePrince）")
    parser.add_argument("--min-chars", type=int, default=4, help="只保留 n_chars >= 该值的句子（默认 4）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集句子比例（默认 0.2）")
    parser.add_argument("--batch-size", type=int, default=32, help="batch 大小（默认 32）")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数（默认 30）")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率（默认 1e-3）")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW 权重衰减（默认 1e-3）")
    parser.add_argument(
        "--lr-schedule", default="cosine", choices=["none", "cosine"],
        help="学习率调度：cosine 按 epoch 余弦衰减到 0（默认），none 恒定",
    )
    parser.add_argument(
        "--select-by", default="val_loss",
        choices=["val_loss", "eeg2text_top1", "eeg2text_top5", "text2eeg_top5"],
        help="用哪个指标选 best checkpoint（默认 val_loss，越小越好；其余越大越好）",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="对比学习温度系数（默认 0.2）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数（默认 0）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Config] device={device} subject={args.subject} novel={args.novel} "
          f"batch={args.batch_size} epochs={args.epochs} lr={args.lr} "
          f"wd={args.weight_decay} schedule={args.lr_schedule} "
          f"select={args.select_by} temperature={args.temperature} seed={args.seed}")

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_args_json(args, out_dir)

    # ---------------- 数据 ----------------
    dataset = ReadingEEGDataset(
        args.subject, args.novel, min_chars=args.min_chars
    )
    first_tokens, _ = dataset[0]  # 触发一次读取，确定通道数 / token 长度
    n_chans, token_samples = first_tokens.shape[1], first_tokens.shape[2]
    sentences = [row["sentence"] for row in dataset.rows]

    train_set, val_set, info = split_by_sentence(dataset, args.val_ratio, args.seed)
    print(
        f"[Data] {args.subject} {args.novel} 样本 {len(dataset)}"
        f"（min_chars={args.min_chars}, dedup=True）\n"
        f"       n_chans={n_chans} token_samples={token_samples}\n"
        f"       句子 {len(set(sentences))} 条 -> "
        f"train {info['n_unique_train']} / val {info['n_unique_val']} "
        f"(样本 {len(train_set)} / {len(val_set)})，句子重叠={info['sentence_overlap']}"
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=pad_collate, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=pad_collate, num_workers=args.num_workers,
    )

    # ---------------- 文本 embedding（冻结，一次性算好） ----------------
    print("[Text] 加载 LaBSE（冻结） ...")
    text_encoder = LaBSETextEncoder()
    lookup, embedding_dim = build_text_lookup(text_encoder, sentences, device)
    print(f"[Text] 已缓存 {len(lookup)} 条句子，embedding_dim={embedding_dim}")

    # ---------------- 模型 ----------------
    model = ReadingEEGEncoder(
        n_chans=n_chans,
        token_samples=token_samples,
        embedding_dim=embedding_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] {type(model).__name__} 参数量={n_params:,} "
        f"d_model={model.d_model} layers={len(model.transformer.layers)} "
        f"max_tokens={model.max_tokens}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # 恒定学习率在 val_loss 走平后仍会继续拟合，余弦衰减到 0 有助于收敛
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )

    # ---------------- 训练 ----------------
    tag = f"{args.subject}_{args.novel}"
    ckpt_path = out_dir / f"{tag}_best.pt"
    train_epochs_path = out_dir / f"{tag}_train_epochs.csv"
    train_steps_path = out_dir / f"{tag}_train_steps.csv"

    lower_is_better = args.select_by == "val_loss"
    best_score = float("inf") if lower_is_better else -float("inf")
    best_metrics = None
    header = (
        f"{'epoch':>5} {'lr':>8} {'train_loss':>10} {'val_loss':>9} "
        f"{'e2t@1':>7} {'e2t@5':>7} {'t2e@1':>7} {'t2e@5':>7}"
    )
    print(f"\n{header}  (best by {args.select_by})")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, lookup, optimizer, device, args.temperature,
            epoch=epoch, step_log_path=train_steps_path,
        )
        if scheduler is not None:
            scheduler.step()
        metrics = evaluate(model, val_loader, lookup, device, args.temperature)

        score = metrics[args.select_by]
        improved = score < best_score if lower_is_better else score > best_score
        flag = ""
        if improved:
            best_score = score
            best_metrics = dict(metrics)
            flag = " *"
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "n_chans": n_chans,
                    "token_samples": token_samples,
                    "embedding_dim": embedding_dim,
                    "metrics": metrics,
                },
                ckpt_path,
            )

        print(
            f"{epoch:>5} {epoch_lr:>8.2e} {train_loss:>10.4f} "
            f"{metrics['val_loss']:>9.4f} "
            f"{metrics['eeg2text_top1']:>7.4f} {metrics['eeg2text_top5']:>7.4f} "
            f"{metrics['text2eeg_top1']:>7.4f} {metrics['text2eeg_top5']:>7.4f}{flag}"
        )

        # 本轮指标追加到 train_epochs.csv（首轮自动写表头）
        append_csv_row(train_epochs_path, EPOCH_LOG_FIELDS, {
            "epoch": epoch,
            "lr": f"{epoch_lr:.6f}",
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{metrics['val_loss']:.6f}",
            "n_val": metrics["n_val"],
            "n_unique_sentences": metrics["n_unique_sentences"],
            "n_text_candidates": metrics["n_text_candidates"],
            "eeg2text_top1": f"{metrics['eeg2text_top1']:.6f}",
            "eeg2text_top5": f"{metrics['eeg2text_top5']:.6f}",
            "text2eeg_top1": f"{metrics['text2eeg_top1']:.6f}",
            "text2eeg_top5": f"{metrics['text2eeg_top5']:.6f}",
            "is_best": int(improved),
            "temperature": args.temperature,
        })

    # 随机基线也要按两个方向的候选池分别算
    n_val = best_metrics["n_val"]
    n_cand = best_metrics["n_text_candidates"]
    print(
        f"\n[Done] best {args.select_by} = {best_score:.4f}\n"
        f"       eeg2text 候选 {n_cand} 句，随机基线 top-1={1 / n_cand:.4f} "
        f"top-5={min(5, n_cand) / n_cand:.4f}\n"
        f"       text2eeg 候选 {n_val} 条，随机基线 top-1={1 / n_val:.4f} "
        f"top-5={min(5, n_val) / n_val:.4f}\n"
        f"       checkpoint: {ckpt_path}\n"
        f"       epochs:     {train_epochs_path}\n"
        f"       steps:      {train_steps_path}"
    )


if __name__ == "__main__":
    main()
