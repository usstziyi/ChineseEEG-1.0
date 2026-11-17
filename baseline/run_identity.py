"""run 身份探针：标签 = run，chapter 只用于内容层混淆诊断。

移植自 ``new_model/plan_run_identity``，差异：

- 数据来自本工程 ``baseline.dataset.ReadingEEGDataset``（128 导 / 256 Hz /
  每句 10 字、每字 100 点，定长 ``(128, 1000)``），Welch ``nperseg`` 由 256
  调整为 128（约 0.5 s）；
- 训练标签是 ``run``；``chapter`` **不参与训练**，只用于回答“run 身份的可分性
  里有多少其实是读到哪一章带来的”；
- 跳过句类混淆诊断、通道坐标与坏导扫描。

评估口径与 ``plan_run_identity`` 一致：

1. 按 run 分层的随机 8:2 holdout（train:test = 4:1，不设 val）；
2. 全量 5 折 ``StratifiedKFold``；
3. 全局打乱 run 标签的置换零分布（默认 2000 次）；
4. 按 run 聚类的 bootstrap 95% CI。

注意（GarnettDream 的结构性局限）：该小说 18 个 run 恰好对应 18 章，
run 与 chapter 一一对应，因此 chapter 诊断会退化为恒等（多数类占比 = 1、
NMI = 1）。此时“run 身份”与“章节内容”在数据上不可分离，不能据此判断高分
来自设备/状态还是内容；有信息量的混淆诊断只能看 LittlePrince（27 章 / 7 run）。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from baseline.dataset import N_CHANNELS, SAMPLES_PER_CHAR, ReadingEEGDataset
from baseline.model import BANDS, NPERSEG, extract_features, make_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "run_identity"

SUMMARY_FIELDS = [
    "subject",
    "novel",
    "n_samples",
    "n_runs",
    "n_chapters",
    "n_features",
    "sfreq",
    "nperseg",
    "band_mode",
    "model",
    "chance_accuracy",
    "test_accuracy",
    "test_accuracy_ci_low",
    "test_accuracy_ci_high",
    "test_macro_f1",
    "test_weighted_f1",
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "cv_accuracy_ci_low",
    "cv_accuracy_ci_high",
    "cv_macro_f1_mean",
    "permutation_n",
    "permutation_null_mean",
    "permutation_null_p95",
    "permutation_z",
    "permutation_p",
    "chapter_majority_baseline_accuracy",
    "chapter_random_baseline_accuracy",
    "chapter_mean_majority_share",
    "chapter_nmi",
    "chapter_uncertainty_coefficient",
    "chapter_cramers_v",
    "chapter_diagnostic_degenerate",
]


# --------------------------------------------------------------- 特征与数据


def extract_dataset_features(
    dataset: ReadingEEGDataset,
    mode: str = "rel",
    nperseg: int = NPERSEG,
    batch_size: int = 128,
    verbose: bool = True,
) -> dict:
    """遍历数据集，返回定长特征矩阵与标签/分组向量。

    句子长度 ``n_chars`` 在 4~10 之间浮动，无法整批 Welch，因此按 ``n_chars``
    分桶，桶内拼成 ``(B, 128, n_chars*100)`` 再算 PSD。**不补零**，避免零填充
    改变功率谱。
    """
    n_samples = len(dataset)
    n_features = len(BANDS) * N_CHANNELS
    feats = np.empty((n_samples, n_features), dtype=np.float32)
    labels = np.empty(n_samples, dtype=np.int64)
    runs = np.empty(n_samples, dtype=np.int64)
    chapters = np.empty(n_samples, dtype=np.int64)
    days = np.empty(n_samples, dtype=np.int64)

    run_values = sorted({int(row["run"]) for row in dataset.rows})
    run_to_label = {run: index for index, run in enumerate(run_values)}

    buckets: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        buckets[int(row["n_chars"])].append(index)

    for n_chars in sorted(buckets):
        length = n_chars * SAMPLES_PER_CHAR
        if nperseg > length:
            raise ValueError(
                f"nperseg={nperseg} 超过最短句子长度 {length}（n_chars={n_chars}）"
            )
        indices = buckets[n_chars]
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            block = np.empty((len(batch_indices), N_CHANNELS, length), dtype=np.float32)
            for j, index in enumerate(batch_indices):
                eeg, _, run, chapter, day = dataset[index]
                # (128, 1000)：10 个字的窗口按时间顺序首尾相接
                block[j] = np.asarray(eeg)
                labels[index] = run_to_label[run]
                runs[index] = run
                chapters[index] = chapter
                days[index] = day
            feats[np.asarray(batch_indices)] = extract_features(
                block, dataset.sfreq, mode=mode, nperseg=nperseg
            )
        if verbose:
            print(
                f"  [特征] n_chars={n_chars:<2} {len(indices):>5} 条 "
                f"(T={length})",
                flush=True,
            )

    return {
        "features": feats,
        "labels": labels,
        "runs": runs,
        "chapters": chapters,
        "days": days,
        "run_values": run_values,
        "n_features": n_features,
    }


# ------------------------------------------------------------------ 统计工具


def split_indices(
    labels: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """逐类分层随机划分（train / test），与 ``plan_run_identity`` 一致。"""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        index = np.flatnonzero(labels == label)
        rng.shuffle(index)
        n_test = max(int(round(len(index) * test_ratio)), 1)
        if n_test >= len(index):
            n_test = len(index) - 1
        test_parts.append(index[:n_test])
        train_parts.append(index[n_test:])
    return (
        np.sort(np.concatenate(train_parts)),
        np.sort(np.concatenate(test_parts)),
    )


def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_boot: int = 5000,
) -> tuple[float, float]:
    """按组（这里每个 run 即一组）重采样的准确率 95% CI。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index_by_group = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([index_by_group[j] for j in pick])
        accs[i] = np.mean(y_true[index] == y_pred[index])
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    folds: int,
    seed: int,
    return_oof: bool = False,
):
    """返回每折 (accuracy, macro_f1)；``return_oof=True`` 时附带 OOF 预测。"""
    accs, f1s = [], []
    oof = np.empty(len(y), dtype=np.asarray(y).dtype) if return_oof else None
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y):
        model = make_model(model_name)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        if oof is not None:
            oof[test_idx] = pred
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
    if oof is not None:
        return np.asarray(accs), np.asarray(f1s), oof
    return np.asarray(accs), np.asarray(f1s)


def _permutation_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    folds: int,
    seed: int,
) -> float:
    """一次置换：打乱标签后跑完整 K 折 CV，返回平均准确率。"""
    with threadpool_limits(limits=1):
        shuffled = np.random.default_rng(seed).permutation(y)
        accs, _ = cross_validate(X, shuffled, model_name, folds, seed)
    return float(accs.mean())


def permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    folds: int,
    seed: int,
    n_perm: int,
    n_jobs: int = 1,
) -> np.ndarray:
    """全局打乱 run 标签后的 5 折 CV 准确率零分布。"""
    if n_jobs > 1:
        null = np.empty(n_perm)
        chunk = 100
        for start in range(0, n_perm, chunk):
            stop = min(start + chunk, n_perm)
            values = Parallel(n_jobs=n_jobs)(
                delayed(_permutation_accuracy)(X, y, model_name, folds, seed + i)
                for i in range(start, stop)
            )
            null[start:stop] = values
            print(
                f"  [置换] {stop}/{n_perm} ({stop / n_perm:.0%}) "
                f"最近均值={np.mean(values):.4f}",
                flush=True,
            )
        return null

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    step = max(n_perm // 10, 1)
    for i in range(n_perm):
        shuffled = rng.permutation(y)
        accs, _ = cross_validate(X, shuffled, model_name, folds, seed)
        null[i] = accs.mean()
        if (i + 1) % step == 0 or i == n_perm - 1:
            print(
                f"  [置换] {i + 1}/{n_perm} ({(i + 1) / n_perm:.0%}) "
                f"acc={null[i]:.4f}",
                flush=True,
            )
    return null


# -------------------------------------------------------------- chapter 诊断


def chapter_diagnostics(
    runs: np.ndarray,
    chapters: np.ndarray,
    run_values: list[int],
) -> dict:
    """量化 run 与 chapter 的耦合程度（chapter 不参与训练）。"""
    rows = []
    total_majority = 0
    for run in run_values:
        mask = runs == run
        counts = Counter(int(c) for c in chapters[mask])
        majority, majority_count = counts.most_common(1)[0]
        total_majority += majority_count
        rows.append(
            {
                "run": int(run),
                "n_trials": int(mask.sum()),
                "n_chapters": len(counts),
                "majority_chapter": majority,
                "majority_share": majority_count / int(mask.sum()),
                "chapters": dict(sorted(counts.items())),
            }
        )

    chapter_ids = sorted({int(c) for c in chapters})
    n_chapters = len(chapter_ids)
    majority_baseline = total_majority / len(runs)

    run_to_row = {int(run): i for i, run in enumerate(run_values)}
    chapter_pos = {chapter: i for i, chapter in enumerate(chapter_ids)}
    table = np.zeros((len(run_values), n_chapters), dtype=np.int64)
    for run_number, chapter in zip(runs, chapters):
        table[run_to_row[int(run_number)], chapter_pos[int(chapter)]] += 1

    try:
        chi2 = float(chi2_contingency(table, correction=False)[0])
        dof = min(table.shape[0] - 1, table.shape[1] - 1)
        cramers_v = float(np.sqrt(chi2 / (len(runs) * max(dof, 1))))
    except ValueError:
        cramers_v = float("nan")

    nmi = float(normalized_mutual_info_score(runs, chapters))
    mi = float(mutual_info_score(runs, chapters))
    _, chapter_counts = np.unique(chapters, return_counts=True)
    p_chapter = chapter_counts / chapter_counts.sum()
    entropy_chapter = float(-(p_chapter * np.log(p_chapter)).sum())
    uncertainty = mi / entropy_chapter if entropy_chapter > 0 else float("nan")

    # run 与 chapter 一一对应时（GarnettDream），诊断退化为恒等
    degenerate = len(run_values) == n_chapters and all(
        row["n_chapters"] == 1 for row in rows
    )

    return {
        "n_runs": len(run_values),
        "n_chapters": n_chapters,
        "chapter_ids": chapter_ids,
        "majority_baseline_accuracy": float(majority_baseline),
        "random_baseline_accuracy": float(1.0 / n_chapters),
        "mean_majority_share": float(np.mean([row["majority_share"] for row in rows])),
        "nmi": nmi,
        "uncertainty_coefficient_chapter_given_run": float(uncertainty),
        "cramers_v": cramers_v,
        "cramers_v_note": (
            "chapter 严格嵌套在 run 内（chapter 可决定 run），对称的 Cramér's V "
            "会饱和到 1，不宜单独解读；请结合 NMI、多数类占比和 "
            "U(chapter|run) 判断"
        ),
        "degenerate": bool(degenerate),
        "note": (
            "run 与 chapter 一一对应，本章节诊断退化为恒等（majority=1, NMI=1），"
            "无法区分设备/状态指纹与内容效应"
            if degenerate
            else ""
        ),
        "runs": rows,
    }


# ------------------------------------------------------------------ 评估主体


def run_subject(
    args,
    subject: str,
    novel: str,
    out_root: Path,
) -> dict:
    print(f"\n===== {subject} / {novel} =====")
    dataset = ReadingEEGDataset(
        subject,
        novel,
        min_chars=args.min_chars,
        dedup=args.dedup,
        preload=args.preload,
        n_chars_exact=args.n_chars_exact,
    )
    print(
        f"[Dataset] {len(dataset)} 条 | {dataset.n_chans} 导 | "
        f"{dataset.sfreq:g} Hz | chars_per_sample={dataset.chars_per_sample} | day {dataset.day_ids}"
    )
    print(f"[Dataset] 过滤统计: {dataset.stats}")

    payload = extract_dataset_features(
        dataset, mode=args.bands_mode, nperseg=args.nperseg
    )
    dataset.close()

    X = payload["features"]
    y = payload["labels"]
    runs = payload["runs"]
    chapters = payload["chapters"]
    run_values = payload["run_values"]
    n_runs = len(run_values)
    chance = 1.0 / n_runs

    folds = min(args.folds, int(np.bincount(y).min()))
    if folds < args.folds:
        print(f"[Warn] 最小的 run 只有 {int(np.bincount(y).min())} 条，折数降为 {folds}")

    confound = chapter_diagnostics(runs, chapters, run_values)
    print(
        f"[Chapter] {confound['n_runs']} run / {confound['n_chapters']} chapter | "
        f"多数类基线={confound['majority_baseline_accuracy']:.4f} "
        f"(随机 {confound['random_baseline_accuracy']:.4f}) | "
        f"平均多数类占比={confound['mean_majority_share']:.4f} | "
        f"NMI={confound['nmi']:.4f} | "
        f"U(chapter|run)={confound['uncertainty_coefficient_chapter_given_run']:.4f} | "
        f"Cramér's V={confound['cramers_v']:.4f}"
    )
    if confound["degenerate"]:
        print(f"[Chapter] 注意：{confound['note']}")

    # ---------- 1. 分层随机 holdout ----------
    train_idx, test_idx = split_indices(y, test_ratio=args.test_ratio, seed=args.seed)
    print(f"[Split] train={len(train_idx)} test={len(test_idx)} (test_ratio={args.test_ratio})")

    model = make_model(args.model)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    ci_low, ci_high = bootstrap_accuracy_ci(y[test_idx], pred, y[test_idx], seed=args.seed)

    holdout = {
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y[test_idx], pred, average="weighted", zero_division=0)
        ),
        "n_samples": int(len(test_idx)),
        "confusion_matrix": confusion_matrix(
            y[test_idx], pred, labels=list(range(n_runs))
        ).tolist(),
        "per_run": [
            {
                "run": int(run),
                "n_test": int((y[test_idx] == c).sum()),
                "recall": (
                    float(np.mean(pred[y[test_idx] == c] == c))
                    if np.any(y[test_idx] == c)
                    else None
                ),
            }
            for c, run in enumerate(run_values)
        ],
    }
    print(
        f"[{args.model}] test acc={holdout['accuracy']:.4f} "
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}] "
        f"macro_f1={holdout['macro_f1']:.4f} (n={holdout['n_samples']}, chance={chance:.4f})"
    )

    # ---------- 2. K 折交叉验证 ----------
    accs, f1s, oof = cross_validate(
        X, y, args.model, folds, args.seed, return_oof=True
    )
    cv_low, cv_high = bootstrap_accuracy_ci(y, oof, y, seed=args.seed)
    cv = {
        "folds": folds,
        "accuracy": [float(a) for a in accs],
        "macro_f1": [float(f) for f in f1s],
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "macro_f1_mean": float(f1s.mean()),
        "macro_f1_std": float(f1s.std()),
        "accuracy_ci_low": cv_low,
        "accuracy_ci_high": cv_high,
    }
    print(
        f"[CV] {folds} 折 acc={cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f} "
        f"95% CI [{cv_low:.4f}, {cv_high:.4f}] macro_f1={cv['macro_f1_mean']:.4f}"
    )

    # ---------- 3. 置换零分布 ----------
    permutation = None
    if args.permutations > 0:
        null = permutation_null(
            X, y, args.model, folds, args.seed, args.permutations, n_jobs=args.n_jobs
        )
        observed = cv["accuracy_mean"]
        permutation = {
            "n": args.permutations,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_p95": float(np.percentile(null, 95)),
            "observed": observed,
            "z_score": float((observed - null.mean()) / (null.std() + 1e-12)),
            "p_value": float((np.sum(null >= observed) + 1) / (len(null) + 1)),
            "null_samples": null.tolist(),
        }
        print(
            f"[置换] null={permutation['null_mean']:.4f} ± {permutation['null_std']:.4f} "
            f"p95={permutation['null_p95']:.4f} z={permutation['z_score']:.1f} "
            f"p={permutation['p_value']:.4f}"
        )

    # ---------- 4. 每个 run 的样本量 ----------
    counts = np.bincount(y, minlength=n_runs)
    per_run = []
    for c, run in enumerate(run_values):
        mask = y == c
        per_run.append(
            {
                "run": int(run),
                "n_total": int(counts[c]),
                "n_train": int(mask[train_idx].sum()),
                "n_test": int(mask[test_idx].sum()),
                "n_chapters": int(len(set(int(v) for v in chapters[mask]))),
                "holdout_recall": holdout["per_run"][c]["recall"],
            }
        )

    print("\n每个 run 的样本量与 holdout recall:")
    header = f"{'run':>4}{'n_total':>9}{'n_train':>9}{'n_test':>8}{'n_chap':>8}{'recall':>9}"
    print(header)
    print("-" * len(header))
    for row in per_run:
        recall = "-" if row["holdout_recall"] is None else f"{row['holdout_recall']:.4f}"
        print(
            f"{row['run']:>4}{row['n_total']:>9}{row['n_train']:>9}"
            f"{row['n_test']:>8}{row['n_chapters']:>8}{recall:>9}"
        )

    # ---------- 5. 落盘 ----------
    out_dir = out_root / f"{subject}_{novel}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "subject": subject,
        "novel": novel,
        "task": "run_identity",
        "label_source": "run",
        "chapter_used_for": "diagnostic_only",
        "n_samples": int(len(y)),
        "n_runs": n_runs,
        "n_chapters": confound["n_chapters"],
        "n_days": int(len(dataset.day_ids)),
        "n_features": int(X.shape[1]),
        "sfreq": float(dataset.sfreq),
        "nperseg": int(args.nperseg),
        "bands": {k: list(v) for k, v in BANDS.items()},
        "band_mode": args.bands_mode,
        "model": args.model,
        "normalize_epochs": False,
        "split": {
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "per_run": per_run,
        "holdout": holdout,
        "cv": cv,
        "permutation": permutation,
        "chapter_diagnostic": confound,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_rows(
        out_dir / "per_run.csv",
        per_run,
        ["run", "n_total", "n_train", "n_test", "n_chapters", "holdout_recall"],
    )
    write_confusion(
        out_dir / "confusion.csv", holdout["confusion_matrix"], run_values
    )
    write_rows(
        out_dir / "chapter_diagnostic.csv",
        [
            {
                "run": row["run"],
                "n_trials": row["n_trials"],
                "n_chapters": row["n_chapters"],
                "majority_chapter": row["majority_chapter"],
                "majority_share": row["majority_share"],
                "chapters": json.dumps(row["chapters"], ensure_ascii=False),
            }
            for row in confound["runs"]
        ],
        ["run", "n_trials", "n_chapters", "majority_chapter", "majority_share", "chapters"],
    )

    summary = {
        "subject": subject,
        "novel": novel,
        "n_samples": int(len(y)),
        "n_runs": n_runs,
        "n_chapters": confound["n_chapters"],
        "n_features": int(X.shape[1]),
        "sfreq": float(dataset.sfreq),
        "nperseg": int(args.nperseg),
        "band_mode": args.bands_mode,
        "model": args.model,
        "chance_accuracy": float(chance),
        "test_accuracy": holdout["accuracy"],
        "test_accuracy_ci_low": holdout["accuracy_ci_low"],
        "test_accuracy_ci_high": holdout["accuracy_ci_high"],
        "test_macro_f1": holdout["macro_f1"],
        "test_weighted_f1": holdout["weighted_f1"],
        "cv_accuracy_mean": cv["accuracy_mean"],
        "cv_accuracy_std": cv["accuracy_std"],
        "cv_accuracy_ci_low": cv["accuracy_ci_low"],
        "cv_accuracy_ci_high": cv["accuracy_ci_high"],
        "cv_macro_f1_mean": cv["macro_f1_mean"],
        "permutation_n": None if permutation is None else permutation["n"],
        "permutation_null_mean": None if permutation is None else permutation["null_mean"],
        "permutation_null_p95": None if permutation is None else permutation["null_p95"],
        "permutation_z": None if permutation is None else permutation["z_score"],
        "permutation_p": None if permutation is None else permutation["p_value"],
        "chapter_majority_baseline_accuracy": confound["majority_baseline_accuracy"],
        "chapter_random_baseline_accuracy": confound["random_baseline_accuracy"],
        "chapter_mean_majority_share": confound["mean_majority_share"],
        "chapter_nmi": confound["nmi"],
        "chapter_uncertainty_coefficient": confound[
            "uncertainty_coefficient_chapter_given_run"
        ],
        "chapter_cramers_v": confound["cramers_v"],
        "chapter_diagnostic_degenerate": confound["degenerate"],
    }
    write_rows(out_dir / "summary.csv", [summary], SUMMARY_FIELDS)
    print(f"[Saved] {out_dir}")
    return summary


# ------------------------------------------------------------------ 输出工具


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path: Path, matrix: list[list[int]], run_values: list[int]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + [f"run_{run:02d}" for run in run_values])
        for run, row in zip(run_values, matrix):
            writer.writerow([f"run_{run:02d}"] + row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="run 身份探针：逐通道 Welch 频带功率 + 收缩 LDA（chapter 仅作诊断）。"
    )
    parser.add_argument("--subject", default="sub-04")
    parser.add_argument(
        "--novel",
        nargs="+",
        default=["GarnettDream", "LittlePrince"],
        help="要跑的小说（会话）名",
    )
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument(
        "--n-chars-exact",
        type=int,
        default=10,
        help="只保留该字数的句子（默认 10，与去掉 mask 后的定长输入配套）",
    )
    parser.add_argument("--dedup", action="store_true", default=True)
    parser.add_argument("--no-dedup", dest="dedup", action="store_false")
    parser.add_argument("--preload", action="store_true", help="整 run 读入内存（更快要更多内存）")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=1, help="置换检验的并行进程数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="lda", choices=("lda", "logreg", "svm"))
    parser.add_argument("--bands-mode", default="rel", choices=("rel", "abs"))
    parser.add_argument("--nperseg", type=int, default=NPERSEG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"run_identity_{datetime.now():%Y%m%d_%H%M%S}"
    out_root = args.out_dir / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    rows = [
        run_subject(args, args.subject, novel, out_root)
        for novel in args.novel
    ]
    write_rows(out_root / "summary_all.csv", rows, SUMMARY_FIELDS)
    print(f"\n[Saved] {out_root / 'summary_all.csv'}")


if __name__ == "__main__":
    main()
