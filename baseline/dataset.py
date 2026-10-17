"""ChineseEEG 阅读任务的句子级 EEG Dataset。

每个样本对应 CSV 里的一行（一个折行片段）：以该行的 ``sample`` 为起点，
按固定 token 长度连续切出 ``n_chars`` 个 token。

``__getitem__`` 返回 ``(tokens, sentence)``：

- ``tokens``：``torch.float32``，(n_tokens, C, TOKEN_SAMPLES)，``C = 128``
- ``sentence``：该行原文片段

采样值单位：MNE 默认返回**伏特（V）**，本 Dataset 读入后统一乘 ``UV_PER_VOLT``
转成**微伏（µV）**再交给模型。
"""

from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EEG_ROOT = PROJECT_ROOT / "datasets/ChineseEEG/derivatives/preproc/filtered_0.5_30"
CSV_ROOT = PROJECT_ROOT / "datasets/preprocessed/filtered_0.5_30"

SFREQ = 256
TASK = "task-reading"

# MNE 读出的数据单位是伏特（V），乘以该系数转成微伏（µV）
UV_PER_VOLT = 1e6

# 每个 token 的采样点数
TOKEN_SAMPLES = 100
# n_chars 的实测上限（1~10），也是 batch 里 token 维度补齐的目标长度
MAX_TOKENS = 10


class ReadingEEGDataset(Dataset):
    """(subject, novel) 一个 run 集合的句子级数据集。

    Args:
        subject: 被试编号，如 ``sub-04``。
        novel: 小说名，如 ``LittlePrince``，对应 ``ses-<novel>`` 会话。
        min_chars: 只保留 ``n_chars >= min_chars`` 的句子，默认 4（滤掉折行碎片）。
            传 0 可关闭过滤。
        dedup: 是否按 ``sentence`` 字符串去重（保留首次出现），默认 True。
        eeg_root: 预处理 EEG 根目录。
        csv_root: ``extract_novel_text.py`` 输出的 CSV 目录。

    注意：过滤后 ``row["index"]`` 仍是 CSV 里的原始行号，不与样本下标对应。
    """

    def __init__(
        self,
        subject: str,
        novel: str,
        min_chars: int = 4,
        dedup: bool = True,
        eeg_root: Path = EEG_ROOT,
        csv_root: Path = CSV_ROOT,
    ):
        self.subject = subject
        self.novel = novel
        self.min_chars = min_chars
        self.dedup = dedup
        self.session = f"ses-{novel}"
        self.eeg_dir = Path(eeg_root) / subject / self.session / "eeg"

        csv_path = Path(csv_root) / f"{subject}_{novel}.csv"
        rows: list[dict] = []
        seen: set[str] = set()
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["n_chars"]) < min_chars:
                    continue
                if dedup and row["sentence"] in seen:
                    continue
                seen.add(row["sentence"])
                rows.append(row)
        self.rows = rows

        # run 用惰性加载 + 缓存：preload=False 时 MNE 用 memmap，切片只读磁盘对应片段
        self._raws: dict[int, mne.io.BaseRaw] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _raw(self, run: int) -> mne.io.BaseRaw:
        if run not in self._raws:
            vhdr = self.eeg_dir / f"{self.subject}_{self.session}_{TASK}_run-{run:02d}_eeg.vhdr"
            self._raws[run] = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
        return self._raws[run]

    def __getitem__(self, index: int):
        row = self.rows[index]
        run = int(row["run"])
        start = int(row["sample"])
        n_tokens = int(row["n_chars"])

        stop = start + n_tokens * TOKEN_SAMPLES
        data = self._raw(run).get_data(start=start, stop=stop)  # (C, n_tokens * TOKEN_SAMPLES)
        data = data * UV_PER_VOLT  # MNE 默认返回伏特（V），这里转成微伏（µV）

        tokens = data.reshape(data.shape[0], n_tokens, TOKEN_SAMPLES).transpose(1, 0, 2)
        return torch.from_numpy(tokens.astype(np.float32)), str(row["sentence"])


def pad_collate(batch: list[tuple[torch.Tensor, str]]):
    """把 ``(tokens, sentence)`` 列表拼成定长 batch，供 ``DataLoader`` 使用。

    ``n_tokens`` 逐句不同（实测取值 1~10），这里统一补齐到 ``MAX_TOKENS``，
    不足的样本在 token 维度**右侧补零**。

    Returns:
        ``(tokens, mask, sentences)``

        - ``tokens``：``(B, MAX_TOKENS, C, TOKEN_SAMPLES)``，float32
        - ``mask``：``(B, MAX_TOKENS)``，bool，``True`` 表示该位置是真实 token（非补零）
        - ``sentences``：长度 ``B`` 的 ``list[str]``
    """
    n_samples = len(batch)
    n_channels = batch[0][0].shape[1]

    tokens = torch.zeros(n_samples, MAX_TOKENS, n_channels, TOKEN_SAMPLES, dtype=torch.float32)
    mask = torch.zeros(n_samples, MAX_TOKENS, dtype=torch.bool)
    sentences: list[str] = []

    for i, (item, sentence) in enumerate(batch):
        length = item.shape[0]
        tokens[i, :length] = item
        mask[i, :length] = True
        sentences.append(sentence)

    return tokens, mask, sentences


def main() -> None:
    """冒烟检查：遍历所有 CSV 建立 dataset，打印规模，并跑一个 batch。"""
    datasets: list[ReadingEEGDataset] = []
    for csv_path in sorted(CSV_ROOT.glob("*.csv")):
        subject, novel = csv_path.stem.split("_", 1)
        dataset = ReadingEEGDataset(subject, novel)
        datasets.append(dataset)

        tokens, sentence = dataset[0]
        print(f"{subject} {novel:<14} {len(dataset):>5} 条 | 首条 {tuple(tokens.shape)} {tokens.dtype} | {sentence!r}")
    first = datasets[0]
    loader = DataLoader(first, batch_size=4, collate_fn=pad_collate)
    tokens, mask, sentences = next(iter(loader))
    print(f"\n{first.subject} {first.novel} 取一个 batch (batch_size=4):")
    print(f"  tokens {tuple(tokens.shape)} {tokens.dtype}")
    print(f"  mask   {tuple(mask.shape)} {mask.dtype} -> 每条真实 token 数 {mask.sum(dim=1).tolist()}")
    print(f"  text   {sentences}")


if __name__ == "__main__":
    main()
