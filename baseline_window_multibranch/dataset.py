"""滑窗切分的句子级 EEG–文本 Dataset（``baseline_window`` 的自包含副本）。

与 ``baseline_window/dataset.py`` **逐行相同**：多分支改动只发生在模型的时间前端
（``model.py``），切分策略、pad_collate、单位换算都要保持一致，才能把指标差异
干净地归因到"多分支时间滤波"这一个变量上。

- **baseline**：以该行的 ``sample`` 为起点，按固定 100 点连续切出 ``n_chars`` 个
  字 token。窗首恰好等于字首，模型可以靠"位置 = 字位置"这条捷径偷分。
- **本模块**：把整句 EEG 用**固定长度、固定步长的滑动窗口**切开，得到等长窗口序列：

      (C, n_chars × 100)
          --window = 1.0 s (256 点), stride = 0.5 s (128 点)-->
      (n_windows, C, 256)

  窗口边界与字边界**不对齐**（128 不是 100 的整数倍），相邻窗口还有 50% 的重叠。

这样"变长性"从"每个 token 的采样点数"转移到了"窗口个数"：每个 token 形状恒定
（可直接复用同一套 encoder），句子之间的长度差异由 collate 在 batch 内 padding 吸收。

窗口数按下式计算，末尾不足一个完整窗口的部分丢弃：

    n_windows = floor((n_chars × 100 - window_samples) / stride_samples) + 1

默认参数（256 Hz 下 window = 256 / stride = 128）下，``min_chars=4`` 时
``n_windows`` 的取值范围是 2（n_chars=4）~ 6（n_chars=10）。

用法::

    from baseline_window_multibranch.dataset import ReadingEEGDataset, pad_collate

    ds = ReadingEEGDataset("sub-04", "LittlePrince")
    tokens, sentence = ds[0]        # tokens: (n_windows, C, 256) float32

    loader = DataLoader(ds, batch_size=64, collate_fn=pad_collate)
    tokens, mask, sentences = next(iter(loader))
    # tokens: (B, n_max, C, 256)；mask: (B, n_max) bool，True 表示有效窗口

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
CSV_ROOT = PROJECT_ROOT / "datasets/preprocessed/novels"

SFREQ = 256
TASK = "task-reading"

# MNE 读出的数据单位是伏特（V），乘以该系数转成微伏（µV）
UV_PER_VOLT = 1e6

# 每个字占用的采样点数：整句 EEG 长度就是 n_chars × CHAR_SAMPLES
CHAR_SAMPLES = 100

# 滑动窗口长度与步长（秒）；256 Hz 下默认对应 256 / 128 个采样点
WINDOW_DURATION = 1.0
STRIDE_DURATION = 0.5


def window_count(span_samples: int, window_samples: int, stride_samples: int) -> int:
    """整段长度 ``span_samples`` 能切出多少个完整窗口（末尾余量丢弃）。"""
    if span_samples < window_samples:
        return 0
    return (span_samples - window_samples) // stride_samples + 1


class ReadingEEGDataset(Dataset):
    """(subject, novel) 一个 run 集合的滑窗级数据集。

    Args:
        subject: 被试编号，如 ``sub-04``。
        novel: 小说名，如 ``LittlePrince``，对应 ``ses-<novel>`` 会话。
        min_chars: 只保留 ``n_chars >= min_chars`` 的句子，默认 4（滤掉折行碎片）。
            传 0 可关闭过滤。
        dedup: 是否按 ``sentence`` 字符串去重（保留首次出现），默认 True。
        window_duration: 滑动窗口长度（秒），默认 1.0；乘 ``SFREQ`` 得到窗口采样点数。
        stride_duration: 相邻窗口起点的间隔（秒），默认 0.5。小于窗口长度时窗口重叠。
        eeg_root: 预处理 EEG 根目录。
        csv_root: ``extract_novel_text.py`` 输出的 CSV 目录。

    注意：
        - 过滤后 ``row["index"]`` 仍是 CSV 里的原始行号，不与样本下标对应；
        - 整句长度不足一个完整窗口的句子会被丢弃，数量见 ``n_too_short``。

    Attributes:
        window_samples: 每个窗口的采样点数（默认 256 Hz × 1.0 s = 256）。
        stride_samples: 相邻窗口起点的采样点间隔（默认 256 Hz × 0.5 s = 128）。
    """

    def __init__(
        self,
        subject: str,
        novel: str,
        min_chars: int = 4,
        dedup: bool = True,
        window_duration: float = WINDOW_DURATION,
        stride_duration: float = STRIDE_DURATION,
        eeg_root: Path = EEG_ROOT,
        csv_root: Path = CSV_ROOT,
    ):
        if window_duration <= 0 or stride_duration <= 0:
            raise ValueError(
                f"window_duration={window_duration}, stride_duration={stride_duration} 必须为正"
            )

        self.subject = subject
        self.novel = novel
        self.min_chars = min_chars
        self.dedup = dedup
        self.window_duration = float(window_duration)
        self.stride_duration = float(stride_duration)
        self.window_samples = int(round(self.window_duration * SFREQ))
        self.stride_samples = int(round(self.stride_duration * SFREQ))
        if self.window_samples <= 0 or self.stride_samples <= 0:
            raise ValueError(
                f"window_duration={window_duration}, stride_duration={stride_duration} "
                f"在 {SFREQ} Hz 下算不出有效窗口长度"
            )

        self.session = f"ses-{novel}"
        self.eeg_dir = Path(eeg_root) / subject / self.session / "eeg"

        csv_path = Path(csv_root) / f"{subject}_{novel}.csv"
        rows: list[dict] = []
        seen: set[str] = set()
        # 段长不足一个完整窗口的句子一个 token 也切不出来，直接丢弃并计数
        n_too_short = 0
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["n_chars"]) < min_chars:
                    continue
                if int(row["n_chars"]) * CHAR_SAMPLES < self.window_samples:
                    n_too_short += 1
                    continue
                if dedup and row["sentence"] in seen:
                    continue
                seen.add(row["sentence"])
                rows.append(row)
        self.rows = rows
        self.n_too_short = n_too_short

        # run 用惰性加载 + 缓存：preload=False 时 MNE 用 memmap，切片只读磁盘对应片段
        self._raws: dict[int, mne.io.BaseRaw] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def span_samples(self, row: dict) -> int:
        """整句 EEG 的采样点数：``n_chars × 100``。"""
        return int(row["n_chars"]) * CHAR_SAMPLES

    def n_windows(self, row: dict) -> int:
        """该行按当前窗长 / 步长能切出的窗口数。"""
        return window_count(self.span_samples(row), self.window_samples, self.stride_samples)

    def _raw(self, run: int) -> mne.io.BaseRaw:
        if run not in self._raws:
            vhdr = self.eeg_dir / f"{self.subject}_{self.session}_{TASK}_run-{run:02d}_eeg.vhdr"
            self._raws[run] = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
        return self._raws[run]

    def __getitem__(self, index: int):
        row = self.rows[index]
        run = int(row["run"])
        start = int(row["sample"])
        span = self.span_samples(row)

        stop = start + span
        # 一次性读出整段 (C, T)，再按 stride 展开成 (n_windows, C, window_samples)
        data = self._raw(run).get_data(start=start, stop=stop)
        data = data * UV_PER_VOLT  # MNE 默认返回伏特（V），这里转成微伏（µV）

        # unfold 的窗口起点间隔即 stride_samples，末尾不足一个窗口的部分自动丢弃
        tokens = torch.from_numpy(data.astype(np.float32))
        tokens = tokens.unfold(-1, self.window_samples, self.stride_samples)
        tokens = tokens.permute(1, 0, 2).contiguous()  # (n_windows, C, window_samples)

        return tokens, str(row["sentence"])


def pad_collate(batch: list[tuple[torch.Tensor, str]]):
    """把 ``(tokens, sentence)`` 列表拼成定长 batch，供 ``DataLoader`` 使用。

    与 baseline 的 ``pad_collate`` 相比，补齐目标不再是常量 ``MAX_TOKENS``，而是
    本 batch 内最长的窗口数 ``n_max``：每个 token 形状已经统一为
    ``(C, window_samples)``，需要 padding 的只有"窗口个数"这一维，代价极小。

    Args:
        batch: ``[(tokens_i, sentence_i), ...]``，``tokens_i`` 形状
            ``(n_i, C, window_samples)``。

    Returns:
        ``(tokens, mask, sentences)``

        - ``tokens``：``(B, n_max, C, window_samples)``，float32，pad 位置填 0
        - ``mask``：``(B, n_max)``，bool，``True`` 表示该位置是真实窗口（非补零）
        - ``sentences``：长度 ``B`` 的 ``list[str]``
    """
    tokens_list, sentences = zip(*batch)
    n_samples = len(tokens_list)
    n_max = max(int(item.shape[0]) for item in tokens_list)
    n_channels, window_samples = tokens_list[0].shape[1], tokens_list[0].shape[2]

    tokens = tokens_list[0].new_zeros(n_samples, n_max, n_channels, window_samples)
    mask = torch.zeros(n_samples, n_max, dtype=torch.bool)

    for i, item in enumerate(tokens_list):
        length = int(item.shape[0])
        tokens[i, :length] = item
        mask[i, :length] = True

    return tokens, mask, list(sentences)


def main() -> None:
    """冒烟检查：遍历所有 CSV 建立 dataset，打印规模，并跑一个 batch。"""
    datasets: list[ReadingEEGDataset] = []
    for csv_path in sorted(CSV_ROOT.glob("*.csv")):
        subject, novel = csv_path.stem.split("_", 1)
        dataset = ReadingEEGDataset(subject, novel)
        datasets.append(dataset)

        counts = [dataset.n_windows(row) for row in dataset.rows]
        tokens, sentence = dataset[0]
        print(
            f"{subject} {novel:<14} {len(dataset):>5} 条 | "
            f"窗口 {min(counts)}~{max(counts)} 个 | "
            f"首条 {tuple(tokens.shape)} {tokens.dtype} | {sentence!r}"
        )
        if dataset.n_too_short:
            print(f"    段长不足一个窗口被丢弃 {dataset.n_too_short} 条")

    first = datasets[0]
    loader = DataLoader(first, batch_size=4, collate_fn=pad_collate)
    tokens, mask, sentences = next(iter(loader))
    print(f"\n{first.subject} {first.novel} 取一个 batch (batch_size=4):")
    print(f"  window_samples={first.window_samples} stride_samples={first.stride_samples}")
    print(f"  tokens {tuple(tokens.shape)} {tokens.dtype}")
    print(f"  mask   {tuple(mask.shape)} {mask.dtype} -> 每条真实窗口数 {mask.sum(dim=1).tolist()}")
    print(f"  text   {sentences}")


if __name__ == "__main__":
    main()
