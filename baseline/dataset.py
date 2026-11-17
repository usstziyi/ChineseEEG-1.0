"""ChineseEEG 阅读任务的句子级 EEG Dataset（直接读 BrainVision，不落盘）。

每个样本对应 ``datasets/preprocessed/{subject}_{novel}.csv`` 的一行：以该行的
``sample`` 为起点，连续切出 ``n_chars`` 个 token，每个 token 固定
``TOKEN_SAMPLES = 100`` 个采样点（256 Hz 下约 0.39 s，对应一个汉字）。

``__getitem__`` 返回 ``(tokens, mask, sentence, run, chapter)``：

- ``tokens``：``torch.float32``，``(n_chars, 128, 100)``，单位**微伏（µV）**；
- ``mask``：``torch.bool``，``(n_chars,)``，全部为 ``True``（真实 token；
  batch 内的补齐在 ``pad_collate`` 阶段才产生）；
- ``sentence``：该行原文片段；
- ``run``：该句所属 EEG 采集轮次（``int``，对应 ``run-{run:02d}`` 文件）；
- ``chapter``：该句所属章节号（``int``，来自 CSV）。

``pad_collate`` 把若干样本拼成 ``(tokens, mask, sentences, runs, chapters)``：
tokens ``(B, 10, 128, 100)``、mask ``(B, 10)``、sentences ``list[str]``、
runs/chapters ``(B,)`` int64。

单位换算
--------
本 derivative 的 ``.vhdr`` 每个通道都写着 ``Ch*=E*, ,0.1,µV``，即物理单位是
µV、分辨率（Resolution）为 0.1 µV：二进制 float32 的裸数值乘 0.1 才是微伏
（实测裸值 std≈1332，对应 133 µV）。MNE 会按 Resolution × Unit 把数据换算成
**伏特（SI）**，所以 ``raw.get_data()`` 返回的是伏特；本模块再统一乘一次
``UV_PER_VOLT`` 还原成微伏：

- ``preload=True``：每个 run 首次访问时整段读入并换算一次，之后只做切片；
- ``preload=False``：每次切片时换算该片段。

两条路径都只换算一次，不会重复乘 ``1e6``。注意：不要绕过 MNE 直接
``np.fromfile``，那样拿到的是 0.1 µV 为单位的裸值，量级会差 10 倍。

坏导
----
按约定**忽略** ``*_bad_channels.json``：128 导全部保留，不做插值或剔除。

不落盘
------
本模块不写任何缓存文件。``preload=True`` 只是在内存里保留每个 run 的
``(128, n_times)`` float32 数组（单个 run 约 272 MB；GarnettDream 18 个 run
全部加载约 4.4 GB）；内存紧张时用 ``preload=False``，代价是按需从磁盘切片
（实测约 2 ms/样本）。

边界处理
--------
构造时对全部行做一次校验，任何异常都直接报错而不静默跳过：

- ``run`` 在 CSV 中存在但没有 ``.vhdr`` → ``FileNotFoundError``；
- ``n_chars`` 不在 ``[1, 10]``、``sample`` 为负或未严格递增、``sample + n_chars*100``
  超过文件长度、``onset`` 与 ``sample / sfreq`` 不一致 → 汇总成一条 ``ValueError``；
- 某个 run 有 EEG 但 CSV 里没有任何行 → ``RuntimeWarning``（不报错）。

用法::

    from baseline.dataset import ReadingEEGDataset, pad_collate

    ds = ReadingEEGDataset("sub-04", "GarnettDream")
    tokens, mask, sentence = ds[0]
    # tokens (n_chars, 128, 100) float32 µV, mask (n_chars,) bool

    loader = DataLoader(ds, batch_size=64, collate_fn=pad_collate, shuffle=True)
    tokens, mask, sentences = next(iter(loader))
"""

from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EEG_ROOT = PROJECT_ROOT / "datasets/ChineseEEG/derivatives/preproc/filtered_0.5_80"
CSV_ROOT = PROJECT_ROOT / "datasets/preprocessed"

TASK = "task-reading"

# 本数据集固定口径（已对 sub-04 全部 25 个 run 逐个头文件核实）
SFREQ = 256.0
N_CHANNELS = 128
CHANNEL_NAMES = tuple(f"E{i}" for i in range(1, N_CHANNELS + 1))

# 每个 token 的采样点数：100 点 ≈ 0.39 s
TOKEN_SAMPLES = 100
# n_chars 的实测上限（1~10），也是 batch 内 token 维度的补齐目标
MAX_TOKENS = 10
# MNE 返回伏特，乘以该系数转成微伏
UV_PER_VOLT = 1e6

# 报错时最多列出多少条明细，避免刷屏
MAX_REPORTED_ISSUES = 10

RUN_FILE_RE = re.compile(r"_run-(\d+)_eeg\.vhdr$")


class ReadingEEGDataset(Dataset):
    """(subject, novel) 一个阅读会话的句子级 EEG 数据集。

    Args:
        subject: 被试编号，如 ``sub-04``。
        novel: 小说名，如 ``GarnettDream``，对应 ``ses-<novel>``。
        min_chars: 只保留 ``n_chars >= min_chars`` 的行，默认 4（滤掉折行碎片）。
            传 0 会保留全部行，此时 ``n_chars == 0`` 的行会由校验显式报错。
        dedup: 是否按 ``sentence`` 字符串去重（保留首次出现），默认 True。
        eeg_root: 预处理 EEG 根目录。
        csv_root: ``extract_novel_text.py`` 输出的 CSV 目录。
        preload: True 时把用到的 run 整段读入内存（快，占内存）；
            False 时每次从磁盘切片（省内存，约 2 ms/样本）。

    Attributes:
        rows: 过滤后的 CSV 行（``dict``，保留原始字符串值，另加 ``_line``
            记录其在 CSV 中的行号），供训练脚本按句子切分使用。
        stats: 过滤统计（``n_rows`` / ``n_dropped_min_chars`` / ``n_dropped_dup``）。
    """

    def __init__(
        self,
        subject: str,
        novel: str,
        min_chars: int = 4,
        dedup: bool = True,
        eeg_root: Path = EEG_ROOT,
        csv_root: Path = CSV_ROOT,
        preload: bool = True,
    ):
        self.subject = subject
        self.novel = novel
        self.session = f"ses-{novel}"
        self.min_chars = min_chars
        self.dedup = dedup
        self.preload = preload

        self.eeg_dir = Path(eeg_root) / subject / self.session / "eeg"
        self.csv_path = Path(csv_root) / f"{subject}_{novel}.csv"

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"找不到 CSV: {self.csv_path}")
        if not self.eeg_dir.is_dir():
            raise FileNotFoundError(f"找不到 EEG 目录: {self.eeg_dir}")

        # run -> n_times，校验时填写
        self._n_times: dict[int, int] = {}
        # preload=True: run -> (128, n_times) float32 µV
        self._signals: dict[int, np.ndarray] = {}
        # preload=False: run -> mne Raw（preload=False，文件句柄常驻）
        self._raws: dict[int, mne.io.BaseRaw] = {}

        raw_rows = self._read_csv()
        self.rows = self._filter_rows(raw_rows)
        if not self.rows:
            raise ValueError(
                f"{subject}/{novel}: 过滤后没有样本"
                f"（min_chars={min_chars}, dedup={dedup}）"
            )
        self._check_runs_and_rows()

    # ------------------------------------------------------------------ 读取

    def _read_csv(self) -> list[dict]:
        """读取 CSV，给每行补一个 ``_line``（CSV 中的 1 基行号，表头为第 1 行）。"""
        with self.csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for line, row in enumerate(reader, start=2):
                row["_line"] = line
                rows.append(row)
        return rows

    def _filter_rows(self, raw_rows: list[dict]) -> list[dict]:
        """按 min_chars 过滤、按句子去重，并记录统计信息。"""
        rows: list[dict] = []
        seen: set[str] = set()
        dropped_short = 0
        dropped_dup = 0
        for row in raw_rows:
            if int(row["n_chars"]) < self.min_chars:
                dropped_short += 1
                continue
            if self.dedup and row["sentence"] in seen:
                dropped_dup += 1
                continue
            seen.add(row["sentence"])
            rows.append(row)

        self.stats = {
            "n_rows": len(rows),
            "n_dropped_min_chars": dropped_short,
            "n_dropped_dup": dropped_dup,
        }
        return rows

    def _discover_run_files(self) -> dict[int, Path]:
        """扫描会话目录，返回 ``{run: vhdr 路径}``。"""
        run_files = {}
        for path in self.eeg_dir.glob(f"*_{TASK}_run-*_eeg.vhdr"):
            matched = RUN_FILE_RE.search(path.name)
            if matched:
                run_files[int(matched.group(1))] = path
        if not run_files:
            raise FileNotFoundError(f"{self.eeg_dir} 下没有任何 *_{TASK}_run-*_eeg.vhdr")
        return run_files

    def _vhdr_path(self, run: int) -> Path:
        return (
            self.eeg_dir
            / f"{self.subject}_{self.session}_{TASK}_run-{run:02d}_eeg.vhdr"
        )

    # ------------------------------------------------------------------ 校验

    def _check_runs_and_rows(self) -> None:
        """校验 run 覆盖、头文件规格和每一行的采样边界；有问题一次性抛出。"""
        run_files = self._discover_run_files()
        rows_by_run: dict[int, list[dict]] = {}
        for row in self.rows:
            rows_by_run.setdefault(int(row["run"]), []).append(row)

        missing = sorted(set(rows_by_run) - set(run_files))
        if missing:
            raise FileNotFoundError(
                f"{self.subject}/{self.novel}: CSV 引用了不存在的 run {missing}；"
                f"目录 {self.eeg_dir} 中实际有 run {sorted(run_files)}"
            )
        extra = sorted(set(run_files) - set(rows_by_run))
        if extra:
            warnings.warn(
                f"{self.subject}/{self.novel}: run {extra} 有 EEG 文件但 CSV 中没有对应行，"
                "这些 run 不会被使用",
                RuntimeWarning,
                stacklevel=2,
            )

        issues: list[str] = []
        for run in sorted(rows_by_run):
            vhdr = run_files[run]
            raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
            try:
                if raw.info["sfreq"] != SFREQ:
                    issues.append(
                        f"run {run:02d}: 采样率 {raw.info['sfreq']} != {SFREQ}"
                    )
                if raw.info["nchan"] != N_CHANNELS:
                    issues.append(
                        f"run {run:02d}: 通道数 {raw.info['nchan']} != {N_CHANNELS}"
                    )
                if tuple(raw.ch_names) != CHANNEL_NAMES:
                    issues.append(
                        f"run {run:02d}: 通道名/顺序不是 E1..E{N_CHANNELS}"
                    )
                if raw.first_samp != 0:
                    issues.append(
                        f"run {run:02d}: first_samp={raw.first_samp}，"
                        "事件 sample 需要平移"
                    )

                n_times = raw.n_times
                self._n_times[run] = n_times
                last_sample = -1
                for row in rows_by_run[run]:
                    n_chars = int(row["n_chars"])
                    sample = int(row["sample"])
                    onset = float(row["onset"])
                    where = f"{self.csv_path.name}:{row['_line']} (run {run:02d})"

                    if not 1 <= n_chars <= MAX_TOKENS:
                        issues.append(
                            f"{where}: n_chars={n_chars} 超出 [1, {MAX_TOKENS}]"
                        )
                    elif sample < 0:
                        issues.append(f"{where}: sample={sample} 为负")
                    elif sample + n_chars * TOKEN_SAMPLES > n_times:
                        issues.append(
                            f"{where}: sample={sample} 起需要 "
                            f"{n_chars * TOKEN_SAMPLES} 点，超出 run 长度 {n_times}"
                        )

                    if sample <= last_sample:
                        issues.append(
                            f"{where}: sample={sample} 未严格递增"
                            f"（上一句 {last_sample}）"
                        )
                    last_sample = sample

                    if abs(onset * SFREQ - sample) > 1e-3:
                        issues.append(
                            f"{where}: onset={onset} 与 sample/sfreq="
                            f"{sample / SFREQ:.9f} 不一致"
                        )
            finally:
                raw.close()

        if issues:
            shown = "\n".join(issues[:MAX_REPORTED_ISSUES])
            more = (
                f"\n... 另有 {len(issues) - MAX_REPORTED_ISSUES} 条问题未列出"
                if len(issues) > MAX_REPORTED_ISSUES
                else ""
            )
            raise ValueError(
                f"{self.subject}/{self.novel}: 数据校验失败，共 {len(issues)} 处\n"
                f"{shown}{more}"
            )

    # ------------------------------------------------------------- 信号读取

    def _open_raw(self, run: int) -> mne.io.BaseRaw:
        raw = self._raws.get(run)
        if raw is not None:
            return raw
        path = self._vhdr_path(run)
        if not path.is_file():
            raise FileNotFoundError(f"run {run:02d} 的 vhdr 不存在: {path}")
        raw = mne.io.read_raw_brainvision(
            path, preload=not self.preload, verbose="ERROR"
        )
        self._raws[run] = raw
        return raw

    def _signal(self, run: int) -> np.ndarray:
        """preload=True 路径：整段读入并换算一次，返回 (128, n_times) µV。"""
        signal = self._signals.get(run)
        if signal is not None:
            return signal
        raw = self._open_raw(run)
        # 唯一一次单位换算：伏特 -> 微伏
        signal = (raw.get_data() * UV_PER_VOLT).astype(np.float32)
        raw.close()
        self._raws.pop(run, None)
        self._signals[run] = signal
        return signal

    def _segment(self, run: int, start: int, stop: int) -> np.ndarray:
        """取 ``[start, stop)`` 的 (128, stop-start) 微伏片段。"""
        if self.preload:
            return self._signal(run)[:, start:stop]
        raw = self._open_raw(run)
        segment = raw.get_data(start=start, stop=stop)
        # 唯一一次单位换算：伏特 -> 微伏
        return (segment * UV_PER_VOLT).astype(np.float32)

    # --------------------------------------------------------------- Dataset

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def n_chans(self) -> int:
        return N_CHANNELS

    @property
    def sfreq(self) -> float:
        return SFREQ

    @property
    def channel_names(self) -> list[str]:
        return list(CHANNEL_NAMES)

    def __getitem__(self, index: int):
        row = self.rows[index]
        run = int(row["run"])
        start = int(row["sample"])
        n_chars = int(row["n_chars"])
        stop = start + n_chars * TOKEN_SAMPLES

        segment = self._segment(run, start, stop)
        expected = (N_CHANNELS, n_chars * TOKEN_SAMPLES)
        if segment.shape != expected:
            raise RuntimeError(
                f"切片形状异常: {segment.shape} != {expected}"
                f"（{self.csv_path.name}:{row['_line']}）"
            )

        # (C, n, T) -> (n, C, T)
        tokens = segment.reshape(N_CHANNELS, n_chars, TOKEN_SAMPLES).transpose(1, 0, 2)
        return (
            torch.from_numpy(np.ascontiguousarray(tokens)),
            torch.ones(n_chars, dtype=torch.bool),
            str(row["sentence"]),
            run,
            int(row["chapter"]),
        )

    def close(self) -> None:
        """释放文件句柄与内存中的信号。"""
        for raw in self._raws.values():
            raw.close()
        self._raws.clear()
        self._signals.clear()


def pad_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, str, int, int]]
):
    """把 ``(tokens, mask, sentence, run, chapter)`` 列表拼成定长 batch。

    句子长度（``n_chars``）在 1~10 之间浮动，这里统一按右侧补零对齐到
    ``MAX_TOKENS``。

    Returns:
        ``(tokens, mask, sentences, runs, chapters)``

        - ``tokens``：``(B, MAX_TOKENS, 128, TOKEN_SAMPLES)``，float32，µV
        - ``mask``：``(B, MAX_TOKENS)``，bool，True 表示真实 token
        - ``sentences``：长度 ``B`` 的 ``list[str]``
        - ``runs`` / ``chapters``：``(B,)`` int64
    """
    n_samples = len(batch)
    n_channels = batch[0][0].shape[1]

    tokens = torch.zeros(
        n_samples, MAX_TOKENS, n_channels, TOKEN_SAMPLES, dtype=torch.float32
    )
    mask = torch.zeros(n_samples, MAX_TOKENS, dtype=torch.bool)
    sentences: list[str] = []
    runs = torch.empty(n_samples, dtype=torch.long)
    chapters = torch.empty(n_samples, dtype=torch.long)

    for i, (item, item_mask, sentence, run, chapter) in enumerate(batch):
        length = item.shape[0]
        if length > MAX_TOKENS:
            raise ValueError(f"样本 token 数 {length} 超过 MAX_TOKENS={MAX_TOKENS}")
        if item.shape[1] != n_channels:
            raise ValueError(
                f"batch 内通道数不一致: {item.shape[1]} != {n_channels}"
            )
        tokens[i, :length] = item
        mask[i, :length] = item_mask
        sentences.append(sentence)
        runs[i] = run
        chapters[i] = chapter

    return tokens, mask, sentences, runs, chapters


def main() -> None:
    """冒烟检查：遍历 CSV 建数据集，打印规模，并跑一个 batch。"""
    datasets: list[ReadingEEGDataset] = []
    for csv_path in sorted(CSV_ROOT.glob("*.csv")):
        subject, novel = csv_path.stem.split("_", 1)
        dataset = ReadingEEGDataset(subject, novel)
        datasets.append(dataset)

        tokens, mask, sentence, run, chapter = dataset[0]
        print(
            f"{subject} {novel:<14} {len(dataset):>5} 条 | "
            f"首条 {tuple(tokens.shape)} {tokens.dtype} | "
            f"run={run} chapter={chapter} | "
            f"µV 范围 [{tokens.min():.1f}, {tokens.max():.1f}] | {sentence!r}"
        )
        print(f"  过滤统计: {dataset.stats}")

    first = datasets[0]
    loader = DataLoader(first, batch_size=4, collate_fn=pad_collate)
    tokens, mask, sentences, runs, chapters = next(iter(loader))
    print(f"\n{first.subject} {first.novel} 取一个 batch (batch_size=4):")
    print(f"  tokens {tuple(tokens.shape)} {tokens.dtype}")
    print(f"  mask   {tuple(mask.shape)} {mask.dtype} -> 每条真实 token 数 {mask.sum(dim=1).tolist()}")
    print(f"  run    {runs.tolist()}  chapter {chapters.tolist()}")
    print(f"  text   {sentences}")

    for dataset in datasets:
        dataset.close()


if __name__ == "__main__":
    main()
