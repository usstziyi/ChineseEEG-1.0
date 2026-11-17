"""跨实验共享的数据视图与分组评估工具。

三个移植实验（句类基线 / run 身份 / LORO-LODO）共用同一套：

- ``extract_labeled_features``：从 ``ReadingEEGDataset`` 取 EEG 并计算
  逐通道 Welch 频带 log 功率，同时返回可配置的标签（``run`` / ``chapter`` /
  ``day``）与 ``run`` / ``chapter`` / ``day`` 分组向量；
- ``stratified_subsample``：按标签分层抽取小样本，用于烟雾测试；
- ``group_oof_predict`` / ``per_group_accuracy``：留一组（leave-one-group-out）
  的折外预测与逐组准确率。

评估口径（holdout 划分、K 折、置换、按组 bootstrap）统一复用
``baseline.run_identity`` 中的实现，避免三套代码漂移。
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from baseline.dataset import N_CHANNELS, SAMPLES_PER_CHAR, ReadingEEGDataset
from baseline.model import BANDS, NPERSEG, extract_features, make_model

LABEL_SOURCES = ("run", "chapter", "day")


def extract_labeled_features(
    dataset: ReadingEEGDataset,
    label_source: str = "chapter",
    mode: str = "rel",
    nperseg: int = NPERSEG,
    batch_size: int = 128,
    max_samples: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """返回特征矩阵与标签/分组。

    Args:
        label_source: ``run`` / ``chapter`` / ``day``，决定 ``labels`` 的来源；
            ``runs`` / ``chapters`` / ``days`` 始终原样返回，供分组评估使用。
        max_samples: 分层抽取的样本上限（``None`` 表示全量），用于烟雾测试；
            抽取只影响进入特征的样本，不影响标签映射。
    """
    if label_source not in LABEL_SOURCES:
        raise ValueError(f"label_source={label_source!r}，可选 {LABEL_SOURCES}")

    rows = dataset.rows
    if not rows:
        raise ValueError("数据集为空，无法提取特征")
    lengths = {int(row["n_chars"]) for row in rows}
    if len(lengths) != 1:
        raise ValueError(f"样本长度不一致 {sorted(lengths)}，请用 n_chars_exact 固定字数")
    length = next(iter(lengths)) * SAMPLES_PER_CHAR

    raw_labels = np.array([int(row[label_source]) for row in rows], dtype=np.int64)
    label_values = sorted(set(raw_labels.tolist()))
    mapping = {value: index for index, value in enumerate(label_values)}
    labels = np.array([mapping[int(v)] for v in raw_labels], dtype=np.int64)

    runs = np.array([int(row["run"]) for row in rows], dtype=np.int64)
    chapters = np.array([int(row["chapter"]) for row in rows], dtype=np.int64)
    days = np.array([int(row["day"]) for row in rows], dtype=np.int64)

    index = stratified_subsample(labels, max_samples, seed=seed)
    if verbose and len(index) < len(rows):
        print(f"  [抽样] {len(rows)} -> {len(index)} 条（按 {label_source} 分层）")

    n_features = len(BANDS) * N_CHANNELS
    features = np.empty((len(index), n_features), dtype=np.float32)
    for start in range(0, len(index), batch_size):
        batch_indices = index[start : start + batch_size]
        block = np.empty((len(batch_indices), N_CHANNELS, length), dtype=np.float32)
        for j, row_index in enumerate(batch_indices):
            eeg, _, _, _, _ = dataset[row_index]
            block[j] = np.asarray(eeg)
        features[start : start + len(batch_indices)] = extract_features(
            block, dataset.sfreq, mode=mode, nperseg=nperseg
        )

    if verbose:
        counts = np.bincount(labels[index], minlength=len(label_values))
        print(
            f"  [特征] {len(index)} 条 × {n_features} 维 | T={length} | "
            f"{label_source} 类别 {len(label_values)} | 最少 {counts.min()} 条/类"
        )

    return {
        "features": features,
        "labels": labels[index],
        "label_values": label_values,
        "label_source": label_source,
        "runs": runs[index],
        "chapters": chapters[index],
        "days": days[index],
        "row_index": index,
        "n_features": n_features,
    }


def stratified_subsample(
    labels: np.ndarray,
    max_samples: int | None,
    seed: int = 42,
) -> np.ndarray:
    """按标签分层，每类最多取 ``max_samples // 类别数`` 条。"""
    labels = np.asarray(labels)
    if max_samples is None or max_samples >= len(labels):
        return np.arange(len(labels))

    unique = np.unique(labels)
    per_class = max(1, int(max_samples) // len(unique))
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for label in unique:
        index = np.flatnonzero(labels == label)
        rng.shuffle(index)
        parts.append(index[:per_class])
    return np.sort(np.concatenate(parts))


def group_oof_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str = "lda",
) -> np.ndarray:
    """留一组的折外预测：每个组整组留出，训练集里只出现其他组。"""
    groups = np.asarray(groups)
    pred = np.empty(len(y), dtype=np.asarray(y).dtype)
    splitter = LeaveOneGroupOut()
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = make_model(model_name)
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
    return pred


def per_group_accuracy(
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
) -> dict[int, float]:
    """逐组准确率（组内样本的平均命中）。"""
    groups = np.asarray(groups)
    return {
        int(group): float(np.mean(pred[groups == group] == y[groups == group]))
        for group in np.unique(groups)
    }
