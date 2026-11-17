"""线性基线的特征前端与分类器（移植自 ``new_model/plan_run_identity/model.py``）。

- ``extract_features``：逐通道 Welch 频带功率（相对 / 绝对），得到
  ``(N, 通道数 × 频段数)`` 的 log 功率特征；
- ``make_model``：在标准化特征上拟合收缩 LDA（主基线）或逻辑回归 / RBF SVM。

与原实现的差别只有采样率相关的参数：原数据 500 Hz、``nperseg=256``
（0.512 s），本工程 256 Hz，因此 ``nperseg=128``（同样约 0.5 s），
频率分辨率为 2 Hz。频段定义、特征拼接顺序 ``[频段][通道]``、
分类器超参都与原实现保持一致。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# 0.5 s @256 Hz；原实现是 0.512 s @500 Hz
NPERSEG = 128


def band_log_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    mode: str = "rel",
) -> np.ndarray:
    """把 ``(N, C, F)`` 的 PSD 压成 ``(N, 频段数 × C)`` 的 log 功率。

    Args:
        psd: Welch 输出的功率谱密度，单位 µV²/Hz。
        freqs: 与 ``psd`` 最后一维对应的频率轴。
        mode: ``rel`` 相对功率（除以该通道总功率）或 ``abs`` 绝对功率。
    """
    psd = psd.astype(np.float32, copy=False)
    if mode == "rel":
        psd = psd / (psd.sum(axis=-1, keepdims=True) + 1e-30)
    elif mode != "abs":
        raise ValueError(f"未知的 bands mode: {mode!r}")

    blocks = [
        np.log10(psd[:, :, (freqs >= lo) & (freqs < hi)].mean(axis=-1) + 1e-30)
        for lo, hi in BANDS.values()
    ]
    return np.concatenate(blocks, axis=1)


def extract_features(
    data: np.ndarray,
    sfreq: float,
    mode: str = "rel",
    chunk: int = 100,
    nperseg: int = NPERSEG,
) -> np.ndarray:
    """逐通道频带功率特征，返回 ``(N, 通道数 × 频段数)``。

    Args:
        data: ``(N, C, T)`` 的 µV 数据，要求同一批内 ``T`` 相同。
        sfreq: 采样率。
        mode: ``rel`` 相对功率或 ``abs`` 绝对功率。
        chunk: 每次进入 Welch 的样本数，控制内存峰值。
        nperseg: Welch 窗长（采样点），需不大于最短试次长度。
    """
    feats: list[np.ndarray] = []
    for start in range(0, len(data), chunk):
        block = data[start : start + chunk]
        freqs, psd = welch(
            x=block,
            fs=sfreq,
            nperseg=nperseg,
            return_onesided=True,
            axis=-1,
        )
        feats.append(band_log_power(psd, freqs, mode))
    return np.concatenate(feats, axis=0)


def make_model(name: str = "lda"):
    """``lda``: 收缩 LDA（主基线）；``logreg``: 逻辑回归；``svm``: RBF SVM。"""
    if name == "lda":
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    raise ValueError(f"未知模型: {name}")
