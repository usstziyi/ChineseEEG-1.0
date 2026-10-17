"""滑窗 token + 多分支时间滤波的句子级 EEG 编码器。

把 ``baseline_window``（滑窗切分）与 ``baseline_multibranch``（多分支时间前端）
两个改动合到一条链路上：

    baseline_window              token = 1.0 s（256 点）的滑动窗口，token 数 = 窗口数
    baseline_multibranch         token 时间前端 = 3/7/15/29 四支并行卷积
    本模块                       token = 256 点滑动窗口；时间前端 = 5/15/35/75 四支

核宽为什么从 3/7/15/29 变成 5/15/35/75
--------------------------------------
两个实验的 token 长度不同，核宽必须按 token 长度等比例缩放，否则覆盖的频段会整体
跑偏。四支的比例取自 ``char_token_window`` 的 5/15/35/75（相对 250 点窗口）：

    baseline_multibranch（token = 100 点）  0.02/0.06/0.14/0.30 × 100 ≈ 3/7/15/29
    本模块（token = 256 点，≈ char_token_window）     5/15/35/75

纯按比例缩放会得到 5.12/15.36/35.84/76.8，取整到最接近的奇数即 5/15/35/77；这里直接用
``char_token_window`` 的整数 5/15/35/75，与 250 点窗口口径完全对齐，也避免 256 点下
凭空多出一个 77 这种"非标"核宽。四支的时长与 ≈-3 dB 截止频率（窗口长 256 Hz，
取 ``0.443 × 256 / m``，与 ``baseline_multibranch`` 口径一致）：

    Conv(1→8,  1×5)    ┐                19.5 ms   ~22.7 Hz  高 β 边缘
    Conv(1→8,  1×15)   ├                58.6 ms   ~7.6 Hz   θ / α 交界
    Conv(1→8,  1×35)   ├               136.7 ms   ~3.2 Hz   δ / θ 交界
    Conv(1→8,  1×75)   ┘               293.0 ms   ~1.5 Hz   δ

这正是滑窗方案存在的意义：字级 token 只有 100 点，最低够不到 ~3.9 Hz 以下；滑窗把
token 拉长到 256 点后，多分支前端才能真正覆盖到 δ。

两个细节（与 ``baseline_multibranch`` 一致）：

1. **same padding + replicate**：四支输出必须共享同一时间长度才能 concat；
   replicate 避免补零在 token 两端制造阶跃（核越宽受影响越大）。
2. **AvgPool(1×5, s=3)**：与 ``baseline_multibranch`` 相同。核宽 5 不超过最短的
   5/15 两支，不会把它们刚分出来的高频成分重新抹平；256 点下池化输出时间
   长度变成 ``(256 - 5) // 3 + 1 = 84``，``flatten_dim`` 随之变大，是 token 变长的
   直接后果（与 ``baseline_window`` 不额外补偿池化的口径一致）。

后续流程（GroupNorm → ELU → 跨电极空间卷积 → GroupNorm → ELU → Dropout →
Linear → L2）以及 Transformer + attention pooling 与 ``baseline_window`` 完全一致，
便于把指标差异归因到"多分支时间滤波"这一组改动。

用法::

    from baseline_window_multibranch.model import MultiBranchReadingEEGEncoder

    model = MultiBranchReadingEEGEncoder(n_chans=128, token_samples=256, embedding_dim=768)
    eeg = model(tokens, mask)      # tokens: (B, n, 128, 256)；mask: (B, n) bool
    # eeg: (B, 768)，已 L2 归一化
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

# 默认四支核宽（采样点），取自 char_token_window 的 5/15/35/75，对应 256 点的滑窗 token
DEFAULT_KERNELS = (5, 15, 35, 75)


def _split_channels(k: int, n_branches: int) -> list[int]:
    """把 k 个特征图尽量平均分给各分支，余数从前往后依次 +1。"""
    base, remainder = divmod(k, n_branches)
    if base == 0:
        raise ValueError(
            f"k={k} 不足以分给 {n_branches} 个分支，每个分支至少需要 1 个通道"
        )
    return [base + (1 if i < remainder else 0) for i in range(n_branches)]


class MultiScaleTemporalConv(nn.Module):
    """并行多支时间卷积，same padding 保证各支输出时间长度一致。

    padding 使用 ``replicate`` 而非默认的补零：token 只是连续 EEG 的一段切片，
    补零会在两端制造人为阶跃，且核越宽污染越重。

    输入 ``(N, 1, C, T)`` -> 输出 ``(N, k, C, T)``，通道维由各分支拼接而成。

    Parameters
    ----------
    k:
        输出特征图总数，按 ``kernels`` 个数平均分配。
    kernels:
        各分支的核宽（采样点），必须为正奇数且不超过 ``token_samples``。
    token_samples:
        单个 token 的时间长度，用于校验核宽。
    """

    def __init__(self, k: int, kernels=DEFAULT_KERNELS, token_samples: int = 256):
        super().__init__()

        kernels = tuple(int(m) for m in kernels)
        if not kernels:
            raise ValueError("kernels 不能为空")
        if any(m <= 0 or m % 2 == 0 for m in kernels):
            raise ValueError(f"kernels 必须全为正奇数，收到 {kernels}")
        if len(set(kernels)) != len(kernels):
            raise ValueError(f"kernels 不能重复，收到 {kernels}")
        if max(kernels) > token_samples:
            raise ValueError(
                f"最大核宽 {max(kernels)} 超过 token_samples={token_samples}"
            )

        branch_channels = _split_channels(k, len(kernels))
        self.kernels = kernels
        self.branch_channels = tuple(branch_channels)
        self.branches = nn.ModuleList(
            # token 是连续 EEG 的切片，复制边界值可以避免补零带来的数值阶跃
            # 这里是唯一直接吃原始 EEG 的层，故 bias=False：有了它输出就不再正比于输入，
            # 小量纲（如 V）时会退化成常量、破坏 GroupNorm 的尺度不变性并让梯度塌掉；
            # 每通道偏移由后续 GroupNorm 自带的可学习 beta 提供，语义更合适
            nn.Conv2d(
                1, out_ch, kernel_size=(1, m), padding=(0, m // 2),
                padding_mode="replicate", bias=False,
            )
            for out_ch, m in zip(branch_channels, kernels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, 1, C, T)`` -> ``(N, k, C, T)``。"""
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class MultiBranchTokenEncoder(nn.Module):
    """把一个窗口 token ``(C, token_samples)`` 编码成 ``d_model`` 维向量。

        (N, C, 256)
            ↓ 四支并行时间卷积 + replicate same padding
        (N, k, C, 256)
            ↓ AvgPool(1×m2, s)
        (N, k, C, T2)              # 256 点、m2=5、s=3 -> T2=84
            ↓ GroupNorm + ELU
            ↓ Conv(k→k, n_chans×1)  跨全部电极的空间卷积
        (N, k, 1, T2)
            ↓ GroupNorm + ELU + Dropout
            ↓ Linear(k·T2 → d_model)

    Parameters
    ----------
    n_chans:
        EEG 通道数，本数据集为 128（E1–E128）。
    token_samples:
        每个窗口的采样点数，本数据集默认 256（256 Hz × 1.0 s）。
    k:
        时间卷积的特征图总数，默认 32（四支各 8 个）。
    kernels:
        多尺度核宽，默认 ``(5, 15, 35, 75)``。传单元素元组可退回单支结构。
    m2 / s:
        时间平均池化的核宽与步长，默认 ``(5, 3)``。
    """

    def __init__(
        self,
        n_chans: int = 128,
        token_samples: int = 256,
        d_model: int = 128,
        k: int = 32,
        kernels=DEFAULT_KERNELS,
        m2: int = 5,
        s: int = 3,
        drop_prob: float = 0.3,
    ):
        super().__init__()

        if n_chans <= 0 or token_samples <= 0 or d_model <= 0 or k <= 0:
            raise ValueError("n_chans / token_samples / d_model / k 必须为正")
        if m2 <= 0 or s <= 0:
            raise ValueError("m2 / s 必须为正")
        if m2 > token_samples:
            raise ValueError(
                f"m2={m2} 超过 token_samples={token_samples}，无法池化"
            )
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob 必须满足 0 <= drop_prob < 1")

        # same padding 之后时间长度仍是 token_samples
        t2 = (token_samples - m2) // s + 1
        if t2 <= 0:
            raise ValueError(
                f"token_samples={token_samples}, m2={m2}, s={s} 算不出有效输出长度"
            )

        self.n_chans = n_chans
        self.token_samples = token_samples
        self.d_model = d_model
        self.m2 = int(m2)
        self.s = int(s)
        self.out_time = t2
        self.flatten_dim = k * t2

        self.temporal = MultiScaleTemporalConv(k, kernels, token_samples)
        self.kernels = self.temporal.kernels
        self.branch_channels = self.temporal.branch_channels
        self.pool = nn.AvgPool2d(kernel_size=(1, m2), stride=(1, s))
        # 与 baseline 一致：时间池化之后再归一化；GroupNorm 逐样本归一化，
        # 不受 collate 补出来的全零 token 影响
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=k)
        self.act = nn.ELU()
        # 一次跨越全部电极的空间卷积
        self.spatial = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(n_chans, 1))
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=k)
        self.dropout = nn.Dropout(drop_prob)
        self.proj = nn.Linear(self.flatten_dim, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: ``(N, C, token_samples)`` -> ``(N, d_model)``。"""
        if tokens.ndim != 3:
            raise ValueError(
                f"token 必须是 (N, C, T) 三维，收到 {tuple(tokens.shape)}"
            )
        if tokens.shape[1] != self.n_chans:
            raise ValueError(
                f"期望 {self.n_chans} 个 EEG 通道，收到 {tokens.shape[1]}"
            )
        if tokens.shape[2] != self.token_samples:
            raise ValueError(
                f"期望每个 token {self.token_samples} 个采样点，收到 {tokens.shape[2]}"
            )

        x = self.temporal(tokens.unsqueeze(1))  # (N, 1, C, T) -> (N, k, C, T)
        x = self.pool(x)                        # (N, k, C, T2)
        x = self.act(self.norm1(x))
        x = self.spatial(x)                     # (N, k, 1, T2)
        x = self.act(self.norm2(x))
        x = self.dropout(x)
        return self.proj(x.flatten(start_dim=1))  # (N, d_model)


class MultiBranchReadingEEGEncoder(nn.Module):
    """句子级 EEG 编码器，输出与文本共享空间的定长 embedding。

    除 token 时间前端换成多分支卷积外，其余与 ``baseline_window`` 方案完全一致。

    Parameters
    ----------
    n_chans / token_samples:
        必须与 ``baseline_window_multibranch.dataset.ReadingEEGDataset`` 的输出一致
        （默认 128 / 256）。
    d_model:
        token 空间与 Transformer 的宽度。
    n_heads / n_layers / dim_feedforward / dropout:
        Transformer 配置，token 数只有 2~6、数据量也小，保持 2 层、4 头。
    max_tokens:
        位置编码支持的最大 token 数。本数据集按默认窗长 / 步长最多切出 6 个窗口，
        默认 16 留出余量；``train.py`` 会按实际数据的最大窗口数显式传入。
    embedding_dim:
        最终 embedding 维度；与 LaBSE 的 768 对齐。
    token_kwargs:
        透传给 ``MultiBranchTokenEncoder``（k / kernels / m2 / s / drop_prob）。
    """

    def __init__(
        self,
        n_chans: int = 128,
        token_samples: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_tokens: int = 16,
        embedding_dim: int = 768,
        **token_kwargs,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须为正")

        self.d_model = d_model
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim

        self.token_encoder = MultiBranchTokenEncoder(
            n_chans=n_chans,
            token_samples=token_samples,
            d_model=d_model,
            **token_kwargs,
        )

        # token 顺序就是时间顺序，位置信息必须显式给出
        self.position = nn.Parameter(torch.zeros(1, max_tokens, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first=True 时 PyTorch 无法启用 nested tensor 加速，显式关掉以免告警
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # attention pooling：一个可学习 query 对所有有效 token 做加权汇总，
        # 唯一目的就是把变长序列池化成一个定长向量。
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(d_model, embedding_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 只有这两个是手工建的 torch.zeros Parameter，必须自己打散；
        # 其余模块（含 out_proj 的 nn.Linear）交给 PyTorch 默认初始化即可。
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens:
            ``(B, n, C, token_samples)``；pad 位置填 0。
        mask:
            ``(B, n)`` bool，``True`` 表示有效 token。

        Returns
        -------
        torch.Tensor:
            ``(B, embedding_dim)``，已做 L2 归一化；与文本 embedding 点积即余弦相似度。
        """
        if tokens.ndim != 4:
            raise ValueError(
                f"tokens 必须是 (B, n, C, T) 四维，收到 {tuple(tokens.shape)}"
            )
        if mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"mask 形状 {tuple(mask.shape)} 与 tokens 前两维 {tuple(tokens.shape[:2])} 不一致"
            )

        batch_size, n_tokens = tokens.shape[:2]
        if n_tokens > self.max_tokens:
            raise ValueError(
                f"token 数 {n_tokens} 超过 max_tokens={self.max_tokens}，"
                "请调大 max_tokens 或换更粗的切分粒度"
            )

        # (B, n, C, T) -> (B·n, C, T)，所有 token 共用同一套权重
        flat = tokens.reshape(batch_size * n_tokens, *tokens.shape[2:])
        hidden = self.token_encoder(flat).reshape(batch_size, n_tokens, self.d_model)
        hidden = hidden + self.position[:, :n_tokens]

        # True 表示要忽略的位置，所以取反
        key_padding_mask = ~mask
        hidden = self.transformer(hidden, src_key_padding_mask=key_padding_mask)

        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(
            query, hidden, hidden, key_padding_mask=key_padding_mask
        )  # (B, 1, d_model)
        pooled = pooled.squeeze(1)  # (B, d_model)

        return F.normalize(self.out_proj(pooled), p=2, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = MultiBranchReadingEEGEncoder(
        n_chans=128, token_samples=256, embedding_dim=768
    )
    encoder = model.token_encoder
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total:,}（可训练 {trainable:,}）")
    print(f"kernels={encoder.kernels} 各分支通道={encoder.branch_channels}")
    print(
        f"token encoder 输出时间长度={encoder.out_time} "
        f"flatten={encoder.flatten_dim} pool=({encoder.m2}, s={encoder.s})"
    )

    batch_size, n_tokens = 6, 6
    tokens = torch.randn(batch_size, n_tokens, 128, 256)
    mask = torch.ones(batch_size, n_tokens, dtype=torch.bool)
    # 模拟 batch 内窗口数不齐：不同样本保留不同数目的窗口（默认口径 2~6）
    keep_list = [2, 3, 4, 5, 6, 6]
    for i, keep in enumerate(keep_list):
        mask[i, keep:] = False
        tokens[i, keep:] = 0.0

    model.eval()
    with torch.no_grad():
        eeg = model(tokens, mask)

    print(f"\ntokens={tuple(tokens.shape)} -> eeg={tuple(eeg.shape)}")
    print("每行有效 token 数:", mask.sum(dim=1).tolist())
    print("L2 范数:", eeg.norm(dim=-1).tolist())

    # pad 位置灌噪声，输出应保持不变
    noisy = tokens.clone()
    for i, keep in enumerate(keep_list):
        noisy[i, keep:] = torch.randn_like(noisy[i, keep:]) * 100.0
    with torch.no_grad():
        eeg_noisy = model(noisy, mask)
    print("padding 位置灌噪声后的最大输出偏差:", (eeg - eeg_noisy).abs().max().item())
