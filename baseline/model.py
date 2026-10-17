"""句子级 EEG 编码器：CNN tokenizer + Transformer + attention pooling。

结构与 ``char_token/model.py`` 一致，只是把默认超参换成本数据集的口径
（``baseline/dataset.py``）：``C = 128``（E1–E128）、``token_samples = 90``、
``n_chars <= 10``。

整体结构：

    tokens  (B, n, C, token_samples)        # n 个等长字 token，n 可变（本数据集 1~10）
        ↓  reshape 成 (B·n, C, token_samples)
    TokenEncoder（共享权重，逐 token 独立编码）
        ↓
    (B, n, d_model)
        ↓  + 可学习位置编码（token 顺序就是字序，必须有）
    TransformerEncoder（n_layers 层）
        ↓  按 mask 屏蔽 padding 的 token
    (B, n, d_model)
        ↓  attention pooling（可学习 query 对有效 token 做加权汇总）
    (B, d_model)
        ↓  Linear
    (B, embedding_dim)
        ↓  L2 归一化

两个关键选择（与 char_token 相同）：

1. **token encoder 权重共享** —— 相当于 ViT 的 patch embedding，把所有字映射到
   同一个 d_model 空间。若每个字符位置各用一套权重，参数会爆炸，也无法泛化到
   没见过的句子长度。

2. **attention pooling 而不是 mean-pool** —— token 个数恰好等于字符数，mean-pool
   会让模型靠"平均了多少个 token"反推句子长度；可学习 query 的加权汇总表达力更强，
   也避免了这种计数泄漏。

用法::

    from baseline.model import ReadingEEGEncoder

    model = ReadingEEGEncoder(n_chans=128, token_samples=90, embedding_dim=768)
    eeg = model(tokens, mask)      # tokens: (B, n, 128, 90)；mask: (B, n) bool
    # eeg: (B, 768)，已 L2 归一化
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TokenEncoder(nn.Module):
    """把一个字 token ``(C, token_samples)`` 编码成 ``d_model`` 维向量。

    NICE 风格的 TSConv 小 CNN：先在时间轴上做卷积 + 平均池化，再用一个跨全部
    电极的空间卷积把通道维压掉。时间卷积核刻意取得比固定长度版本小
    （默认 m1=9, m2=13, s=2），因为这里的输入只有 90 个采样点（约 0.35 s），
    用大核会把时间信息压没。

    注意：归一化用 ``GroupNorm(1, k)`` 而非 ``BatchNorm2d``。collate 会往 batch 里
    塞 pad 出来的全零 token，BatchNorm 会把这些零样本算进 batch 统计量，污染真实
    样本的归一化；GroupNorm 是逐样本归一化，完全不受 padding 影响。

    Parameters
    ----------
    n_chans:
        EEG 通道数，本数据集为 128（E1–E128）。
    token_samples:
        每个 token 的采样点数，本数据集为 90。
    """

    def __init__(
        self,
        n_chans: int = 128,
        token_samples: int = 90,
        d_model: int = 128,
        k: int = 32,
        m1: int = 9,
        m2: int = 13,
        s: int = 2,
        drop_prob: float = 0.3,
    ):
        super().__init__()

        if n_chans <= 0 or token_samples <= 0 or d_model <= 0 or k <= 0:
            raise ValueError("n_chans / token_samples / d_model / k 必须为正")
        if m1 <= 0 or m2 <= 0 or s <= 0:
            raise ValueError("m1 / m2 / s 必须为正")
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob 必须满足 0 <= drop_prob < 1")

        t1 = token_samples - m1 + 1
        if t1 < m2:
            raise ValueError(
                f"token_samples={token_samples} 太短：m1={m1}, m2={m2} 至少需要 "
                f"{m1 + m2 - 1} 个采样点"
            )
        t2 = (t1 - m2) // s + 1

        self.n_chans = n_chans
        self.token_samples = token_samples
        self.d_model = d_model
        self.out_time = t2
        self.flatten_dim = k * t2

        self.tsconv = nn.Sequential(
            # 时间卷积：(B·n, 1, C, T) -> (B·n, k, C, T1)
            # 唯一直接吃原始 EEG 的层，故 bias=False：有了它卷积输出就不再正比于输入，
            # 小量纲（如 V）时会变成常量、破坏 GroupNorm 的尺度不变性，并让梯度塌掉；
            # 每通道偏移由后续 GroupNorm 自带的可学习 beta 提供，语义更合适
            nn.Conv2d(in_channels=1, out_channels=k, kernel_size=(1, m1), bias=False),
            # 时间平均池化：(B·n, k, C, T1) -> (B·n, k, C, T2)
            nn.AvgPool2d(kernel_size=(1, m2), stride=(1, s)),
            nn.GroupNorm(num_groups=1, num_channels=k),
            nn.ELU(),
            # 空间卷积：一次跨越全部电极，(B·n, k, C, T2) -> (B·n, k, 1, T2)
            nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(n_chans, 1)),
            nn.GroupNorm(num_groups=1, num_channels=k),
            nn.ELU(),
            nn.Dropout(drop_prob),
        )
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

        x = self.tsconv(tokens.unsqueeze(1))  # (N, 1, C, T) -> (N, k, 1, T2)
        return self.proj(x.flatten(start_dim=1))  # (N, d_model)


class ReadingEEGEncoder(nn.Module):
    """句子级 EEG 编码器，输出与文本共享空间的定长 embedding。

    Parameters
    ----------
    n_chans / token_samples:
        必须与 ``baseline.dataset.ReadingEEGDataset`` 的输出一致（默认 128 / 90）。
    d_model:
        token 空间与 Transformer 的宽度。
    n_heads / n_layers / dim_feedforward / dropout:
        Transformer 配置。token 数只有 1~10、数据量也小，默认给得很浅
        （2 层、4 头），别往上堆。
    max_tokens:
        位置编码支持的最大 token 数。本数据集 ``n_chars`` 最大 10，
        ``pad_collate`` 会补齐到 ``MAX_TOKENS = 10``，默认留到 16 以留出余量。
    embedding_dim:
        最终 embedding 维度；与文本侧对齐，LaBSE 是 768。
    token_kwargs:
        透传给 ``TokenEncoder``（k / m1 / m2 / s / drop_prob）。

    Attributes
    ----------
    token_encoder:
        逐 token 的小 CNN，权重在所有字之间共享。
    """

    def __init__(
        self,
        n_chans: int = 128,
        token_samples: int = 90,
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

        self.token_encoder = TokenEncoder(
            n_chans=n_chans,
            token_samples=token_samples,
            d_model=d_model,
            **token_kwargs,
        )

        # token 顺序就是字序，位置信息必须显式给出
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

    model = ReadingEEGEncoder(n_chans=128, token_samples=90, embedding_dim=768)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total:,}（可训练 {trainable:,}）")
    print(
        f"token encoder 输出时间长度={model.token_encoder.out_time} "
        f"flatten={model.token_encoder.flatten_dim}"
    )

    batch_size, n_tokens = 6, 10
    tokens = torch.randn(batch_size, n_tokens, 128, 90)
    mask = torch.ones(batch_size, n_tokens, dtype=torch.bool)
    # 模拟 batch 内长度不齐：不同样本保留不同数目的 token（本数据集 n_chars 为 1~10）
    keep_list = [1, 3, 5, 7, 9, 10]
    for i, keep in enumerate(keep_list):
        mask[i, keep:] = False
        tokens[i, keep:] = 0.0

    model.eval()
    with torch.no_grad():
        eeg = model(tokens, mask)

    print(f"\ntokens={tuple(tokens.shape)} -> eeg={tuple(eeg.shape)}")
    print("每行有效 token 数:", mask.sum(dim=1).tolist())
    print("L2 范数:", eeg.norm(dim=-1).tolist())

    # token 之外的 padding 数量不应影响结果：给 pad 位置灌噪声，输出应保持不变
    noisy = tokens.clone()
    for i, keep in enumerate(keep_list):
        noisy[i, keep:] = torch.randn_like(noisy[i, keep:]) * 100.0
    with torch.no_grad():
        eeg_noisy = model(noisy, mask)
    print("padding 位置灌噪声后的最大输出偏差:", (eeg - eeg_noisy).abs().max().item())
