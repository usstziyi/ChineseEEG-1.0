"""滑窗切分 + 多分支时间滤波（自包含实验目录）。

    dataset.py       ReadingEEGDataset / pad_collate   # 滑窗切分（同 baseline_window）
    model.py         多分支时间卷积 + Transformer / attention pooling
    text_encoder.py  LaBSETextEncoder
    train.py         切分、损失、评测与训练循环

相对 ``baseline_window`` 只改了一处：token 的时间前端从单一 ``Conv(1→32, 1×9)``
换成 5/15/35/75 四支并行卷积（核宽按 token 长度相对 ``char_token_window`` 等比例取得），
其余切分、损失、评测、日志格式完全一致。

目录内不依赖 ``baseline_window/`` 或 ``baseline_multibranch/`` 的任何模块，
可单独拷走运行。

运行::

    uv run python baseline_window_multibranch/train.py
"""
