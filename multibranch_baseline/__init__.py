"""baseline + 多分支时间滤波（自包含实验目录）。

    dataset.py       ReadingEEGDataset / pad_collate
    model.py         多分支时间卷积 + Transformer / attention pooling
    text_encoder.py  LaBSETextEncoder
    train.py         切分、损失、评测与训练循环

目录内不依赖 ``baseline/`` 的任何模块，可单独拷走运行。

运行::

    uv run python multibranch_baseline/train.py
"""
