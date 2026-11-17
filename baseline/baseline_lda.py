"""句类分类基线：Welch 频带 log 功率 + StandardScaler + 收缩 LDA。

移植自 ``new_model/plan_baseline_lda``，默认标签为 ``chapter``（内容层标签），
可用 ``--label`` 切换成 ``run`` / ``day``。评估口径：

1. 按标签分层的随机 8:2 holdout（train:test = 4:1，不设 val）；
2. 分层 K 折交叉验证（默认 5 折）；
3. 全局打乱标签的置换零分布（默认 2000 次）；
4. 按 **run** 聚类的 bootstrap 95% CI（同一 run 的句子高度相关，不能按样本
   独立重采样）。

注意：随机划分允许同一 run / 同一 chapter 的句子同时出现在 train 和 test，
因此这里的指标是“同分布内可分性”，不是跨 run 泛化；跨 run 的结论要看
``baseline.loro_lodo``。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from baseline.dataset import ReadingEEGDataset
from baseline.features import LABEL_SOURCES, extract_labeled_features
from baseline.model import BANDS, NPERSEG, make_model
from baseline.run_identity import (
    bootstrap_accuracy_ci,
    cross_validate,
    permutation_null,
    split_indices,
    write_confusion,
    write_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "baseline_lda"

SUMMARY_FIELDS = [
    "subject",
    "novel",
    "label_source",
    "n_samples",
    "n_classes",
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
    "cv_macro_f1_mean",
    "permutation_n",
    "permutation_null_mean",
    "permutation_z",
    "permutation_p",
]


def run_subject(args, subject: str, novel: str, out_root: Path) -> dict:
    print(f"\n===== {subject} / {novel} | label={args.label} =====")
    dataset = ReadingEEGDataset(
        subject,
        novel,
        min_chars=args.min_chars,
        dedup=args.dedup,
        preload=args.preload,
        n_chars_exact=args.n_chars_exact,
    )
    payload = extract_labeled_features(
        dataset,
        label_source=args.label,
        mode=args.bands_mode,
        nperseg=args.nperseg,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    sfreq = float(dataset.sfreq)
    dataset.close()

    X, y = payload["features"], payload["labels"]
    runs = payload["runs"]
    n_classes = len(payload["label_values"])
    chance = 1.0 / n_classes
    folds = min(args.folds, int(np.bincount(y).min()))

    train_idx, test_idx = split_indices(y, test_ratio=args.test_ratio, seed=args.seed)
    print(
        f"[Split] train={len(train_idx)} test={len(test_idx)} | "
        f"{n_classes} 类 chance={chance:.4f} | folds={folds}"
    )

    model = make_model(args.model)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    ci_low, ci_high = bootstrap_accuracy_ci(y[test_idx], pred, runs[test_idx], seed=args.seed)

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
            y[test_idx], pred, labels=list(range(n_classes))
        ).tolist(),
        "per_class": [
            {
                "value": int(value),
                "n_test": int((y[test_idx] == index).sum()),
                "recall": (
                    float(np.mean(pred[y[test_idx] == index] == index))
                    if np.any(y[test_idx] == index)
                    else None
                ),
            }
            for index, value in enumerate(payload["label_values"])
        ],
    }
    print(
        f"[{args.model}] test acc={holdout['accuracy']:.4f} "
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}] "
        f"macro_f1={holdout['macro_f1']:.4f} (chance={chance:.4f})"
    )

    accs, f1s, oof = cross_validate(X, y, args.model, folds, args.seed, return_oof=True)
    cv_low, cv_high = bootstrap_accuracy_ci(y, oof, runs, seed=args.seed)
    cv = {
        "folds": folds,
        "accuracy": [float(v) for v in accs],
        "macro_f1": [float(v) for v in f1s],
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "macro_f1_mean": float(f1s.mean()),
        "accuracy_ci_low": cv_low,
        "accuracy_ci_high": cv_high,
    }
    print(
        f"[CV] {folds} 折 acc={cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f} "
        f"95% CI [{cv_low:.4f}, {cv_high:.4f}]"
    )

    permutation = None
    if args.permutations > 0:
        null = permutation_null(
            X, y, args.model, folds, args.seed, args.permutations, n_jobs=args.n_jobs
        )
        permutation = {
            "n": args.permutations,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_p95": float(np.percentile(null, 95)),
            "observed": cv["accuracy_mean"],
            "z_score": float((cv["accuracy_mean"] - null.mean()) / (null.std() + 1e-12)),
            "p_value": float((np.sum(null >= cv["accuracy_mean"]) + 1) / (len(null) + 1)),
        }
        print(
            f"[置换] null={permutation['null_mean']:.4f} ± {permutation['null_std']:.4f} "
            f"p={permutation['p_value']:.4f}"
        )

    if args.label == "chapter":
        counts = Counter(int(v) for v in payload["chapters"])
        print(
            f"[Note] 随机划分下 train/test 共享 run；chapter 只用于内容层标签，"
            f"{len(counts)} 个 chapter，每章 {min(counts.values())}~{max(counts.values())} 条"
        )

    out_dir = out_root / f"{subject}_{novel}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "subject": subject,
        "novel": novel,
        "task": "baseline_lda",
        "label_source": args.label,
        "n_samples": int(len(y)),
        "n_classes": n_classes,
        "n_features": int(X.shape[1]),
        "sfreq": sfreq,
        "nperseg": int(args.nperseg),
        "bands": {k: list(v) for k, v in BANDS.items()},
        "band_mode": args.bands_mode,
        "model": args.model,
        "split": {
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "holdout": holdout,
        "cv": cv,
        "permutation": permutation,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_rows(
        out_dir / "per_class.csv",
        [
            {
                "value": row["value"],
                "n_test": row["n_test"],
                "recall": row["recall"],
            }
            for row in holdout["per_class"]
        ],
        ["value", "n_test", "recall"],
    )
    write_confusion(
        out_dir / "confusion.csv",
        holdout["confusion_matrix"],
        payload["label_values"],
    )

    summary = {
        "subject": subject,
        "novel": novel,
        "label_source": args.label,
        "n_samples": int(len(y)),
        "n_classes": n_classes,
        "n_features": int(X.shape[1]),
        "sfreq": sfreq,
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
        "cv_macro_f1_mean": cv["macro_f1_mean"],
        "permutation_n": None if permutation is None else permutation["n"],
        "permutation_null_mean": None if permutation is None else permutation["null_mean"],
        "permutation_z": None if permutation is None else permutation["z_score"],
        "permutation_p": None if permutation is None else permutation["p_value"],
    }
    write_rows(out_dir / "summary.csv", [summary], SUMMARY_FIELDS)
    print(f"[Saved] {out_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="句类分类基线：Welch 频带功率 + 收缩 LDA（标签可配置）。"
    )
    parser.add_argument("--subject", default="sub-04")
    parser.add_argument(
        "--novel", nargs="+", default=["GarnettDream", "LittlePrince"]
    )
    parser.add_argument("--label", default="chapter", choices=LABEL_SOURCES)
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--n-chars-exact", type=int, default=10)
    parser.add_argument("--dedup", action="store_true", default=True)
    parser.add_argument("--no-dedup", dest="dedup", action="store_false")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="分层抽样上限，用于烟雾测试"
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="lda", choices=("lda", "logreg", "svm"))
    parser.add_argument("--bands-mode", default="rel", choices=("rel", "abs"))
    parser.add_argument("--nperseg", type=int, default=NPERSEG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"baseline_lda_{datetime.now():%Y%m%d_%H%M%S}"
    out_root = args.out_dir / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    rows = [run_subject(args, args.subject, novel, out_root) for novel in args.novel]
    write_rows(out_root / "summary_all.csv", rows, SUMMARY_FIELDS)
    print(f"\n[Saved] {out_root / 'summary_all.csv'}")


if __name__ == "__main__":
    main()
