"""LORO / LODO：留一 run、留一天的内容标签泛化评估。

移植自 ``new_model/plan_loro_lodo``，默认标签为 ``chapter``（内容层标签），
可用 ``--label`` 切换。两组评估共用同一套口径：

1. ``LeaveOneGroupOut`` 逐组留出，汇总池化准确率与逐组准确率；
2. 同数据同特征的随机 K 折折外预测作为参照，按组计算配对差
   （参照 − 留一）；
3. 按组重采样的 bootstrap 95% CI（配对差）；
4. 逐组配对差的 Wilcoxon 符号秩检验；
5. 全局打乱标签的置换零分布（默认 200 次）；
6. 组内多数类基线（乐观基线：用被留出组自身的多数类）。

已知退化（脚本会在结果里显式标记 ``degenerate``）：

- ``chapter`` 严格嵌套在 ``run`` 内，GarnettDream 中 chapter 与 run 一一对应，
  LittlePrince 中每章也只出现在一个 run；因此留一 run / 留一天时测试集里的
  chapter 在训练集中从未出现，准确率必然接近 0。这不是 bug，而是标签与分组
  完全嵌套导致的定义性问题，结论只能用于说明“该标签不支持跨组泛化”。
- LittlePrince 只有 2 个 day，Wilcoxon 双侧最小 p = 0.5，统计上无意义。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits

from baseline.dataset import ReadingEEGDataset
from baseline.features import (
    LABEL_SOURCES,
    extract_labeled_features,
    group_oof_predict,
    per_group_accuracy,
)
from baseline.model import BANDS, NPERSEG, make_model
from baseline.run_identity import cross_validate, write_rows

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "loro_lodo"

SUMMARY_FIELDS = [
    "subject",
    "novel",
    "label_source",
    "group_kind",
    "n_samples",
    "n_classes",
    "n_groups",
    "chance_accuracy",
    "loro_pooled_accuracy",
    "reference_pooled_accuracy",
    "majority_baseline",
    "paired_mean_diff",
    "paired_diff_ci_low",
    "paired_diff_ci_high",
    "wilcoxon_p",
    "permutation_n",
    "permutation_null_mean",
    "permutation_p",
    "degenerate",
]


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    """对逐组配对差做按组重采样的 95% CI。"""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(values), size=len(values))
        means[i] = values[pick].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def majority_baseline(y: np.ndarray, groups: np.ndarray) -> float:
    """组内多数类基线（用被评估组自己的多数类，属乐观基线）。"""
    correct = 0
    for group in np.unique(groups):
        mask = groups == group
        values, counts = np.unique(y[mask], return_counts=True)
        correct += int(counts.max())
    return float(correct / len(y))


def _logo_permutation_accuracy(X, y, groups, model_name, seed) -> float:
    with threadpool_limits(limits=1):
        shuffled = np.random.default_rng(seed).permutation(y)
        pred = group_oof_predict(X, shuffled, groups, model_name)
    return float(np.mean(pred == shuffled))


def permutation_logo(
    X, y, groups, model_name, n_perm: int, seed: int, n_jobs: int = 1
) -> np.ndarray:
    if n_perm <= 0:
        return np.empty(0)
    if n_jobs > 1:
        null = np.empty(n_perm)
        chunk = 50
        for start in range(0, n_perm, chunk):
            stop = min(start + chunk, n_perm)
            values = Parallel(n_jobs=n_jobs)(
                delayed(_logo_permutation_accuracy)(X, y, groups, model_name, seed + i)
                for i in range(start, stop)
            )
            null[start:stop] = values
            print(f"  [置换-{len(np.unique(groups))} 组] {stop}/{n_perm}", flush=True)
        return null

    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _logo_permutation_accuracy(X, y, groups, model_name, seed + i)
        if (i + 1) % max(n_perm // 5, 1) == 0:
            print(f"  [置换] {i + 1}/{n_perm}", flush=True)
    return null


def evaluate_groups(
    args,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str,
) -> dict:
    n_groups = len(np.unique(groups))
    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes

    labels_per_group = [len(np.unique(y[groups == g])) for g in np.unique(groups)]
    groups_per_label = [len(np.unique(groups[y == label])) for label in np.unique(y)]
    degenerate = bool(max(groups_per_label) == 1)

    print(
        f"\n[{kind}] {n_groups} 组 | {n_classes} 类 chance={chance:.4f} | "
        f"每组类别数 {min(labels_per_group)}~{max(labels_per_group)} | "
        f"跨组出现的标签数 {sum(1 for v in groups_per_label if v > 1)}/{n_classes}"
    )
    if degenerate:
        print(
            f"[{kind}] 注意：所有标签都只出现在一个组内，留出该组时测试类别"
            "在训练集中不可见，准确率按定义接近 0"
        )

    pred_logo = group_oof_predict(X, y, groups, args.model)
    pooled = float(np.mean(pred_logo == y))
    fold_acc = per_group_accuracy(y, pred_logo, groups)

    folds = min(args.folds, int(np.bincount(y).min()))
    ref_accs, _, ref_oof = cross_validate(
        X, y, args.model, folds, args.seed, return_oof=True
    )
    ref_pooled = float(np.mean(ref_oof == y))
    ref_by_group = per_group_accuracy(y, ref_oof, groups)

    group_values = np.array([fold_acc[int(g)] for g in np.unique(groups)])
    ref_values = np.array([ref_by_group[int(g)] for g in np.unique(groups)])
    diffs = ref_values - group_values
    diff_low, diff_high = bootstrap_mean_ci(diffs, seed=args.seed)

    if len(diffs) >= 2 and np.any(diffs != 0):
        wilcoxon_p = float(wilcoxon(diffs, alternative="two-sided").pvalue)
    else:
        wilcoxon_p = float("nan")

    null = permutation_logo(
        X, y, groups, args.model, args.permutations, args.seed, n_jobs=args.n_jobs
    )
    permutation = None
    if len(null):
        permutation = {
            "n": int(len(null)),
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_p95": float(np.percentile(null, 95)),
            "observed": pooled,
            "p_value": float((np.sum(null >= pooled) + 1) / (len(null) + 1)),
        }

    result = {
        "group_kind": kind,
        "n_groups": n_groups,
        "n_classes": n_classes,
        "chance_accuracy": float(chance),
        "pooled_accuracy": pooled,
        "reference_pooled_accuracy": ref_pooled,
        "reference_cv_folds": folds,
        "majority_baseline": majority_baseline(y, groups),
        "paired_mean_diff": float(diffs.mean()),
        "paired_diff_ci_low": diff_low,
        "paired_diff_ci_high": diff_high,
        "wilcoxon_p": wilcoxon_p,
        "wilcoxon_n_groups": int(len(diffs)),
        "permutation": permutation,
        "degenerate": degenerate,
        "labels_spanning_multiple_groups": int(
            sum(1 for v in groups_per_label if v > 1)
        ),
        "folds": [
            {
                "group": int(group),
                "n_samples": int((groups == group).sum()),
                "logo_accuracy": float(fold_acc[int(group)]),
                "reference_accuracy": float(ref_by_group[int(group)]),
                "paired_diff": float(diffs[i]),
            }
            for i, group in enumerate(np.unique(groups))
        ],
    }

    print(
        f"[{kind}] LORO={pooled:.4f} | 参照 {folds} 折={ref_pooled:.4f} | "
        f"多数类基线={result['majority_baseline']:.4f}"
    )
    print(
        f"[{kind}] 配对差={diffs.mean():+.4f} "
        f"CI [{diff_low:+.4f}, {diff_high:+.4f}] | Wilcoxon p={wilcoxon_p:.4f} "
        f"(n={len(diffs)} 组)"
    )
    if permutation:
        print(
            f"[{kind}] 置换 null={permutation['null_mean']:.4f} "
            f"p={permutation['p_value']:.4f} (n={permutation['n']})"
        )
    return result


def run_subject(args, subject: str, novel: str, out_root: Path) -> list[dict]:
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
    kinds = ["run", "day"] if args.leave_out == "both" else [args.leave_out]

    results = []
    for kind in kinds:
        groups = payload["runs"] if kind == "run" else payload["days"]
        if len(np.unique(groups)) < 2:
            print(f"[{kind}] 只有 {len(np.unique(groups))} 组，跳过")
            continue
        result = evaluate_groups(args, X, y, groups, kind)
        results.append(result)

    out_dir = out_root / f"{subject}_{novel}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "subject": subject,
        "novel": novel,
        "task": "loro_lodo",
        "label_source": args.label,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "sfreq": sfreq,
        "nperseg": int(args.nperseg),
        "bands": {k: list(v) for k, v in BANDS.items()},
        "band_mode": args.bands_mode,
        "model": args.model,
        "seed": args.seed,
        "results": results,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    for result in results:
        kind = result["group_kind"]
        write_rows(
            out_dir / f"folds_{kind}.csv",
            result["folds"],
            ["group", "n_samples", "logo_accuracy", "reference_accuracy", "paired_diff"],
        )
        rows.append(
            {
                "subject": subject,
                "novel": novel,
                "label_source": args.label,
                "group_kind": kind,
                "n_samples": int(len(y)),
                "n_classes": result["n_classes"],
                "n_groups": result["n_groups"],
                "chance_accuracy": result["chance_accuracy"],
                "loro_pooled_accuracy": result["pooled_accuracy"],
                "reference_pooled_accuracy": result["reference_pooled_accuracy"],
                "majority_baseline": result["majority_baseline"],
                "paired_mean_diff": result["paired_mean_diff"],
                "paired_diff_ci_low": result["paired_diff_ci_low"],
                "paired_diff_ci_high": result["paired_diff_ci_high"],
                "wilcoxon_p": result["wilcoxon_p"],
                "permutation_n": (
                    None if result["permutation"] is None else result["permutation"]["n"]
                ),
                "permutation_null_mean": (
                    None
                    if result["permutation"] is None
                    else result["permutation"]["null_mean"]
                ),
                "permutation_p": (
                    None
                    if result["permutation"] is None
                    else result["permutation"]["p_value"]
                ),
                "degenerate": result["degenerate"],
            }
        )
    write_rows(out_dir / "summary.csv", rows, SUMMARY_FIELDS)
    print(f"[Saved] {out_dir}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="LORO/LODO：留一 run、留一天的内容标签泛化（标签可配置）。"
    )
    parser.add_argument("--subject", default="sub-04")
    parser.add_argument(
        "--novel", nargs="+", default=["GarnettDream", "LittlePrince"]
    )
    parser.add_argument("--label", default="chapter", choices=LABEL_SOURCES)
    parser.add_argument(
        "--leave-out",
        default="both",
        choices=("run", "day", "both"),
        help="run=LORO，day=LODO，both=两者都做",
    )
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--n-chars-exact", type=int, default=10)
    parser.add_argument("--dedup", action="store_true", default=True)
    parser.add_argument("--no-dedup", dest="dedup", action="store_false")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="分层抽样上限，用于烟雾测试"
    )
    parser.add_argument("--folds", type=int, default=5, help="参照随机划分的折数")
    parser.add_argument("--permutations", type=int, default=200)
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
    run_name = args.run_name or f"loro_lodo_{datetime.now():%Y%m%d_%H%M%S}"
    out_root = args.out_dir / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for novel in args.novel:
        rows.extend(run_subject(args, args.subject, novel, out_root))
    write_rows(out_root / "summary_all.csv", rows, SUMMARY_FIELDS)
    print(f"\n[Saved] {out_root / 'summary_all.csv'}")


if __name__ == "__main__":
    main()
