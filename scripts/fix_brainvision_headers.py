"""修复 BrainVision 头文件里指向不存在文件的 DataFile/MarkerFile 引用。

背景
----
BrainVision 把头文件（``.vhdr`` / ``.vmrk``）和二进制数据（``.eeg``）分开存放：

- ``.vhdr`` 用 ``DataFile=`` 指向 ``.eeg``，用 ``MarkerFile=`` 指向 ``.vmrk``；
- ``.vmrk`` 用 ``DataFile=`` 指向 ``.eeg``。

MNE 靠这些字段定位二进制文件。若预处理的 pybv 导出把头文件内容写错
（例如目录与实际文件是 ``ses-GarnettDream``，头里却写成 ``ses-GranettDream``），
MNE 就会去找一个不存在的文件并抛 ``FileNotFoundError``。

修复规则
--------
对每个头文件，用**它自己的文件名**（去掉扩展名）推导应为的兄弟文件名：

- ``DataFile`` 应为 ``<stem>.eeg``
- ``MarkerFile`` 应为 ``<stem>.vmrk``

仅当原引用所指文件**不存在**、而推导出的兄弟文件**存在**时才改写，
因此不会动任何本来能正常解析的引用。

用法
----
    python scripts/fix_brainvision_headers.py            # 直接纠错（默认扫描 filtered_0.5_30）
    python scripts/fix_brainvision_headers.py --dry-run  # 只预览，不写回
    python scripts/fix_brainvision_headers.py --root <目录>

    # filtered_0.5_80 下同样存在 ses-GranettDream 拼写错误，实跑纠错：
    python scripts/fix_brainvision_headers.py --root datasets/ChineseEEG/derivatives/preproc/filtered_0.5_80

    # 实跑前先预览 filtered_0.5_80 的改动（50 个头文件 / 54 处引用）：
    python scripts/fix_brainvision_headers.py --root datasets/ChineseEEG/derivatives/preproc/filtered_0.5_80 --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EEG_ROOT = PROJECT_ROOT / "datasets/ChineseEEG/derivatives/preproc/filtered_0.5_30"

# 头文件字段 -> 该字段应指向的兄弟文件扩展名
EXPECTED_EXT = {"DataFile": ".eeg", "MarkerFile": ".vmrk"}


def fix_file(path: Path, apply: bool) -> tuple[list[str], list[str]]:
    """修复单个头文件，返回 (已修正的改动说明, 无法修正的悬空引用说明)。"""
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    stem = path.stem  # 如 sub-04_ses-GarnettDream_task-reading_run-01_eeg

    fixed: list[str] = []
    unresolved: list[str] = []
    for i, line in enumerate(lines):
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or key not in EXPECTED_EXT:
            continue

        referenced = value.strip()
        if (path.parent / referenced).exists():
            continue  # 引用有效，不动

        expected = f"{stem}{EXPECTED_EXT[key]}"
        if not (path.parent / expected).exists():
            unresolved.append(f"{key}: {referenced}（推导出的 {expected} 也不存在）")
            continue

        eol = line[len(line.rstrip("\r\n")):]
        lines[i] = f"{key}={expected}{eol}"
        fixed.append(f"{key}: {referenced} -> {expected}")

    if fixed and apply:
        path.write_bytes("".join(lines).encode("utf-8"))
    return fixed, unresolved


def main(root: Path, apply: bool) -> None:
    headers = sorted(
        [*root.rglob("*.vhdr"), *root.rglob("*.vmrk")],
    )
    if not headers:
        print(f"未在 {root} 下找到 .vhdr/.vmrk 文件")
        return

    n_fixed = 0
    n_unresolved = 0
    for path in headers:
        fixed, unresolved = fix_file(path, apply)
        for item in fixed:
            print(f"[修正] {path.relative_to(root)}  {item}")
            n_fixed += 1
        for item in unresolved:
            print(f"[跳过] {path.relative_to(root)}  {item}")
            n_unresolved += 1

    verb = "已修正" if apply else "待修正"
    prefix = "" if apply else "[dry-run] "
    print(
        f"\n{prefix}扫描 {len(headers)} 个头文件，{verb} {n_fixed} 处引用，"
        f"无法修正 {n_unresolved} 处"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="修正 BrainVision 头文件里悬空的 DataFile/MarkerFile 引用")
    parser.add_argument("--root", type=Path, default=EEG_ROOT, help=f"扫描目录，缺省 {EEG_ROOT}")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要做的修改，不写回文件")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.root, apply=not args.dry_run)
