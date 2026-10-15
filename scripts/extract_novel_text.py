"""从 segmented_novel 下的整本小说文件中提取逐行文本，导出为 CSV。

数据源约定
----------
只读取每本小说目录下的 ``segmented_Chinense_novel.xlsx``（单列 Chinese_text）。
该文件是「前言 + 全部正文」的完整版本，``_display.xlsx`` 与 ``run_N.xlsx``
分别是它的逐帧重复副本和子集，均不参与提取。

处理规则
--------
1. 前言：首个章节号之前的行全部丢弃。
2. 章节号：形如 ``19`` 的纯数字行只用于标记章节起点，不写入正文。
3. ``sentence`` 为原始折行片段，不做拼接或断句处理。
4. ``n_chars``：非标点字符数，标点按 Unicode 类别 ``P*`` 判定。
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENTED_ROOT = PROJECT_ROOT / "datasets/ChineseEEG/derivatives/novels/segmented_novel"
OUTPUT_DIR = PROJECT_ROOT / "datasets/preprocessed/novels"

NOVEL_FILE = "segmented_Chinense_novel.xlsx"
TEXT_COLUMN = 0  # 单列 Chinese_text
FIELDS = ["index", "chapter", "sentence", "n_chars"]

CHAPTER_RE = re.compile(r"^\d+$")


def count_chars(text: str) -> int:
    """统计非标点字符数，Unicode 类别以 P 开头的字符视为标点。"""
    return sum(1 for char in text if not unicodedata.category(char).startswith("P"))


def read_lines(xlsx_path: Path) -> list[str]:
    """读取单列 xlsx，返回去掉表头后每一行的文本。"""
    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    lines = []
    for row_number, row in enumerate(sheet.iter_rows(values_only=True)):
        if row_number == 0:  # 表头 Chinese_text
            continue
        cell = row[TEXT_COLUMN]
        lines.append("" if cell is None else str(cell).strip())
    workbook.close()
    return lines


def extract(xlsx_path: Path) -> list[dict]:
    """按章节整理出逐行记录，返回 index / chapter / sentence / n_chars 列表。"""
    lines = read_lines(xlsx_path)

    # 前言：首个章节号之前的内容整段跳过
    body_start = next(i for i, text in enumerate(lines) if CHAPTER_RE.match(text))

    records: list[dict] = []
    chapter = 0
    for text in lines[body_start:]:
        if CHAPTER_RE.match(text):
            chapter = int(text)  # 章节号仅用于划分章节
            continue
        if not text:
            continue
        records.append(
            {
                "index": len(records),
                "chapter": chapter,
                "sentence": text,
                "n_chars": count_chars(text),
            }
        )
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for novel_dir in sorted(path for path in SEGMENTED_ROOT.iterdir() if path.is_dir()):
        records = extract(novel_dir / NOVEL_FILE)
        csv_path = OUTPUT_DIR / f"{novel_dir.name}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        print(
            f"{novel_dir.name}: {len(records)} 行, "
            f"章节 1-{max(r['chapter'] for r in records)} -> {csv_path}"
        )


if __name__ == "__main__":
    main()
