"""从 segmented_novel 下的整本小说文件中提取逐行文本，导出为 CSV。

数据源约定
----------
正文文本只取自 ``segmented_Chinense_novel.xlsx``（单列 Chinese_text），
它是「前言 + 全部正文」的完整版本；``_display.xlsx`` 是它的逐帧重复副本，
不参与提取。

``run_N.xlsx`` 本身不提供文本，只用来确定 run 边界：已验证
「主文件正文 == 按数字序拼接 run_1..run_N」，且每个 run 起点都落在章节号行上
（GarnettDream 1 run = 1 章，LittlePrince 1 run = 4 章），
所以 run 划分只能由这些文件的长度推出。

``onset`` / ``sample`` 取自同一被试在 ``derivatives/preproc/filtered_0.5_30`` 下
的 ``*_events.tsv``：以 CH* 事件为界分段，每段第一个 ROWS 是章节号行本身的显示
（时长仅约 0.37s），从第二个 ROWS 起才依次对应本章每个 sentence。

输出
----
对 ``subjects`` 里的每个被试、每本小说各出一份 CSV，命名为
``{subject}_{novel}.csv``（如 ``sub-04_LittlePrince.csv``）。

处理规则
--------
1. 前言：首个章节号之前的行全部丢弃。
2. 章节号：形如 ``19`` 的纯数字行只用于标记章节起点，不写入正文。
3. ``run``：EEG 采集轮次，从 1 起编号。
4. ``onset`` / ``sample``：该 sentence 对应 ROWS 事件的时间与采样点；无 EEG 数据的整行丢弃。
5. ``sentence`` 为原始折行片段，不做拼接或断句处理。
6. ``n_chars``：非标点字符数，标点按 Unicode 类别 ``P*`` 判定。
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENTED_ROOT = PROJECT_ROOT / "datasets/ChineseEEG/derivatives/novels/segmented_novel"
EVENTS_ROOT = PROJECT_ROOT / "datasets/ChineseEEG/derivatives/preproc/filtered_0.5_30"
OUTPUT_DIR = PROJECT_ROOT / "datasets/preprocessed/novels"

NOVEL_FILE = "segmented_Chinense_novel.xlsx"
TEXT_COLUMN = 0  # 单列 Chinese_text
FIELDS = ["index", "chapter", "run", "onset", "sample", "sentence", "n_chars"]

SUBJECTS = ["sub-04", "sub-05"]
TASK = "task-reading"

CHAPTER_RE = re.compile(r"^\d+$")
RUN_FILE_RE = re.compile(rf"{Path(NOVEL_FILE).stem}_run_(\d+)\.xlsx$")
RUN_NUMBER_RE = re.compile(r"_run-(\d+)_")
CH_EVENT_RE = re.compile(r"^CH(\d+)$")


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


def read_run_lengths(novel_dir: Path) -> list[int]:
    """按 run 序号升序读取各 run 文件的行数。"""
    run_files = []
    for path in novel_dir.iterdir():
        matched = RUN_FILE_RE.match(path.name)
        if matched:
            run_files.append((int(matched.group(1)), path))
    run_files.sort()
    return [len(read_lines(path)) for _, path in run_files]


def read_events(events_path: Path) -> list[dict]:
    """读取 *_events.tsv。"""
    with events_path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def index_events(events_path: Path) -> dict[int, list[tuple[str, str]]]:
    """以 CH* 为界切段，返回 {章节号: [(onset, sample), ...]}。

    每段第一个 ROWS 对应章节号行本身，从第二个 ROWS 起才对应 sentence。
    """
    segments: dict[int, list[tuple[str, str]]] = {}
    chapter: int | None = None
    for event in read_events(events_path):
        matched = CH_EVENT_RE.match(event["trial_type"])
        if matched:
            chapter = int(matched.group(1))
            segments[chapter] = []
        elif chapter is not None and event["trial_type"] == "ROWS":
            segments[chapter].append((event["onset"], event["sample"]))
    return {number: slots[1:] for number, slots in segments.items()}


def build_event_lookup(subject: str, novel_name: str) -> dict[tuple[int, int], list[tuple[str, str]]]:
    """汇总该被试该小说各 run 的事件，返回 {(run, chapter): [(onset, sample), ...]}。"""
    session_dir = EVENTS_ROOT / subject / f"ses-{novel_name}" / "eeg"
    if not session_dir.is_dir():
        raise FileNotFoundError(f"{subject} 缺少 {session_dir.name} 目录: {session_dir}")
    events_files = sorted(
        session_dir.glob(f"*_{TASK}_run-*_events.tsv"),
        key=lambda path: int(RUN_NUMBER_RE.search(path.name).group(1)),
    )

    lookup: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for events_path in events_files:
        run = int(RUN_NUMBER_RE.search(events_path.name).group(1))
        for chapter, slots in index_events(events_path).items():
            lookup[(run, chapter)] = slots
    return lookup


def extract(novel_dir: Path, subject: str) -> tuple[list[dict], str]:
    """整理出逐行记录，返回 (记录列表, 对齐情况备注)。"""
    lines = read_lines(novel_dir / NOVEL_FILE)
    run_lengths = read_run_lengths(novel_dir)
    events = build_event_lookup(subject, novel_dir.name)

    # 前言：首个章节号之前的内容整段跳过
    body_start = next(i for i, text in enumerate(lines) if CHAPTER_RE.match(text))

    # 按各 run 的行数把正文逐段划归到 run，run 序号从 1 起
    run_of_line = [0] * len(lines)
    cursor = body_start
    for run, length in enumerate(run_lengths, start=1):
        for i in range(cursor, cursor + length):
            run_of_line[i] = run
        cursor += length
    if cursor != len(lines):
        raise ValueError(f"{novel_dir.name}: run 文件合计 {cursor - body_start} 行，正文为 {len(lines) - body_start} 行")

    records: list[dict] = []
    chapter = 0
    dropped = 0
    used: dict[tuple[int, int], int] = defaultdict(int)
    for i in range(body_start, len(lines)):
        text = lines[i]
        if CHAPTER_RE.match(text):
            chapter = int(text)  # 章节号仅用于划分章节
            continue
        if not text:
            continue

        run = run_of_line[i]
        slots = events.get((run, chapter), [])
        position = used[(run, chapter)]
        used[(run, chapter)] = position + 1
        if position >= len(slots):
            dropped += 1  # 没有对应的 ROWS 事件，无法对齐，整行丢弃
            continue

        onset, sample = slots[position]
        records.append(
            {
                "index": len(records),
                "chapter": chapter,
                "run": run,
                "onset": onset,
                "sample": sample,
                "sentence": text,
                "n_chars": count_chars(text),
            }
        )

    # 每章的 ROWS 必须与 sentence 一一耗尽，否则说明对齐假设被破坏
    for (run, number), slots in events.items():
        if used[(run, number)] != len(slots):
            raise ValueError(
                f"{novel_dir.name}: run {run} 第 {number} 章有 {len(slots)} 个 ROWS，正文只用到 {used[(run, number)]} 个"
            )

    note = f"丢弃 {dropped} 行（无 EEG 事件）" if dropped else "全部对齐"
    return records, note


def main(subjects: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    novel_dirs = sorted(path for path in SEGMENTED_ROOT.iterdir() if path.is_dir())

    for subject in subjects:
        for novel_dir in novel_dirs:
            records, note = extract(novel_dir, subject)
            csv_path = OUTPUT_DIR / f"{subject}_{novel_dir.name}.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(records)
            print(
                f"{subject} {novel_dir.name}: {len(records)} 行, "
                f"章节 1-{max(r['chapter'] for r in records)}, "
                f"run 1-{max(r['run'] for r in records)}, "
                f"{note} -> {csv_path}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为指定被试导出小说逐行文本 CSV")
    parser.add_argument("subjects", nargs="*", help="被试编号，如 sub-04 sub-05；缺省时使用 SUBJECTS")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.subjects or SUBJECTS)
