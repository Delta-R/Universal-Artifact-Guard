#!/usr/bin/env python3
"""
批量重命名图像文件，将文件名末尾的数字序号按照设定的起始值顺序递增。

用法示例：
    python rename_images.py Dataset/fake --start 90
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
TRAILING_DIGITS_PATTERN = re.compile(r"(\d+)$")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将文件夹中的图片文件名末尾序号按顺序递增重命名"
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="需要重命名的文件夹路径"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="序号起始值，默认为 1"
    )
    parser.add_argument(
        "--extensions",
        "-e",
        nargs="+",
        help="需要处理的扩展名列表，默认为常见图片格式"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印计划，不实际重命名"
    )
    return parser.parse_args()

def normalize_extensions(extensions: Sequence[str] | None) -> List[str]:
    items = extensions or DEFAULT_EXTENSIONS
    normalized: List[str] = []
    for ext in items:
        ext = ext.lower()
        if not ext.startswith('.'):
            ext = f'.{ext}'
        normalized.append(ext)
    return normalized

def collect_image_files(folder: Path, extensions: Sequence[str]) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"找不到文件夹：{folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"指定路径不是文件夹：{folder}")
    eligible = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    eligible.sort(key=lambda p: p.name)
    return eligible

def build_new_name(path: Path, number: int) -> Path:
    stem = path.stem
    match = TRAILING_DIGITS_PATTERN.search(stem)
    if match:
        base = stem[:match.start()]
        padding = len(match.group(1))
        # 如果原始基名结尾不是分隔符，则保留原样；否则直接追加
        base_to_use = base if base.endswith(("_", "-", "#", "@")) or base == "" else base
    else:
        base_to_use = stem + "_"
        padding = 0
    formatted_number = f"{number:0{padding}d}" if padding else str(number)
    new_stem = f"{base_to_use}{formatted_number}"
    return path.with_name(new_stem + path.suffix)

def ensure_no_external_conflicts(plan: Sequence[Tuple[Path, Path]]) -> None:
    sources = {src.resolve() for src, _ in plan}
    for _, dest in plan:
        if dest.resolve() in sources:
            continue
        if dest.exists():
            raise FileExistsError(f"目标文件已存在且不在重命名列表中：{dest}")

def perform_renames(plan: Sequence[Tuple[Path, Path]], dry_run: bool = False) -> None:
    if dry_run:
        for src, dest in plan:
            print(f"[DRY-RUN] {src.name} -> {dest.name}")
        return
    temp_entries: List[Tuple[Path, Path]] = []
    for src, dest in plan:
        if src.resolve() == dest.resolve():
            continue
        tmp_name = src.with_name(f".__tmp_{uuid.uuid4().hex}{src.suffix}")
        src.rename(tmp_name)
        temp_entries.append((tmp_name, dest))
    for tmp, dest in temp_entries:
        tmp.rename(dest)

def main() -> None:
    args = parse_arguments()
    if args.start < 0:
        print("起始序号必须为非负整数", file=sys.stderr)
        sys.exit(1)
    extensions = normalize_extensions(args.extensions)
    files = collect_image_files(args.folder, extensions)
    if not files:
        print("未在目标文件夹中找到匹配的图片文件。")
        return
    plan: List[Tuple[Path, Path]] = []
    current_number = args.start
    for file_path in files:
        new_path = build_new_name(file_path, current_number)
        plan.append((file_path, new_path))
        current_number += 1
    ensure_no_external_conflicts(plan)
    perform_renames(plan, dry_run=args.dry_run)
    last_number = args.start + len(files) - 1
    print(f"总计处理 {len(files)} 个文件。")
    print(f"序号范围：{args.start} -> {last_number}")

if __name__ == "__main__":
    main()
