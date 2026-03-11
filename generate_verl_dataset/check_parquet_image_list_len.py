#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Iterable, List

import pyarrow.parquet as pq


def resolve_parquet_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix != ".parquet":
            raise ValueError(f"Input file is not a parquet file: {input_path}")
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
        if not files:
            raise ValueError(f"No parquet files found in directory: {input_path}")
        return files

    raise ValueError(f"Path does not exist: {input_path}")


def image_list_len(value) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)

    # Fallback for non-list decoded values.
    try:
        return len(value)
    except TypeError:
        return -1


def get_nested_field(row: dict, path: Iterable[str], default=""):
    cur = row
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
    return cur


def check_one_file(parquet_path: Path) -> int:
    parquet_file = pq.ParquetFile(parquet_path)
    existing_columns = set(parquet_file.schema_arrow.names)
    if "images" not in existing_columns:
        raise ValueError(f"Column 'images' not found in parquet: {parquet_path}")

    read_columns = ["images"]
    if "extra_info" in existing_columns:
        read_columns.append("extra_info")

    table = pq.read_table(parquet_path, columns=read_columns)
    images = table.column("images").to_pylist()

    extra_info = []
    if "extra_info" in table.column_names:
        extra_info = table.column("extra_info").to_pylist()
    else:
        extra_info = [{} for _ in range(len(images))]

    bad_count = 0
    for idx, img_list in enumerate(images):
        n_images = image_list_len(img_list)
        if n_images != 1:
            bad_count += 1
            extra = extra_info[idx] if idx < len(extra_info) else {}
            sample_id = get_nested_field(extra, ["sample_id"], default="")
            subset = get_nested_field(extra, ["subset"], default="")
            print(
                f"[BAD] file={parquet_path.name} row={idx} image_len={n_images} "
                f"sample_id={sample_id!r} subset={subset!r}"
            )

    print(
        f"[DONE] {parquet_path} total_rows={len(images)} bad_rows={bad_count}"
    )
    return bad_count


def main():
    parser = argparse.ArgumentParser(
        description="Check each row in parquet file(s) where len(images) != 1."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to one parquet file or a directory containing parquet files.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    parquet_files = resolve_parquet_files(input_path)

    total_bad = 0
    for parquet_file in parquet_files:
        total_bad += check_one_file(parquet_file)

    print(f"[SUMMARY] checked_files={len(parquet_files)} total_bad_rows={total_bad}")


if __name__ == "__main__":
    main()
