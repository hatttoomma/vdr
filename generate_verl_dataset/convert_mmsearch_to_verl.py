#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert HuggingFace dataset CaraJ/MMSearch to verl-formatted multimodal parquet files.

Minimal runnable multimodal setting:
- choose one subset only: end2end
- use only `query_image` as the image input
- put one <image> placeholder into prompt
- write train/val parquet for verl

Usage:
    python convert_mmsearch_to_verl_mm.py --output_dir ./mmsearch_verl_mm

Optional:
    python convert_mmsearch_to_verl_mm.py \
        --output_dir ./mmsearch_verl_mm \
        --subset end2end \
        --use_image_field query_image \
        --val_size 20 \
        --max_samples 300
"""

import os
import json
import argparse
from typing import Any, Dict, List

from datasets import load_dataset, Dataset


DATASET_NAME = "CaraJ/MMSearch"


def safe_get(example: Dict[str, Any], key: str, default=None):
    return example[key] if key in example else default


def try_load_subset(dataset_name: str, subset: str) -> Dataset:
    errors: List[str] = []

    try:
        ds = load_dataset(dataset_name, subset, split=subset)
        return ds
    except Exception as e:
        errors.append(f"load_dataset({dataset_name!r}, {subset!r}, split={subset!r}) failed: {e}")

    try:
        ds = load_dataset(dataset_name, subset, split="train")
        return ds
    except Exception as e:
        errors.append(f"load_dataset({dataset_name!r}, {subset!r}, split='train') failed: {e}")

    try:
        ds_dict = load_dataset(dataset_name, subset)
        if len(ds_dict) == 0:
            raise RuntimeError("dataset dict is empty")
        first_split = list(ds_dict.keys())[0]
        return ds_dict[first_split]
    except Exception as e:
        errors.append(f"load_dataset({dataset_name!r}, {subset!r}) failed: {e}")

    try:
        ds = load_dataset(dataset_name, split=subset)
        return ds
    except Exception as e:
        errors.append(f"load_dataset({dataset_name!r}, split={subset!r}) failed: {e}")

    raise RuntimeError(
        "Failed to load dataset. Tried several patterns:\n" + "\n".join(errors)
    )


def has_valid_image(x: Any) -> bool:
    """
    HF Image feature may decode to:
    - PIL.Image.Image
    - dict like {"path": ..., "bytes": ...}
    - sometimes None
    """
    if x is None:
        return False

    # dict-form image
    if isinstance(x, dict):
        if x.get("bytes", None) is not None:
            return True
        if x.get("path", None):
            return True
        return False

    # PIL image or other decoded object: accept
    return True


def build_prompt_text(example: Dict[str, Any], use_image_field: str) -> str:
    query = safe_get(example, "query", "").strip()
    area = safe_get(example, "area", "")
    subfield = safe_get(example, "subfield", "")
    timestamp = safe_get(example, "timestamp", "")

    lines = [
        "You are a helpful multimodal assistant.",
        "Use the image and the question to answer accurately.",
        "",
        "<image>",
        "",
        f"Question: {query}",
    ]

    meta_lines = []
    if area:
        meta_lines.append(f"Area: {area}")
    if subfield:
        meta_lines.append(f"Subfield: {subfield}")
    if timestamp:
        meta_lines.append(f"Date: {timestamp}")
    if meta_lines:
        lines.extend(["", "Context metadata:"])
        lines.extend(meta_lines)

    lines.extend([
        "",
        "Return only the final short answer."
    ])

    return "\n".join(lines)


def make_map_fn(subset: str, use_image_field: str):
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        gt_answer = safe_get(example, "gt_answer", "")
        alt_answers = safe_get(example, "alternative_gt_answers", [])
        gt_requery = safe_get(example, "gt_requery", "")

        image_obj = safe_get(example, use_image_field, None)
        images = [image_obj] if has_valid_image(image_obj) else []

        prompt_text = build_prompt_text(example, use_image_field)

        # Important for verl multimodal:
        # - prompt contains <image>
        # - sample includes `images` list
        row = {
            "data_source": f"{DATASET_NAME}:{subset}",
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            "images": images,
            "ability": "multimodal_search_qa",
            "reward_model": {
                "style": "rule",
                "ground_truth": gt_answer,
                "alternative_ground_truths": alt_answers,
            },
            "extra_info": {
                "index": idx,
                "subset": subset,
                "sample_id": safe_get(example, "sample_id", ""),
                "area": safe_get(example, "area", ""),
                "subfield": safe_get(example, "subfield", ""),
                "timestamp": safe_get(example, "timestamp", ""),
                "gt_requery": gt_requery,
                "image_field_used": use_image_field,
            },
            "responses": [gt_answer] if gt_answer else [],
            "question": safe_get(example, "query", ""),
            "answer": gt_answer,
        }
        return row

    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./mmsearch_data/')
    parser.add_argument("--subset", type=str, default="end2end",
                        choices=["end2end", "rerank", "summarization"])
    parser.add_argument("--use_image_field", type=str, default="query_image",
                        choices=["query_image", "image_search_result"])
    parser.add_argument("--val_size", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Use -1 for all samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop_samples_without_image", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {DATASET_NAME} subset={args.subset} ...")
    ds = try_load_subset(DATASET_NAME, args.subset)
    print(ds)
    print("Columns:", ds.column_names)

    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    if args.drop_samples_without_image:
        ds = ds.filter(lambda x: has_valid_image(x.get(args.use_image_field, None)))

    ds = ds.shuffle(seed=args.seed)

    if len(ds) < 2:
        raise ValueError("Dataset too small after filtering.")

    val_size = min(args.val_size, max(1, len(ds) // 10))
    train_size = len(ds) - val_size
    if train_size <= 0:
        raise ValueError(f"Dataset too small after split: len={len(ds)}, val_size={val_size}")

    train_ds = ds.select(range(train_size))
    val_ds = ds.select(range(train_size, len(ds)))

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    map_fn = make_map_fn(args.subset, args.use_image_field)
    train_ds = train_ds.map(map_fn, with_indices=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(map_fn, with_indices=True, remove_columns=val_ds.column_names)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    preview_path = os.path.join(args.output_dir, "preview.json")

    print(f"Saving train to {train_path}")
    train_ds.to_parquet(train_path)

    print(f"Saving val to {val_path}")
    val_ds.to_parquet(val_path)

    # preview: avoid dumping raw image bytes into json
    preview = dict(train_ds[0])
    if "images" in preview:
        preview["images"] = [f"<image:{type(x).__name__}>" for x in preview["images"]]

    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2, default=str)

    print("Done.")
    print(f"- train:  {train_path}")
    print(f"- val:    {val_path}")
    print(f"- sample: {preview_path}")


if __name__ == "__main__":
    main()