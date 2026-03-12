#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import io
import json
import os
from typing import Any

from datasets import Dataset, load_dataset
from PIL import Image


TARGET_IMAGE_SIZE = (512, 512)


def safe_get(example: dict[str, Any], key: str, default: Any = None) -> Any:
    return example[key] if key in example else default


def decode_base64_string(image_b64: str) -> bytes:
    image_b64 = image_b64.strip()
    if image_b64.startswith("data:") and "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    # Some sources omit padding, so add it if needed.
    pad_len = (-len(image_b64)) % 4
    if pad_len:
        image_b64 += "=" * pad_len

    try:
        return base64.b64decode(image_b64, validate=True)
    except Exception:
        # Fallback for non-standard base64 payloads.
        return base64.b64decode(image_b64)


def decode_simplevqa_image(image_field: Any) -> Image.Image:
    """
    Decode image field into RGB PIL image.
    Priority follows requirement: base64 decode -> PIL.
    """
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")

    if isinstance(image_field, str):
        image_bytes = decode_base64_string(image_field)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if isinstance(image_field, bytes):
        return Image.open(io.BytesIO(image_field)).convert("RGB")

    if isinstance(image_field, dict):
        if image_field.get("bytes") is not None:
            image_bytes = image_field["bytes"]
            if isinstance(image_bytes, str):
                image_bytes = decode_base64_string(image_bytes)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if image_field.get("image") is not None:
            return decode_simplevqa_image(image_field["image"])

        if image_field.get("path") is not None:
            return Image.open(image_field["path"]).convert("RGB")

    raise ValueError(f"Unsupported image field type: {type(image_field)}")


def to_resized_pil_image(image_field: Any, target_size: tuple[int, int] = TARGET_IMAGE_SIZE) -> Image.Image | None:
    try:
        image = decode_simplevqa_image(image_field)
    except Exception:
        return None

    return image.resize(target_size, Image.Resampling.LANCZOS)


def build_prompt_text(atomic_question: str) -> str:
    lines = [
        "You are a helpful multimodal assistant.",
        "Answer the question according to the image.",
        "",
        "<image>",
        "",
        f"Question: {atomic_question}",
    ]
    lines.extend([
        "",
        "Think step by step and provide your final answer in <answer>...</answer>"
    ])

    return "\n".join(lines)


def make_map_fn(data_source: str, split: str):
    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        atomic_question = str(safe_get(example, "atomic_question", "")).strip()
        atomic_fact = str(safe_get(example, "atomic_fact", "")).strip()
        data_id = safe_get(example, "data_id", "")

        resized_image = to_resized_pil_image(safe_get(example, "image", None))
        images = [resized_image] if resized_image is not None else []

        return {
            "data_source": f"{data_source}:{split}",
            "prompt": [{"role": "user", "content": build_prompt_text(atomic_question)}],
            "images": images,
            "ability": "vqa",
            "reward_model": {
                "style": "rule",
                "ground_truth": atomic_fact,
            },
            "extra_info": {
                "index": idx,
                "split": split,
                "data_id": data_id,
                "atomic_question": atomic_question,
                "atomic_fact": atomic_fact,
            },
            "responses": [atomic_fact],
            "question": atomic_question,
            "answer": atomic_fact,
        }

    return process_fn


def try_load_split(dataset_name: str, split: str) -> Dataset:
    errors: list[str] = []

    try:
        return load_dataset(dataset_name, split=split)
    except Exception as e:
        errors.append(f"load_dataset({dataset_name!r}, split={split!r}) failed: {e}")

    try:
        ds_dict = load_dataset(dataset_name)
        if split in ds_dict:
            return ds_dict[split]
        first_split = list(ds_dict.keys())[0]
        return ds_dict[first_split]
    except Exception as e:
        errors.append(f"load_dataset({dataset_name!r}) failed: {e}")

    raise RuntimeError("Failed to load dataset. Tried patterns:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="m-a-p/SimpleVQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./simplevqa_data")
    parser.add_argument("--val_size", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=-1, help="Use -1 for all samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset={args.dataset_name}, split={args.split} ...")
    ds = try_load_split(args.dataset_name, args.split)
    print(ds)
    print("Columns:", ds.column_names)

    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    ds = ds.shuffle(seed=args.seed)
    if len(ds) < 2:
        raise ValueError("Dataset too small.")

    val_size = min(args.val_size, max(1, len(ds) // 10))
    train_size = len(ds) - val_size
    if train_size <= 0:
        raise ValueError(f"Dataset too small after split: len={len(ds)}, val_size={val_size}")

    train_ds = ds.select(range(train_size))
    val_ds = ds.select(range(train_size, len(ds)))
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    map_fn = make_map_fn(args.dataset_name, args.split)
    train_ds = train_ds.map(map_fn, with_indices=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(map_fn, with_indices=True, remove_columns=val_ds.column_names)

    # Keep only rows with at least one valid decoded image.
    train_before = len(train_ds)
    val_before = len(val_ds)
    train_ds = train_ds.filter(lambda x: isinstance(x.get("images"), list) and len(x["images"]) > 0)
    val_ds = val_ds.filter(lambda x: isinstance(x.get("images"), list) and len(x["images"]) > 0)
    print(
        f"After image guard -> train: {len(train_ds)} "
        f"(dropped {train_before - len(train_ds)}), "
        f"val: {len(val_ds)} (dropped {val_before - len(val_ds)})"
    )

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    preview_path = os.path.join(args.output_dir, "preview.json")

    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    preview = dict(train_ds[0])
    preview["images"] = [f"<image:{type(x).__name__}>" for x in preview.get("images", [])]
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2, default=str)

    print("Done.")
    print(f"- train:  {train_path}")
    print(f"- val:    {val_path}")
    print(f"- sample: {preview_path}")


if __name__ == "__main__":
    main()
