import argparse
import base64
import io
import os
from typing import Any

import datasets
from PIL import Image


def decode_image(image_field: Any) -> Image.Image:
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")

    if isinstance(image_field, bytes):
        return Image.open(io.BytesIO(image_field)).convert("RGB")

    if isinstance(image_field, dict):
        if "bytes" in image_field and image_field["bytes"] is not None:
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
        if "path" in image_field and image_field["path"] is not None:
            return Image.open(image_field["path"]).convert("RGB")

    if isinstance(image_field, str):
        img_b64 = image_field.strip()
        if img_b64.startswith("data:"):
            img_b64 = img_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(img_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    raise ValueError(f"Unsupported image field type: {type(image_field)}")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_prompt(question: str) -> list[dict[str, str]]:
    instruction = (
        "Answer the question based on the image. "
        "Return the final answer in <answer>...</answer>."
    )
    return [{"role": "user", "content": f"{question}\n\n{instruction}"}]


def make_map_fn(data_source: str, split: str):
    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        image = decode_image(example["image"])
        image_bytes = image_to_png_bytes(image)

        question = str(example["question"])
        answer = str(example["answer"])

        return {
            "data_source": data_source,
            "prompt": build_prompt(question),
            "images": [image_bytes],
            "ability": "vqa",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "id": example.get("id", idx),
                "question": question,
                "answer": answer,
            },
        }

    return process_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Osilly/VDR-Bench-testmini")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./data/vdr_bench_verl")
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_val_samples", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = datasets.load_dataset(args.dataset_name, split=args.train_split)
    val_dataset = datasets.load_dataset(args.dataset_name, split=args.val_split)

    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_val_samples > 0:
        val_dataset = val_dataset.select(range(min(args.max_val_samples, len(val_dataset))))

    train_dataset = train_dataset.map(
        make_map_fn(args.dataset_name, args.train_split),
        with_indices=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        make_map_fn(args.dataset_name, args.val_split),
        with_indices=True,
        remove_columns=val_dataset.column_names,
    )

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"Saved train parquet to: {train_path}")
    print(f"Saved val parquet to:   {val_path}")


if __name__ == "__main__":
    main()
