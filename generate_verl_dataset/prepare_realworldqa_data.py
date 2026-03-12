import argparse
import io
import os
from typing import Any

import datasets
from PIL import Image


def decode_image(image_field: Any) -> Image.Image:
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, dict):
        if image_field.get("bytes") is not None:
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
        if image_field.get("path") is not None:
            return Image.open(image_field["path"]).convert("RGB")
    if isinstance(image_field, bytes):
        return Image.open(io.BytesIO(image_field)).convert("RGB")
    raise ValueError(f"Unsupported image field type: {type(image_field)}")


def resize_to_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    max_side = int(max_pixels ** 0.5)
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_prompt(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question}]


def make_map_fn(data_source: str, split: str, max_pixels: int):
    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        image = decode_image(example["image"])
        image = resize_to_max_pixels(image, max_pixels)
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
                "question": question,
                "answer": answer,
            },
        }

    return process_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="xai-org/RealworldQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./data/realworldqa_verl")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = datasets.load_dataset(args.dataset_name, split=args.split)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    dataset = dataset.map(
        make_map_fn(args.dataset_name, args.split, args.max_pixels),
        with_indices=True,
        remove_columns=dataset.column_names,
    )

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    dataset.to_parquet(train_path)
    dataset.to_parquet(val_path)

    print(f"Saved train parquet to: {train_path}")
    print(f"Saved val parquet to:   {val_path}")


if __name__ == "__main__":
    main()
