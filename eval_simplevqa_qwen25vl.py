#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import io
import json
import re
import string
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_score_ground_truth(solution_str: str, ground_truth: str) -> float:
    pred = _normalize(_extract_answer(solution_str or ""))
    norm_gt = _normalize(str(ground_truth))
    if not pred:
        return 0.0
    if pred == norm_gt or norm_gt in pred or pred in norm_gt:
        return 1.0
    return 0.0


def decode_base64_string(image_b64: str) -> bytes:
    image_b64 = image_b64.strip()
    if image_b64.startswith("data:") and "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    pad_len = (-len(image_b64)) % 4
    if pad_len:
        image_b64 += "=" * pad_len

    try:
        return base64.b64decode(image_b64, validate=True)
    except Exception:
        return base64.b64decode(image_b64)


def decode_image(image_field: Any) -> Image.Image:
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")

    if isinstance(image_field, bytes):
        return Image.open(io.BytesIO(image_field)).convert("RGB")

    if isinstance(image_field, str):
        image_bytes = decode_base64_string(image_field)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if isinstance(image_field, dict):
        image_bytes = image_field.get("bytes")
        if isinstance(image_bytes, str):
            image_bytes = decode_base64_string(image_bytes)
        if image_bytes is not None:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_path = image_field.get("path")
        if image_path:
            return Image.open(image_path).convert("RGB")

        nested = image_field.get("image")
        if nested is not None:
            return decode_image(nested)

    raise ValueError(f"Unsupported image field type: {type(image_field)}")


def get_prompt_text(row: dict[str, Any]) -> str:
    if row.get("question"):
        return str(row["question"])

    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        first_turn = prompt[0]
        if isinstance(first_turn, dict):
            content = first_turn.get("content", "")
            if isinstance(content, str):
                return content.replace("<image>", "").strip()
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                merged = "\n".join(text_parts).replace("<image>", "").strip()
                if merged:
                    return merged

    return ""


def get_ground_truth(row: dict[str, Any]) -> str:
    if row.get("answer") is not None:
        return str(row.get("answer", ""))
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        return str(reward_model.get("ground_truth", ""))
    return ""


@torch.inference_mode()
def generate_answer(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    question: str,
    max_new_tokens: int,
) -> str:
    user_text = (
        f"{question}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    input_ids = inputs["input_ids"]
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_parquet", type=str, default="simplevqa_data/train.parquet", help="Path to data parquet.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_file", type=str, default="simplevqa_eval_predictions.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=-1, help="Evaluate first N samples, -1 for all.")
    args = parser.parse_args()

    dataset = load_dataset("parquet", data_files=args.data_parquet, split="train")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print(f"Loaded {len(dataset)} samples from: {args.data_parquet}")
    print(f"Loading model: {args.model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    num_total = 0
    score_sum = 0.0
    num_errors = 0

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for idx, row in enumerate(tqdm(dataset, total=len(dataset))):
            num_total += 1
            sample_id = row.get("extra_info", {}).get("data_id", idx)
            question = get_prompt_text(row)
            ground_truth = get_ground_truth(row)

            try:
                images = row.get("images", [])
                if not isinstance(images, list) or len(images) == 0:
                    raise ValueError("No image found in `images` field.")
                image = decode_image(images[0])

                prediction = generate_answer(
                    model=model,
                    processor=processor,
                    image=image,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                )
                score = compute_score_ground_truth(prediction, ground_truth)
                score_sum += score

                record = {
                    "id": sample_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "score": score,
                }
            except Exception as exc:  # noqa: BLE001
                num_errors += 1
                record = {
                    "id": sample_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": None,
                    "score": 0.0,
                    "error": repr(exc),
                }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    accuracy = score_sum / num_total if num_total > 0 else 0.0
    print("Evaluation finished.")
    print(f"Total samples: {num_total}")
    print(f"Errors: {num_errors}")
    print(f"Accuracy: {accuracy:.4f} ({score_sum:.1f}/{num_total})")
    print(f"Saved predictions: {args.output_file}")


if __name__ == "__main__":
    main()
