import os
import io
import json
import base64
import argparse
from typing import Any, Dict

import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def decode_image(image_field: Any) -> Image.Image:

    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")

    if isinstance(image_field, bytes):
        return Image.open(io.BytesIO(image_field)).convert("RGB")

    if isinstance(image_field, dict):
        # 兼容某些 datasets image 特征
        if "bytes" in image_field and image_field["bytes"] is not None:
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
        if "path" in image_field and image_field["path"] is not None:
            return Image.open(image_field["path"]).convert("RGB")

    if isinstance(image_field, str):
        img_b64 = image_field.strip()

        # 兼容 data URL
        if img_b64.startswith("data:"):
            img_b64 = img_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(img_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    raise ValueError(f"Unsupported image field type: {type(image_field)}")


@torch.inference_mode()
def generate_answer(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 128,
) -> str:
    """
    对单条样本做推理。
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="Osilly/VDR-Bench-testmini")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_file", type=str, default="vdr_bench_qwen3vl_2b_predictions.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=-1, help="只跑前多少条，-1 表示全量")
    parser.add_argument("--resume", action="store_true", help="若输出文件已存在，则跳过已完成样本")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name} [{args.split}]")
    dataset = load_dataset(args.dataset_name, split=args.split)

    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    processed_ids = set()
    if args.resume and os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                processed_ids.add(item["id"])
        print(f"Resume enabled. Found {len(processed_ids)} finished samples.")

    print(f"Loading model: {args.model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.output_file, "a" if args.resume else "w", encoding="utf-8") as fout:
        for sample in tqdm(dataset, total=len(dataset)):
            sample_id = sample["id"]

            if sample_id in processed_ids:
                continue

            question = sample["question"]
            ground_truth = sample["answer"]

            try:
                image = decode_image(sample["image"])
                prediction = generate_answer(
                    model=model,
                    processor=processor,
                    image=image,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                )

                record: Dict[str, Any] = {
                    "id": sample_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                }

            except Exception as e:
                record = {
                    "id": sample_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": None,
                    "error": repr(e),
                }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Done. Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()