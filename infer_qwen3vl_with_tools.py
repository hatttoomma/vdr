import argparse
import json
import os
from json import JSONDecodeError
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from infer_qwen3vl import decode_image
from text_search import TextSearch


def load_system_prompt(prompt_file: str, prompt_key: str) -> str:
    with open(prompt_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["system_prompt"][prompt_key]


def run_generation(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: list[dict[str, Any]],
    max_new_tokens: int,
) -> str:
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


def extract_tool_call(output_text: str) -> dict[str, Any] | None:
    open_tag = "<tool>"
    close_tag = "</tool>"

    start = output_text.find(open_tag)
    if start == -1:
        return None

    start += len(open_tag)
    end = output_text.find(close_tag, start)
    if end == -1:
        payload = output_text[start:]
    else:
        payload = output_text[start:end]

    payload = payload.strip()
    if not payload:
        return None

    try:
        tool_call = json.loads(payload)
    except JSONDecodeError:
        return None

    if not isinstance(tool_call, dict):
        return None
    return tool_call


def execute_tool(tool_call: dict[str, Any], search_tool: TextSearch) -> tuple[str, str]:
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args") or {}

    if tool_name != "web_search":
        return tool_name or "unknown", f"Tool error: unsupported tool `{tool_name}`."

    query = tool_args.get("query")
    if not query:
        return tool_name, "Tool error: missing required argument `query`."

    return tool_name, search_tool.search(query)


@torch.inference_mode()
def generate_answer_with_tools(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Any,
    question: str,
    system_prompt: str,
    search_tool: TextSearch,
    max_new_tokens: int = 128,
    max_tool_turns: int = 1,
) -> str:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]

    tool_turns = 0
    while True:
        output_text = run_generation(
            model=model,
            processor=processor,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )

        tool_call = extract_tool_call(output_text)
        if tool_call is None or tool_turns >= max_tool_turns:
            return output_text

        tool_name, tool_result = execute_tool(tool_call, search_tool)
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output_text}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Tool `{tool_name}` returned:\n{tool_result}\n\n"
                            "Continue reasoning and provide the final answer inside "
                            "<answer>...</answer>."
                        ),
                    }
                ],
            }
        )
        tool_turns += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="Osilly/VDR-Bench-testmini")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_file", type=str, default="vdr_bench_qwen3vl_2b_predictions.jsonl")
    parser.add_argument("--prompt_file", type=str, default="prompt.yaml")
    parser.add_argument("--prompt_key", type=str, default="web_search_only")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_tool_turns", type=int, default=1)
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

    system_prompt = load_system_prompt(args.prompt_file, args.prompt_key)
    search_tool = TextSearch()

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
                prediction = generate_answer_with_tools(
                    model=model,
                    processor=processor,
                    image=image,
                    question=question,
                    system_prompt=system_prompt,
                    search_tool=search_tool,
                    max_new_tokens=args.max_new_tokens,
                    max_tool_turns=args.max_tool_turns,
                )
                record: Dict[str, Any] = {
                    "id": sample_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                }
            except Exception as exc:  # noqa: BLE001
                record = {
                    "id": sample_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": None,
                    "error": repr(exc),
                }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Done. Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
