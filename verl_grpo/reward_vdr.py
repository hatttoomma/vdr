import re
import string
from typing import Any


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    del data_source, extra_info
    pred = _normalize(_extract_answer(solution_str))
    gt = _normalize(str(ground_truth))

    if not pred:
        return 0.0
    if pred == gt:
        return 1.0
    if gt in pred or pred in gt:
        return 0.5
    return 0.0
