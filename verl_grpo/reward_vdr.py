import re
import string
from collections import Counter, defaultdict
from typing import Any


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _majority_vote_labels(preds: list[str], group_keys: list[str]) -> list[str]:
    grouped = defaultdict(list)
    for idx, (pred, key) in enumerate(zip(preds, group_keys)):
        grouped[key].append((idx, pred))

    majority = {}
    for key, pairs in grouped.items():
        counter = Counter(pred for _, pred in pairs if pred)
        if not counter:
            raise ValueError(f"No valid predictions found for group key: {key}")
        max_cnt = max(counter.values())
        candidates = {p for p, c in counter.items() if c == max_cnt}
        # deterministic tie-break: earliest appeared candidate in this group
        for _, pred in pairs:
            if pred in candidates:
                majority[key] = pred
                break

    return [majority.get(key, "") for key in group_keys]


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    data_sources: list[str] | None = None,
    solution_strs: list[str] | None = None,
    ground_truths: list[str] | None = None,
    extra_infos: list[dict[str, Any] | None] | None = None,
    uids: list[str] | None = None,
) -> float | list[float]:
    # Training reward: use majority vote in each rollout group as pseudo label.
    preds = [_normalize(_extract_answer(s or "")) for s in solution_strs]
    uid_values = uids
    if uid_values is None or len(uid_values) == 0:
        raise ValueError("uids is required for training reward")
    group_keys = [str(u) for u in uid_values]
    pseudo_labels = _majority_vote_labels(preds, group_keys)
    return [1.0 if pred and (pred == label or label in pred or pred in label) else 0.0 for pred, label in zip(preds, pseudo_labels)]



def compute_score_ground_truth(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    data_sources: list[str] | None = None,
    solution_strs: list[str] | None = None,
    ground_truths: list[str] | None = None,
    extra_infos: list[dict[str, Any] | None] | None = None,
    uids: list[str] | None = None,
) -> float | list[float]:
    # Validation reward: always compare prediction with dataset ground-truth.
    if solution_strs is not None:
        gts = ground_truths or [""] * len(solution_strs)
        scores = []
        for s, gt in zip(solution_strs, gts):
            pred = _normalize(_extract_answer(s or ""))
            norm_gt = _normalize(str(gt))
            scores.append(1.0 if pred and (pred == norm_gt or norm_gt in pred or pred in norm_gt) else 0.0)
        return scores

    pred = _normalize(_extract_answer(solution_str or ""))
    norm_gt = _normalize(str(ground_truth))
    if not pred:
        return 0.0
    if pred == norm_gt or norm_gt in pred or pred in norm_gt:
        return 1.0
    return 0.0
