from typing import Iterable


MBTI_TYPES = (
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
)


def dimension_similarity(predicted: str, gold: str) -> float:
    if len(predicted) != 4 or len(gold) != 4:
        return 0.0
    matches = sum(p == g for p, g in zip(predicted, gold))
    return matches / 4.0


def ndcg_placeholder(ranked_types: Iterable[str], gold: str, k: int = 3) -> float:
    ranked = list(ranked_types)[:k]
    if not ranked:
        return 0.0
    for index, item in enumerate(ranked, start=1):
        if item == gold:
            return 1.0 / index
    return 0.0
