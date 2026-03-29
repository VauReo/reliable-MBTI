from __future__ import annotations

import re
from typing import Any, Iterable

from dataset.sft_generation import parse_ranked_mbti

MBTI_TYPES = (
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP',
)

ANSWER_BLOCK_RE = re.compile(r'<answer>\s*([\s\S]*?)\s*</answer>', re.IGNORECASE)


def extract_answer_block(text: str) -> str | None:
    match = ANSWER_BLOCK_RE.search(text or '')
    if match is None:
        return None
    block = match.group(1).strip()
    return block or None


def dimension_similarity(predicted: str, gold: str) -> float:
    if len(predicted) != 4 or len(gold) != 4:
        return 0.0
    matches = sum(p == g for p, g in zip(predicted.upper(), gold.upper(), strict=True))
    return matches / 4.0


def reward_format(output: str) -> float:
    return 1.0 if extract_answer_block(output) is not None else 0.0


def reward_parse_success(output: str) -> float:
    return 1.0 if parse_ranked_mbti(output) else 0.0


def reward_invalid_answer(output: str) -> float:
    return 0.0 if parse_ranked_mbti(output) else 1.0


def reward_top_1(ranked_types: Iterable[str], gold: str) -> float:
    ranked = [item.upper() for item in ranked_types]
    gold = gold.strip().upper()
    return 1.0 if ranked and ranked[0] == gold else 0.0


def reward_top_k(ranked_types: Iterable[str], gold: str, top_k: int = 3) -> float:
    ranked = [item.upper() for item in ranked_types][: max(1, int(top_k))]
    gold = gold.strip().upper()
    return 1.0 if gold in ranked else 0.0


def reward_reciprocal_rank(ranked_types: Iterable[str], gold: str, k: int = 16) -> float:
    gold = gold.strip().upper()
    for index, item in enumerate(ranked_types, start=1):
        if index > max(1, int(k)):
            break
        if item.upper() == gold:
            return 1.0 / index
    return 0.0


def reward_dimension_similarity(best_prediction: str, gold: str) -> float:
    return dimension_similarity(best_prediction, gold)


def ndcg_placeholder(ranked_types: Iterable[str], gold: str, k: int = 3) -> float:
    return reward_reciprocal_rank(ranked_types, gold, k=k)


def score_completion(output: str, gold: str, reward_cfg: dict[str, Any]) -> dict[str, Any]:
    gold = gold.strip().upper()
    ranked_types = parse_ranked_mbti(output)
    best_prediction = ranked_types[0] if ranked_types else ''
    top_k = int(reward_cfg.get('top_k', 3))
    use_dimension = bool(reward_cfg.get('use_dimension_similarity', True))

    components = {
        'format_reward': reward_format(output),
        'parse_reward': reward_parse_success(output),
        'invalid_answer_reward': reward_invalid_answer(output),
        'top_1_reward': reward_top_1(ranked_types, gold),
        'top_k_reward': reward_top_k(ranked_types, gold, top_k=top_k),
        'reciprocal_rank_reward': reward_reciprocal_rank(ranked_types, gold, k=16),
        'dimension_similarity_reward': (
            reward_dimension_similarity(best_prediction, gold) if use_dimension and best_prediction else 0.0
        ),
    }
    weights = {
        'format_reward': float(reward_cfg.get('format_weight', 0.0)),
        'parse_reward': float(reward_cfg.get('parse_weight', 0.0)),
        'invalid_answer_reward': float(reward_cfg.get('invalid_answer_penalty', 0.0)),
        'top_1_reward': float(
            reward_cfg.get('top_1_weight', reward_cfg.get('exact_match_weight', 0.0))
        ),
        'top_k_reward': float(reward_cfg.get('top_k_weight', 0.0)),
        'reciprocal_rank_reward': float(
            reward_cfg.get('reciprocal_rank_weight', reward_cfg.get('ndcg_weight', 0.0))
        ),
        'dimension_similarity_reward': float(reward_cfg.get('dimension_similarity_weight', 0.0)),
    }
    total_reward = sum(components[name] * weights[name] for name in components)
    return {
        'gold': gold,
        'ranked_types': ranked_types,
        'best_prediction': best_prediction,
        'components': components,
        'weights': weights,
        'total_reward': total_reward,
    }
