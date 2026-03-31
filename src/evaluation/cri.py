from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v <= min_v:
        return np.full(values.shape, 0.5, dtype=float)
    return (values - min_v) / (max_v - min_v)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = np.sqrt(float((x_c**2).sum()) * float((y_c**2).sum()))
    if denom <= 0.0:
        return 0.0
    return float((x_c * y_c).sum() / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    return _pearson(_ranks(x), _ranks(y))


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0 or labels.size == 0:
        return 0.0
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return 0.0
    return float(roc_auc_score(labels, scores))


def _cri_additive(u: np.ndarray, s: np.ndarray, w_u: float, w_s: float) -> np.ndarray:
    return np.clip(1.0 - (w_u * u + w_s * (1.0 - s)), 0.0, 1.0)


def _cri_multiplicative(u: np.ndarray, s: np.ndarray, w_u: float, w_s: float) -> np.ndarray:
    left = np.power(np.clip(1.0 - u, 1e-12, 1.0), w_u)
    right = np.power(np.clip(s, 1e-12, 1.0), w_s)
    return np.clip(left * right, 0.0, 1.0)


def _score_formula(formula: str, u: np.ndarray, s: np.ndarray, w_u: float, w_s: float) -> np.ndarray:
    if formula == 'additive':
        return _cri_additive(u, s, w_u, w_s)
    if formula == 'multiplicative':
        return _cri_multiplicative(u, s, w_u, w_s)
    raise ValueError(f'Unknown CRI formula: {formula}')


def _bucket_accuracy(scores: np.ndarray, y: np.ndarray, bucket_count: int) -> list[dict[str, float]]:
    if scores.size == 0:
        return []
    order = np.argsort(-scores)
    bucket_count = max(1, int(bucket_count))
    bucket_size = max(1, int(math.ceil(len(scores) / bucket_count)))
    rows: list[dict[str, float]] = []
    for idx in range(bucket_count):
        chunk_idx = order[idx * bucket_size : (idx + 1) * bucket_size]
        if chunk_idx.size == 0:
            continue
        chunk_scores = scores[chunk_idx]
        chunk_y = y[chunk_idx]
        rows.append(
            {
                'bucket': float(idx + 1),
                'size': float(chunk_idx.size),
                'score_min': float(chunk_scores.min()),
                'score_max': float(chunk_scores.max()),
                'score_mean': float(chunk_scores.mean()),
                'accuracy': float(chunk_y.mean()),
            }
        )
    return rows


def _variant_metrics(name: str, scores: np.ndarray, y: np.ndarray, bucket_count: int) -> dict[str, Any]:
    return {
        'name': name,
        'mean_score': float(scores.mean()) if scores.size else 0.0,
        'spearman_with_correctness': _spearman(scores, y),
        'auc_correctness': _binary_auc(scores, y),
        'bucket_analysis': _bucket_accuracy(scores, y, bucket_count),
    }


def _split_indices(
    n: int,
    fit_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n <= 1:
        idx = np.arange(n, dtype=int)
        return idx, idx
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fit_size = max(1, min(n - 1, int(round(fit_fraction * n))))
    fit_idx = np.asarray(indices[:fit_size], dtype=int)
    eval_idx = np.asarray(indices[fit_size:], dtype=int)
    return fit_idx, eval_idx


def _learn_weights_grid(
    *,
    formula: str,
    u: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    grid_step: float,
) -> tuple[float, float, dict[str, float]]:
    step = float(grid_step)
    if step <= 0.0:
        step = 0.05
    candidates = np.arange(0.0, 1.0 + step / 2.0, step)
    best_u = 0.5
    best_obj = -1.0
    for w_u in candidates:
        w_s = 1.0 - float(w_u)
        scores = _score_formula(formula, u, s, float(w_u), w_s)
        objective = _binary_auc(scores, y)
        if objective > best_obj:
            best_obj = objective
            best_u = float(w_u)
    best_w_u = float(best_u)
    best_w_s = float(1.0 - best_w_u)
    diagnostics = {
        'objective_auc_fit': float(best_obj),
        'grid_step': float(step),
        'candidate_count': float(len(candidates)),
    }
    return best_w_u, best_w_s, diagnostics


def compute_cri(
    per_example: list[dict[str, Any]],
    *,
    stability_metrics: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not per_example:
        return per_example, {'enabled': False, 'reason': 'empty input'}

    formula_mode = str(config.get('formula_mode', 'both')).strip().lower()
    if formula_mode not in {'additive', 'multiplicative', 'both'}:
        raise ValueError('cri.formula_mode must be one of additive|multiplicative|both')
    formulas = ['additive', 'multiplicative'] if formula_mode == 'both' else [formula_mode]
    primary_formula = str(config.get('primary_formula', formulas[-1])).strip().lower()
    if primary_formula not in {'additive', 'multiplicative'}:
        primary_formula = formulas[-1]

    y = np.asarray([1.0 if row.get('top_1_correct') else 0.0 for row in per_example], dtype=float)
    entropy = np.asarray([float(row.get('entropy', 0.0)) for row in per_example], dtype=float)
    margin = np.asarray([float(row.get('margin', 0.0)) for row in per_example], dtype=float)
    entropy_norm = _minmax_normalize(entropy)
    inv_margin_norm = 1.0 - _minmax_normalize(margin)
    signal_cfg = config.get('uncertainty_signal', {})
    entropy_weight = float(signal_cfg.get('entropy_weight', 0.5))
    margin_weight = float(signal_cfg.get('inverse_margin_weight', 0.5))
    weight_sum = entropy_weight + margin_weight
    if weight_sum <= 0.0:
        entropy_weight, margin_weight = 0.5, 0.5
        weight_sum = 1.0
    entropy_weight /= weight_sum
    margin_weight /= weight_sum
    u = np.clip(entropy_weight * entropy_norm + margin_weight * inv_margin_norm, 0.0, 1.0)

    fallback_stability = float(config.get('fallback_stability', 0.5))
    fallback_stability = float(np.clip(fallback_stability, 0.0, 1.0))
    stability_by_id: dict[str, float] = {}
    per_row_stability = stability_metrics.get('per_example', [])
    if isinstance(per_row_stability, list):
        for row in per_row_stability:
            row_id = str(row.get('row_id', ''))
            if not row_id:
                continue
            value = float(row.get('stability_score', fallback_stability))
            stability_by_id[row_id] = float(np.clip(value, 0.0, 1.0))
    s = np.asarray(
        [stability_by_id.get(str(row.get('row_id', '')), fallback_stability) for row in per_example],
        dtype=float,
    )
    s = np.clip(s, 0.0, 1.0)

    weight_mode = str(config.get('weight_mode', 'learned')).strip().lower()
    learning_cfg = config.get('learning', {})
    fit_fraction = float(learning_cfg.get('fit_fraction', 0.5))
    grid_step = float(learning_cfg.get('grid_step', 0.05))
    fit_idx, eval_idx = _split_indices(len(per_example), fit_fraction, int(seed))

    fixed_cfg = config.get('fixed_weights', {})
    fixed_w_u = float(fixed_cfg.get('uncertainty', 0.5))
    fixed_w_u = float(np.clip(fixed_w_u, 0.0, 1.0))
    fixed_w_s = 1.0 - fixed_w_u

    variant_scores: dict[str, np.ndarray] = {
        'uncertainty_only': np.clip(1.0 - u, 0.0, 1.0),
        'stability_only': s.copy(),
    }
    learned_weights: dict[str, dict[str, float]] = {}
    learning_diagnostics: dict[str, dict[str, float]] = {}

    for formula in formulas:
        if weight_mode == 'learned':
            w_u, w_s, diag = _learn_weights_grid(
                formula=formula,
                u=u[fit_idx],
                s=s[fit_idx],
                y=y[fit_idx],
                grid_step=grid_step,
            )
        else:
            w_u, w_s = fixed_w_u, fixed_w_s
            diag = {'objective_auc_fit': 0.0, 'grid_step': 0.0, 'candidate_count': 0.0}
        learned_weights[formula] = {'w_u': float(w_u), 'w_s': float(w_s)}
        learning_diagnostics[formula] = diag
        variant_scores[f'cri_{formula}'] = _score_formula(formula, u, s, w_u, w_s)

    if f'cri_{primary_formula}' in variant_scores:
        primary_key = f'cri_{primary_formula}'
    else:
        primary_key = next(key for key in variant_scores if key.startswith('cri_'))

    for idx, row in enumerate(per_example):
        row['uncertainty_u'] = float(u[idx])
        row['stability_s'] = float(s[idx])
        if 'cri_additive' in variant_scores:
            row['cri_additive'] = float(variant_scores['cri_additive'][idx])
        if 'cri_multiplicative' in variant_scores:
            row['cri_multiplicative'] = float(variant_scores['cri_multiplicative'][idx])
        row['primary_cri'] = float(variant_scores[primary_key][idx])
        row['primary_cri_100'] = float(100.0 * row['primary_cri'])

    bucket_count = int(config.get('bucket_count', 5))
    variant_metrics = {
        key: _variant_metrics(key, scores, y, bucket_count) for key, scores in variant_scores.items()
    }

    holdout_metrics = {
        key: _variant_metrics(
            f'{key}_holdout',
            scores[eval_idx],
            y[eval_idx],
            bucket_count,
        )
        for key, scores in variant_scores.items()
    }
    summary = {
        'enabled': True,
        'formula_mode': formula_mode,
        'primary_formula': primary_formula,
        'primary_variant': primary_key,
        'weight_mode': weight_mode,
        'weights': learned_weights,
        'learning': {
            'fit_fraction': float(fit_fraction),
            'fit_size': int(len(fit_idx)),
            'holdout_size': int(len(eval_idx)),
            'diagnostics': learning_diagnostics,
        },
        'signals': {
            'uncertainty': {
                'entropy_weight': float(entropy_weight),
                'inverse_margin_weight': float(margin_weight),
                'mean_u': float(u.mean()),
            },
            'stability': {
                'fallback_stability': float(fallback_stability),
                'rows_with_task5_stability': int(sum(1 for row in per_example if str(row.get('row_id', '')) in stability_by_id)),
                'mean_s': float(s.mean()),
            },
        },
        'comparison_all': variant_metrics,
        'comparison_holdout': holdout_metrics,
    }
    return per_example, summary
