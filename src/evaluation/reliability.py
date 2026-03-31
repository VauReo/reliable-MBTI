from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sklearn.metrics import f1_score

from dataset.preprocessing import load_jsonl
from evaluation.cri import compute_cri
from losses.ranking_reward import MBTI_TYPES, reward_reciprocal_rank, reward_top_k, reward_top_1
from models.mbti_ranker import MBTIRankerModel
from models.transformer_ranker import TransformerMBTIRanker
from utils.io import ensure_dir

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional at runtime
    plt = None

ROOT = Path(__file__).resolve().parents[2]
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?\n]*", re.MULTILINE)
LEXICAL_SUBSTITUTIONS = {
    'because': 'since',
    'maybe': 'perhaps',
    'people': 'folks',
    'think': 'believe',
    'feel': 'sense',
    'good': 'positive',
    'bad': 'negative',
    'always': 'consistently',
    'often': 'frequently',
    'sometimes': 'occasionally',
    'really': 'quite',
    'very': 'highly',
    'want': 'prefer',
    'like': 'enjoy',
    'help': 'support',
    'friend': 'companion',
    'friends': 'companions',
    'job': 'role',
    'work': 'tasks',
}


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


@dataclass(slots=True)
class EvaluableExample:
    row_id: str
    label: str
    text: str
    raw: dict[str, Any]


@dataclass(slots=True)
class EvaluableDataset:
    examples: list[EvaluableExample]
    label_field: str
    text_field: str

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        label_field: str = 'label',
        text_field: str = 'text',
        id_field: str = 'row_id',
    ) -> 'EvaluableDataset':
        rows = load_jsonl(_resolve(path))
        examples: list[EvaluableExample] = []
        for idx, row in enumerate(rows):
            label = str(row[label_field]).strip().upper()
            text = str(row[text_field])
            row_id = str(row.get(id_field) or row.get('id') or row.get('source_row_index') or idx)
            examples.append(EvaluableExample(row_id=row_id, label=label, text=text, raw=row))
        return cls(examples=examples, label_field=label_field, text_field=text_field)

    def labels(self) -> list[str]:
        return [example.label for example in self.examples]

    def texts(self) -> list[str]:
        return [example.text for example in self.examples]


class Scorer(Protocol):
    def score_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        ...


def _scores_to_probabilities(scores: list[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1 or arr.size != len(MBTI_TYPES):
        raise ValueError(f'Expected {len(MBTI_TYPES)} scores, got shape {arr.shape}')
    if np.any(arr < 0.0) or not np.isfinite(arr).all():
        shifted = arr - float(np.max(arr))
        exp = np.exp(shifted)
        denom = exp.sum()
        return exp / denom if denom > 0.0 else np.full_like(exp, 1.0 / len(exp))
    total = float(arr.sum())
    if total <= 0.0:
        return np.full_like(arr, 1.0 / len(arr))
    return arr / total


def _ranking_from_probabilities(probs: np.ndarray) -> list[str]:
    order = np.argsort(-probs)
    return [MBTI_TYPES[idx] for idx in order]


class SklearnRankerScorer:
    def __init__(self, model: MBTIRankerModel) -> None:
        self.model = model

    @classmethod
    def from_config(cls, train_dataset: EvaluableDataset, config: dict[str, Any]) -> 'SklearnRankerScorer':
        vectorizer_cfg = config.get('vectorizer', {})
        classifier_cfg = config.get('classifier', {})
        ngram_raw = vectorizer_cfg.get('ngram_range', [1, 2])
        model = MBTIRankerModel(
            max_features=vectorizer_cfg.get('max_features', 50_000),
            min_df=int(vectorizer_cfg.get('min_df', 2)),
            ngram_range=(int(ngram_raw[0]), int(ngram_raw[1])),
            c_value=float(classifier_cfg.get('c_value', classifier_cfg.get('C', 4.0))),
            max_iter=int(classifier_cfg.get('max_iter', 2_000)),
            random_state=int(config.get('seed', 42)),
        )
        model.fit(train_dataset.texts(), train_dataset.labels())
        return cls(model)

    def save(self, path: str | Path) -> Path:
        return self.model.save(path)

    def score_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        _, probs = self.model.predict_proba(texts)
        outputs: list[dict[str, Any]] = []
        for row in probs:
            row_probs = np.asarray(row, dtype=float)
            outputs.append(
                {
                    'probabilities': {label: float(score) for label, score in zip(MBTI_TYPES, row_probs, strict=True)},
                    'scores': {label: float(score) for label, score in zip(MBTI_TYPES, row_probs, strict=True)},
                    'ranking': _ranking_from_probabilities(row_probs),
                }
            )
        return outputs


class TransformerRankerScorer:
    def __init__(self, model: TransformerMBTIRanker) -> None:
        self.model = model

    @classmethod
    def from_config(
        cls,
        train_dataset: EvaluableDataset,
        config: dict[str, Any],
        *,
        val_dataset: EvaluableDataset | None = None,
    ) -> 'TransformerRankerScorer':
        transformer_cfg = config.get('transformer', {})
        model = TransformerMBTIRanker(
            model_name=str(transformer_cfg.get('model_name', 'distilbert-base-uncased')),
            max_length=int(transformer_cfg.get('max_length', 256)),
            learning_rate=float(transformer_cfg.get('learning_rate', 2.0e-5)),
            weight_decay=float(transformer_cfg.get('weight_decay', 0.01)),
            train_batch_size=int(transformer_cfg.get('train_batch_size', 8)),
            eval_batch_size=int(transformer_cfg.get('eval_batch_size', 16)),
            num_train_epochs=int(transformer_cfg.get('num_train_epochs', 1)),
            warmup_ratio=float(transformer_cfg.get('warmup_ratio', 0.0)),
            random_state=int(config.get('seed', 42)),
            max_train_samples=(
                int(transformer_cfg['max_train_samples'])
                if transformer_cfg.get('max_train_samples') is not None
                else None
            ),
            max_eval_samples=(
                int(transformer_cfg['max_eval_samples'])
                if transformer_cfg.get('max_eval_samples') is not None
                else None
            ),
        )
        model.fit(
            train_dataset.texts(),
            train_dataset.labels(),
            val_texts=val_dataset.texts() if val_dataset is not None else None,
            val_labels=val_dataset.labels() if val_dataset is not None else None,
        )
        return cls(model)

    def save(self, path: str | Path) -> Path:
        return self.model.save(path)

    def score_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        _, probs = self.model.predict_proba(texts)
        outputs: list[dict[str, Any]] = []
        for row in probs:
            row_probs = np.asarray(row, dtype=float)
            outputs.append(
                {
                    'probabilities': {label: float(score) for label, score in zip(MBTI_TYPES, row_probs, strict=True)},
                    'scores': {label: float(score) for label, score in zip(MBTI_TYPES, row_probs, strict=True)},
                    'ranking': _ranking_from_probabilities(row_probs),
                }
            )
        return outputs


class JsonPredictionScorer:
    def __init__(self, predictions_by_id: dict[str, dict[str, Any]]) -> None:
        self.predictions_by_id = predictions_by_id

    @classmethod
    def from_jsonl(cls, path: str | Path) -> 'JsonPredictionScorer':
        rows = load_jsonl(_resolve(path))
        by_id: dict[str, dict[str, Any]] = {}
        for idx, row in enumerate(rows):
            row_id = str(row.get('row_id') or row.get('id') or row.get('source_row_index') or idx)
            by_id[row_id] = row
        return cls(by_id)

    def score_dataset(self, dataset: EvaluableDataset) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for example in dataset.examples:
            if example.row_id not in self.predictions_by_id:
                raise KeyError(f'Missing prediction for row_id={example.row_id}')
            outputs.append(self._normalize_prediction(self.predictions_by_id[example.row_id]))
        return outputs

    def score_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        raise RuntimeError('Prediction JSON mode cannot score arbitrary perturbed texts.')

    def _normalize_prediction(self, row: dict[str, Any]) -> dict[str, Any]:
        probs: np.ndarray
        if 'probabilities' in row and isinstance(row['probabilities'], dict):
            probs = np.asarray([float(row['probabilities'].get(label, 0.0)) for label in MBTI_TYPES])
        elif 'scores' in row and isinstance(row['scores'], dict):
            probs = _scores_to_probabilities([float(row['scores'].get(label, 0.0)) for label in MBTI_TYPES])
        elif 'scores' in row and isinstance(row['scores'], list):
            probs = _scores_to_probabilities([float(value) for value in row['scores']])
        elif 'ranked_mbti' in row:
            probs = np.full(len(MBTI_TYPES), 1e-6, dtype=float)
            for rank, label in enumerate(row['ranked_mbti']):
                label_u = str(label).strip().upper()
                if label_u in MBTI_TYPES:
                    probs[MBTI_TYPES.index(label_u)] = float(len(MBTI_TYPES) - rank)
            probs = _scores_to_probabilities(probs.tolist())
        else:
            raise ValueError('Prediction row must contain probabilities, scores, or ranked_mbti.')
        return {
            'probabilities': {label: float(score) for label, score in zip(MBTI_TYPES, probs, strict=True)},
            'scores': {label: float(score) for label, score in zip(MBTI_TYPES, probs, strict=True)},
            'ranking': _ranking_from_probabilities(probs),
        }


def _entropy(probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum() / math.log(len(MBTI_TYPES)))


def _margin(probabilities: np.ndarray) -> float:
    top2 = np.sort(probabilities)[-2:]
    return float(top2[-1] - top2[-2])


def _rank_of_label(ranking: list[str], label: str) -> int:
    label_u = label.strip().upper()
    try:
        return ranking.index(label_u) + 1
    except ValueError:
        return len(ranking) + 1


def _expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    *,
    bins: int,
) -> tuple[float, list[dict[str, float]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    details: list[dict[str, float]] = []
    ece = 0.0
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidences >= start) & (confidences < end if end < 1.0 else confidences <= end)
        count = int(mask.sum())
        if count == 0:
            details.append({'bin_start': float(start), 'bin_end': float(end), 'count': 0.0})
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc = float(correctness[mask].mean())
        weight = count / max(1, len(confidences))
        ece += abs(avg_acc - avg_conf) * weight
        details.append(
            {
                'bin_start': float(start),
                'bin_end': float(end),
                'count': float(count),
                'avg_confidence': avg_conf,
                'avg_accuracy': avg_acc,
                'gap': float(abs(avg_acc - avg_conf)),
            }
        )
    return float(ece), details


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _random_deletion(text: str, rng: random.Random, delete_ratio: float) -> str:
    tokens = TOKEN_RE.findall(text)
    if len(tokens) < 8:
        return text
    kept = [tok for tok in tokens if rng.random() > delete_ratio]
    if len(kept) < max(5, len(tokens) // 3):
        kept = tokens[: max(5, len(tokens) // 3)]
    return _untokenize(kept)


def _sentence_shuffle(text: str, rng: random.Random) -> str:
    sentences = [chunk.strip() for chunk in SENTENCE_RE.findall(text) if chunk.strip()]
    if len(sentences) < 2:
        return text
    rng.shuffle(sentences)
    return ' '.join(sentences)


def _lexical_substitution(text: str, rng: random.Random, max_replacements: int) -> str:
    tokens = TOKEN_RE.findall(text)
    replaced = 0
    output: list[str] = []
    for token in tokens:
        lookup = token.lower()
        replacement = LEXICAL_SUBSTITUTIONS.get(lookup)
        if replacement and replaced < max_replacements and rng.random() < 0.6:
            if token.istitle():
                replacement = replacement.title()
            elif token.isupper():
                replacement = replacement.upper()
            output.append(replacement)
            replaced += 1
        else:
            output.append(token)
    return _untokenize(output)


def _untokenize(tokens: list[str]) -> str:
    out = ''
    for token in tokens:
        if not out:
            out = token
        elif re.match(r"[^\w\s]", token):
            out += token
        elif out.endswith(('(', '[', '{', '\n')):
            out += token
        else:
            out += f' {token}'
    return out


def build_perturbation_fn(spec: dict[str, Any]):
    name = str(spec.get('name', '')).strip().lower()
    if name == 'random_deletion':
        delete_ratio = float(spec.get('delete_ratio', 0.1))
        return lambda text, rng: _random_deletion(text, rng, delete_ratio)
    if name == 'sentence_shuffle':
        return lambda text, rng: _sentence_shuffle(text, rng)
    if name in {'lexical_substitution', 'paraphrase_lite'}:
        max_replacements = int(spec.get('max_replacements', 12))
        return lambda text, rng: _lexical_substitution(text, rng, max_replacements)
    raise ValueError(f'Unsupported perturbation name: {name}')


class ReliabilityEvaluator:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        seed: int = 42,
    ) -> None:
        self.output_dir = ensure_dir(_resolve(output_dir))
        self.seed = seed

    def evaluate(
        self,
        dataset: EvaluableDataset,
        scored_rows: list[dict[str, Any]],
        *,
        scorer: Scorer | None,
        uncertainty_cfg: dict[str, Any],
        stability_cfg: dict[str, Any],
        cri_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(dataset.examples) != len(scored_rows):
            raise ValueError('Dataset and scored rows must have the same length.')

        per_example = self._build_per_example_records(dataset, scored_rows)

        if uncertainty_cfg.get('enabled', True):
            uncertainty_metrics = self._compute_uncertainty(per_example, uncertainty_cfg)
        else:
            uncertainty_metrics = {'enabled': False, 'reason': 'uncertainty disabled', 'bucket_analysis': []}
        self._write_json(self.output_dir / 'uncertainty_metrics.json', uncertainty_metrics)
        self._write_jsonl(
            self.output_dir / 'uncertainty_bucket_analysis.jsonl',
            uncertainty_metrics['bucket_analysis'],
        )

        stability_metrics: dict[str, Any] = {'enabled': False, 'reason': 'stability disabled'}
        if stability_cfg.get('enabled', False):
            if scorer is None:
                stability_metrics = {
                    'enabled': False,
                    'reason': 'stability audit requires a live scorer, not offline prediction JSON.',
                }
            else:
                stability_metrics = self._compute_stability(dataset, per_example, scorer, stability_cfg)
            self._write_json(self.output_dir / 'stability_metrics.json', stability_metrics)

        cri_cfg = cri_cfg or {}
        cri_enabled = bool(cri_cfg.get('enabled', True))
        if cri_enabled:
            per_example, cri_metrics = compute_cri(
                per_example,
                stability_metrics=stability_metrics,
                config=cri_cfg,
                seed=self.seed,
            )
        else:
            cri_metrics = {'enabled': False, 'reason': 'cri disabled'}
        self._write_jsonl(self.output_dir / 'per_example_predictions.jsonl', per_example)
        self._write_json(self.output_dir / 'cri_metrics.json', cri_metrics)

        summary = {
            'num_examples': len(per_example),
            'top_1_accuracy': _safe_mean([float(row['top_1_correct']) for row in per_example]),
            'top_3_accuracy': _safe_mean([float(row['top_3_correct']) for row in per_example]),
            'mrr': _safe_mean([float(row['reciprocal_rank']) for row in per_example]),
            'ndcg_at_5': _safe_mean([float(row['ndcg_at_5']) for row in per_example]),
            'macro_f1': self._compute_macro_f1(per_example),
            'weighted_f1': self._compute_weighted_f1(per_example),
            'uncertainty': uncertainty_metrics,
            'stability': stability_metrics,
            'cri': cri_metrics,
        }
        self._write_json(self.output_dir / 'metrics_summary.json', summary)
        self._maybe_plot_uncertainty(per_example, uncertainty_metrics)
        self._maybe_plot_quality_slices(per_example)
        self._maybe_plot_stability(stability_metrics)
        self._maybe_plot_cri(summary.get('cri', {}))
        return summary

    def _build_per_example_records(
        self,
        dataset: EvaluableDataset,
        scored_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for example, scored in zip(dataset.examples, scored_rows, strict=True):
            probs = np.asarray([float(scored['probabilities'][label]) for label in MBTI_TYPES], dtype=float)
            ranking = scored.get('ranking') or _ranking_from_probabilities(probs)
            pred = ranking[0]
            rr = reward_reciprocal_rank(ranking, example.label, k=16)
            records.append(
                {
                    'row_id': example.row_id,
                    'label': example.label,
                    'prediction': pred,
                    'ranking': ranking,
                    'probabilities': {label: float(score) for label, score in zip(MBTI_TYPES, probs, strict=True)},
                    'confidence': float(probs[MBTI_TYPES.index(pred)]),
                    'entropy': _entropy(probs),
                    'margin': _margin(probs),
                    'top_1_correct': bool(reward_top_1(ranking, example.label)),
                    'top_3_correct': bool(reward_top_k(ranking, example.label, top_k=3)),
                    'reciprocal_rank': float(rr),
                    'ndcg_at_5': float(rr),
                    'true_label_rank': _rank_of_label(ranking, example.label),
                    'text_length_chars': len(example.text),
                    'text_length_tokens': len(TOKEN_RE.findall(example.text)),
                }
            )
        return records

    def _compute_uncertainty(self, per_example: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        bins = int(config.get('calibration_bins', 10))
        confidences = np.asarray([row['confidence'] for row in per_example], dtype=float)
        correctness = np.asarray([1.0 if row['top_1_correct'] else 0.0 for row in per_example], dtype=float)
        entropies = [float(row['entropy']) for row in per_example]
        margins = [float(row['margin']) for row in per_example]
        incorrect = [row for row in per_example if not row['top_1_correct']]
        correct = [row for row in per_example if row['top_1_correct']]
        ece, calibration = _expected_calibration_error(confidences, correctness, bins=bins)

        sorted_rows = sorted(per_example, key=lambda item: item['entropy'], reverse=True)
        bucket_count = int(config.get('uncertainty_buckets', 5))
        bucket_size = max(1, math.ceil(len(sorted_rows) / bucket_count))
        bucket_analysis: list[dict[str, Any]] = []
        for idx in range(bucket_count):
            chunk = sorted_rows[idx * bucket_size : (idx + 1) * bucket_size]
            if not chunk:
                continue
            bucket_analysis.append(
                {
                    'bucket': idx + 1,
                    'size': len(chunk),
                    'avg_entropy': _safe_mean([float(row['entropy']) for row in chunk]),
                    'avg_margin': _safe_mean([float(row['margin']) for row in chunk]),
                    'accuracy': _safe_mean([1.0 if row['top_1_correct'] else 0.0 for row in chunk]),
                    'avg_confidence': _safe_mean([float(row['confidence']) for row in chunk]),
                }
            )

        return {
            'enabled': True,
            'ece': ece,
            'avg_confidence': float(confidences.mean()) if len(confidences) else 0.0,
            'accuracy': float(correctness.mean()) if len(correctness) else 0.0,
            'avg_entropy': _safe_mean(entropies),
            'avg_margin': _safe_mean(margins),
            'avg_entropy_correct': _safe_mean([float(row['entropy']) for row in correct]),
            'avg_entropy_incorrect': _safe_mean([float(row['entropy']) for row in incorrect]),
            'avg_margin_correct': _safe_mean([float(row['margin']) for row in correct]),
            'avg_margin_incorrect': _safe_mean([float(row['margin']) for row in incorrect]),
            'calibration_curve': calibration,
            'bucket_analysis': bucket_analysis,
        }

    def _compute_stability(
        self,
        dataset: EvaluableDataset,
        per_example: list[dict[str, Any]],
        scorer: Scorer,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        rng = random.Random(self.seed)
        top_k = int(config.get('top_k', 3))
        subset_size = min(int(config.get('subset_size', len(dataset.examples))), len(dataset.examples))
        indexed_examples = list(zip(dataset.examples, per_example, strict=True))
        subset = indexed_examples if subset_size >= len(indexed_examples) else rng.sample(indexed_examples, subset_size)

        perturbation_specs = config.get('perturbations', [])
        perturbation_rows: list[dict[str, Any]] = []
        perturbation_summary: list[dict[str, Any]] = []
        per_example_stability: dict[str, list[float]] = {}
        save_examples = int(config.get('save_example_count', 30))

        for spec in perturbation_specs:
            name = str(spec.get('name', 'unknown'))
            perturb = build_perturbation_fn(spec)
            perturbed_texts = [perturb(example.text, rng) for example, _ in subset]
            perturbed_scores = scorer.score_texts(perturbed_texts)
            top1_same: list[float] = []
            topk_same: list[float] = []
            rr_delta: list[float] = []

            for index, ((example, baseline), perturbed_text, scored) in enumerate(
                zip(subset, perturbed_texts, perturbed_scores, strict=True)
            ):
                probs = np.asarray([float(scored['probabilities'][label]) for label in MBTI_TYPES], dtype=float)
                ranking = scored.get('ranking') or _ranking_from_probabilities(probs)
                base_topk = baseline['ranking'][:top_k]
                new_topk = ranking[:top_k]
                same_top1 = float(ranking[0] == baseline['prediction'])
                same_topk = float(set(base_topk) == set(new_topk))
                pert_rr = reward_reciprocal_rank(ranking, example.label, k=16)
                top1_same.append(same_top1)
                topk_same.append(same_topk)
                rr_delta.append(float(pert_rr - baseline['reciprocal_rank']))
                per_example_stability.setdefault(example.row_id, []).append((same_top1 + same_topk) / 2.0)
                if index < save_examples:
                    perturbation_rows.append(
                        {
                            'perturbation': name,
                            'row_id': example.row_id,
                            'label': example.label,
                            'original_prediction': baseline['prediction'],
                            'perturbed_prediction': ranking[0],
                            'top_1_changed': bool(not same_top1),
                            f'top_{top_k}_set_changed': bool(not same_topk),
                            'original_text_preview': example.text[:400],
                            'perturbed_text_preview': perturbed_text[:400],
                            'original_top_k': base_topk,
                            'perturbed_top_k': new_topk,
                        }
                    )
            perturbation_summary.append(
                {
                    'perturbation': name,
                    'num_examples': len(subset),
                    'top_1_flip_rate': 1.0 - _safe_mean(top1_same),
                    'top_k_consistency': _safe_mean(topk_same),
                    f'top_{top_k}_consistency': _safe_mean(topk_same),
                    'mean_reciprocal_rank_delta': _safe_mean(rr_delta),
                }
            )

        self._write_jsonl(self.output_dir / 'stability_examples.jsonl', perturbation_rows)
        self._write_jsonl(self.output_dir / 'stability_by_perturbation.jsonl', perturbation_summary)
        per_example_rows = [
            {
                'row_id': row_id,
                'stability_score': _safe_mean(values),
            }
            for row_id, values in per_example_stability.items()
        ]
        self._write_jsonl(self.output_dir / 'stability_per_example.jsonl', per_example_rows)
        return {
            'enabled': True,
            'evaluated_examples': len(subset),
            'top_k': top_k,
            'perturbations': perturbation_summary,
            'per_example': per_example_rows,
        }

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        ensure_dir(path.parent)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        ensure_dir(path.parent)
        with path.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    def _compute_macro_f1(self, per_example: list[dict[str, Any]]) -> float:
        if not per_example:
            return 0.0
        y_true = [row['label'] for row in per_example]
        y_pred = [row['prediction'] for row in per_example]
        return float(f1_score(y_true, y_pred, labels=list(MBTI_TYPES), average='macro', zero_division=0.0))

    def _compute_weighted_f1(self, per_example: list[dict[str, Any]]) -> float:
        if not per_example:
            return 0.0
        y_true = [row['label'] for row in per_example]
        y_pred = [row['prediction'] for row in per_example]
        return float(
            f1_score(
                y_true,
                y_pred,
                labels=list(MBTI_TYPES),
                average='weighted',
                zero_division=0.0,
            )
        )

    def _maybe_plot_uncertainty(
        self,
        per_example: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> None:
        if plt is None or not metrics.get('enabled'):
            return
        correct = [row['entropy'] for row in per_example if row['top_1_correct']]
        incorrect = [row['entropy'] for row in per_example if not row['top_1_correct']]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(correct, bins=20, alpha=0.7, label='correct')
        ax.hist(incorrect, bins=20, alpha=0.7, label='incorrect')
        ax.set_title('Entropy by correctness')
        ax.set_xlabel('Normalized entropy')
        ax.set_ylabel('Count')
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / 'uncertainty_entropy_hist.png', dpi=160)
        plt.close(fig)

        calibration = metrics.get('calibration_curve', [])
        xs = [row.get('avg_confidence', 0.0) for row in calibration if row.get('count', 0.0) > 0]
        ys = [row.get('avg_accuracy', 0.0) for row in calibration if row.get('count', 0.0) > 0]
        if xs and ys:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.plot(xs, ys, marker='o')
            ax.set_title('Reliability diagram')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy')
            fig.tight_layout()
            fig.savefig(self.output_dir / 'reliability_diagram.png', dpi=160)
            plt.close(fig)

        correct_margin = [row['margin'] for row in per_example if row['top_1_correct']]
        incorrect_margin = [row['margin'] for row in per_example if not row['top_1_correct']]
        if correct_margin and incorrect_margin:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot([correct_margin, incorrect_margin], labels=['correct', 'incorrect'])
            ax.set_title('Margin by correctness')
            ax.set_ylabel('Top-1 minus Top-2 probability')
            fig.tight_layout()
            fig.savefig(self.output_dir / 'margin_by_correctness_boxplot.png', dpi=160)
            plt.close(fig)

        bucket_rows = metrics.get('bucket_analysis', [])
        if bucket_rows:
            fig, ax = plt.subplots(figsize=(8, 4))
            xs = [str(row['bucket']) for row in bucket_rows]
            acc = [row['accuracy'] for row in bucket_rows]
            conf = [row['avg_confidence'] for row in bucket_rows]
            ax.bar(xs, acc, alpha=0.8, label='accuracy')
            ax.plot(xs, conf, marker='o', color='darkorange', linewidth=2, label='avg confidence')
            ax.set_title('Accuracy by uncertainty bucket')
            ax.set_xlabel('Bucket (1 = most uncertain)')
            ax.set_ylim(0.0, 1.0)
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / 'accuracy_by_uncertainty_bucket.png', dpi=160)
            plt.close(fig)

        non_empty_calibration = [row for row in calibration if row.get('count', 0.0) > 0]
        if non_empty_calibration:
            labels = [f"{row['bin_start']:.1f}-{row['bin_end']:.1f}" for row in non_empty_calibration]
            conf = [row.get('avg_confidence', 0.0) for row in non_empty_calibration]
            acc = [row.get('avg_accuracy', 0.0) for row in non_empty_calibration]
            fig, ax = plt.subplots(figsize=(9, 4))
            x = np.arange(len(labels))
            ax.bar(x - 0.2, conf, width=0.4, label='confidence')
            ax.bar(x + 0.2, acc, width=0.4, label='accuracy')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha='right')
            ax.set_ylim(0.0, 1.0)
            ax.set_title('Confidence vs accuracy by calibration bin')
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / 'confidence_accuracy_by_bin.png', dpi=160)
            plt.close(fig)

    def _maybe_plot_quality_slices(self, per_example: list[dict[str, Any]]) -> None:
        if plt is None or not per_example:
            return
        self._plot_text_length_bucket_accuracy(per_example)
        self._plot_mbti_type_accuracy(per_example)
        self._plot_confusion_matrix(per_example)
        self._plot_true_label_rank_distribution(per_example)
        self._plot_confidence_vs_entropy_scatter(per_example)

    def _plot_text_length_bucket_accuracy(self, per_example: list[dict[str, Any]]) -> None:
        token_lengths = np.asarray([row['text_length_tokens'] for row in per_example], dtype=float)
        if len(token_lengths) < 4:
            return
        quantiles = np.quantile(token_lengths, [0.0, 0.25, 0.5, 0.75, 1.0])
        bucket_labels: list[str] = []
        accuracies: list[float] = []
        counts: list[int] = []
        for idx, (start, end) in enumerate(zip(quantiles[:-1], quantiles[1:], strict=True)):
            if idx == len(quantiles) - 2:
                rows = [row for row in per_example if start <= row['text_length_tokens'] <= end]
            else:
                rows = [row for row in per_example if start <= row['text_length_tokens'] < end]
            if not rows:
                continue
            bucket_labels.append(f'{int(start)}-{int(end)}')
            accuracies.append(_safe_mean([1.0 if row['top_1_correct'] else 0.0 for row in rows]))
            counts.append(len(rows))
        if not bucket_labels:
            return
        fig, ax1 = plt.subplots(figsize=(8, 4))
        positions = np.arange(len(bucket_labels))
        ax1.bar(positions, accuracies, color='steelblue', alpha=0.85)
        ax1.set_ylim(0.0, 1.0)
        ax1.set_ylabel('Top-1 accuracy')
        ax1.set_xlabel('Text length bucket (tokens)')
        ax1.set_xticks(positions)
        ax1.set_xticklabels(bucket_labels)
        ax2 = ax1.twinx()
        ax2.plot(positions, counts, color='darkorange', marker='o')
        ax2.set_ylabel('Examples')
        ax1.set_title('Accuracy by text length bucket')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'accuracy_by_text_length_bucket.png', dpi=160)
        plt.close(fig)

    def _plot_mbti_type_accuracy(self, per_example: list[dict[str, Any]]) -> None:
        labels: list[str] = []
        accuracies: list[float] = []
        for mbti in MBTI_TYPES:
            rows = [row for row in per_example if row['label'] == mbti]
            if not rows:
                continue
            labels.append(mbti)
            accuracies.append(_safe_mean([1.0 if row['top_1_correct'] else 0.0 for row in rows]))
        if not labels:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(labels, accuracies, color='seagreen', alpha=0.85)
        ax.set_ylim(0.0, 1.0)
        ax.set_title('Top-1 accuracy by MBTI type')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=35)
        fig.tight_layout()
        fig.savefig(self.output_dir / 'accuracy_by_mbti_type.png', dpi=160)
        plt.close(fig)

    def _plot_confusion_matrix(self, per_example: list[dict[str, Any]]) -> None:
        matrix = np.zeros((len(MBTI_TYPES), len(MBTI_TYPES)), dtype=int)
        index_by_label = {label: idx for idx, label in enumerate(MBTI_TYPES)}
        for row in per_example:
            matrix[index_by_label[row['label']], index_by_label[row['prediction']]] += 1
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(matrix, cmap='Blues')
        ax.set_xticks(np.arange(len(MBTI_TYPES)))
        ax.set_yticks(np.arange(len(MBTI_TYPES)))
        ax.set_xticklabels(MBTI_TYPES, rotation=45, ha='right')
        ax.set_yticklabels(MBTI_TYPES)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Top-1 confusion matrix')
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(self.output_dir / 'confusion_matrix_top1.png', dpi=160)
        plt.close(fig)

    def _plot_true_label_rank_distribution(self, per_example: list[dict[str, Any]]) -> None:
        ranks = [min(int(row['true_label_rank']), 10) for row in per_example]
        bins = np.arange(1, 12) - 0.5
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(ranks, bins=bins, color='mediumpurple', alpha=0.85, rwidth=0.9)
        ax.set_xticks(list(range(1, 11)))
        ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10+'])
        ax.set_xlabel('Rank of true label')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of the true label rank')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'true_label_rank_distribution.png', dpi=160)
        plt.close(fig)

    def _plot_confidence_vs_entropy_scatter(self, per_example: list[dict[str, Any]]) -> None:
        x = [row['confidence'] for row in per_example]
        y = [row['entropy'] for row in per_example]
        colors = ['tab:green' if row['top_1_correct'] else 'tab:red' for row in per_example]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, c=colors, alpha=0.45, s=18)
        ax.set_xlabel('Top-1 confidence')
        ax.set_ylabel('Normalized entropy')
        ax.set_title('Confidence vs entropy')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'confidence_vs_entropy_scatter.png', dpi=160)
        plt.close(fig)

    def _maybe_plot_stability(self, metrics: dict[str, Any]) -> None:
        if plt is None or not metrics.get('enabled'):
            return
        rows = metrics.get('perturbations', [])
        if not rows:
            return
        names = [row['perturbation'] for row in rows]
        flip_rates = [row['top_1_flip_rate'] for row in rows]
        consistencies = [row.get('top_k_consistency', 0.0) for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        positions = np.arange(len(names))
        ax.bar(positions - 0.2, flip_rates, width=0.4, label='top-1 flip rate')
        ax.bar(positions + 0.2, consistencies, width=0.4, label='top-k consistency')
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=20)
        ax.set_ylim(0.0, 1.0)
        ax.set_title('Stability audit')
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / 'stability_audit.png', dpi=160)
        plt.close(fig)

    def _maybe_plot_cri(self, metrics: dict[str, Any]) -> None:
        if plt is None or not metrics.get('enabled'):
            return
        comparison = metrics.get('comparison_holdout', {})
        if not isinstance(comparison, dict) or not comparison:
            return
        keys = list(comparison.keys())
        auc_values = [float(comparison[key].get('auc_correctness', 0.0)) for key in keys]
        corr_values = [
            float(comparison[key].get('spearman_with_correctness', 0.0)) for key in keys
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        pos = np.arange(len(keys))

        axes[0].bar(pos, auc_values, color='#2563eb')
        axes[0].set_title('CRI Variant AUC (holdout)')
        axes[0].set_xticks(pos)
        axes[0].set_xticklabels(keys, rotation=30, ha='right')
        axes[0].set_ylim(0.0, 1.0)

        axes[1].bar(pos, corr_values, color='#7c3aed')
        axes[1].set_title('CRI Variant Spearman (holdout)')
        axes[1].set_xticks(pos)
        axes[1].set_xticklabels(keys, rotation=30, ha='right')
        axes[1].set_ylim(-1.0, 1.0)

        fig.tight_layout()
        fig.savefig(self.output_dir / 'cri_comparison.png', dpi=160)
        plt.close(fig)
