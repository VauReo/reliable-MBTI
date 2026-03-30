from __future__ import annotations

from pathlib import Path

from executors.base import RunContext
from evaluation.reliability import (
    EvaluableDataset,
    JsonPredictionScorer,
    ReliabilityEvaluator,
    SklearnRankerScorer,
)


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _require_existing_file(path_value: str, *, hint: str) -> None:
    path = _resolve_project_path(path_value)
    if path.exists():
        return
    raise FileNotFoundError(f"Required file not found: {path}\n{hint}")


class EvaluationExecutor:
    def __init__(self, context: RunContext, config: dict) -> None:
        self.context = context
        self.config = config

    def run(self) -> None:
        print(f"[eval] experiment={self.context.experiment_name}")
        print(f"[eval] output_dir={self.context.output_dir}")

        data_cfg = self.config.get("data", {})
        label_field = str(data_cfg.get("label_field", "label"))
        text_field = str(data_cfg.get("text_field", "text"))
        train_source = data_cfg.get("train_source")
        eval_source = data_cfg.get("eval_source") or self.config.get("label_source")
        if not eval_source:
            raise ValueError(
                "Evaluation config requires data.eval_source or label_source."
            )
        _require_existing_file(
            str(eval_source),
            hint=(
                "Evaluation split is missing. Run "
                "`bash manage/run_prepare_data.sh --config configs/data.yaml` "
                "after placing the raw MBTI CSV into `data/raw/kaggle_mbti/`."
            ),
        )

        eval_dataset = EvaluableDataset.from_jsonl(
            eval_source,
            label_field=label_field,
            text_field=text_field,
        )

        model_cfg = self.config.get("model", {})
        source_kind = str(model_cfg.get("source", "baseline")).strip().lower()
        scorer = None

        if source_kind == "baseline":
            if not train_source:
                raise ValueError("Baseline eval requires data.train_source.")
            _require_existing_file(
                str(train_source),
                hint=(
                    "Training split is missing. Run "
                    "`bash manage/run_prepare_data.sh --config configs/data.yaml` "
                    "to build the train/val/test JSONL files."
                ),
            )
            train_dataset = EvaluableDataset.from_jsonl(
                train_source,
                label_field=label_field,
                text_field=text_field,
            )
            scorer = SklearnRankerScorer.from_config(
                train_dataset,
                {
                    **model_cfg.get("baseline", {}),
                    "seed": self.config.get("seed", 42),
                },
            )
            scored_rows = scorer.score_texts(eval_dataset.texts())
            artifact_path = scorer.save(
                f"{self.context.output_dir}/baseline_ranker.pkl"
            )
            print(f"[eval] saved baseline ranker to {artifact_path}")
        elif source_kind == "predictions":
            prediction_source = model_cfg.get("prediction_source") or self.config.get(
                "prediction_source"
            )
            if not prediction_source:
                raise ValueError(
                    "Prediction eval requires model.prediction_source or prediction_source."
                )
            _require_existing_file(
                str(prediction_source),
                hint=(
                    "Prediction JSONL is missing. Update `model.prediction_source` "
                    "to an existing file."
                ),
            )
            json_scorer = JsonPredictionScorer.from_jsonl(prediction_source)
            scored_rows = json_scorer.score_dataset(eval_dataset)
        else:
            raise ValueError(f"Unsupported model.source={source_kind!r}")

        evaluator = ReliabilityEvaluator(
            output_dir=self.context.output_dir,
            seed=int(self.config.get("seed", 42)),
        )
        summary = evaluator.evaluate(
            eval_dataset,
            scored_rows,
            scorer=scorer,
            uncertainty_cfg=self.config.get("uncertainty", {}),
            stability_cfg=self.config.get("stability", {}),
        )

        print(f"[eval] top_1_accuracy={summary['top_1_accuracy']:.4f}")
        print(f"[eval] top_3_accuracy={summary['top_3_accuracy']:.4f}")
        print(f"[eval] mrr={summary['mrr']:.4f}")
        print(f"[eval] ndcg_at_5={summary['ndcg_at_5']:.4f}")
        print(f"[eval] uncertainty_ece={summary['uncertainty']['ece']:.4f}")
        if summary["stability"].get("enabled"):
            for row in summary["stability"].get("perturbations", []):
                print(
                    f"[eval] perturbation={row['perturbation']} "
                    f"top_1_flip_rate={row['top_1_flip_rate']:.4f}"
                )
        else:
            print(
                f"[eval] stability skipped: {summary['stability'].get('reason', 'disabled')}"
            )
