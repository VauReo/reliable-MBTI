# reliable-MBTI

This project implements MBTI ranking with reliability auditing.

## Run baseline + reliability audit

```bash
bash manage/run_pipeline.sh --config configs/pipeline.yaml
```

Or run only evaluation:

```bash
bash manage/run_eval.sh --config configs/eval.yaml
```

For the full one-command pipeline, including dataset download, split creation, baseline training, and all reliability metrics:

```bash
bash manage/run_pipeline.sh --config configs/pipeline.yaml
```

This single command runs:

- Kaggle dataset download into `data/raw/kaggle_mbti`
- preprocessing plus stratified `train/val/test` split creation
- baseline `TF-IDF + multinomial logistic regression` training
- ranking metrics plus uncertainty and perturbation-stability auditing

There is also a small Transformer baseline with a classification head:

```bash
bash manage/run_pipeline.sh --config configs/pipeline_transformer.yaml
```

That pipeline uses [`configs/eval_transformer.yaml`](/Users/eneminova/reliable-MBTI/configs/eval_transformer.yaml) and trains a `distilbert-base-uncased` classifier over the same train/val/test splits before running the same ranking and reliability evaluation.

Important config fields in [`configs/eval.yaml`](/Users/eneminova/reliable-MBTI/configs/eval.yaml):

- `data.train_source` and `data.eval_source`: train/test JSONL paths from preprocessing.
- `model.source`: `baseline` to fit a local baseline, or `predictions` to audit an existing `predictions.jsonl`.
- `model.baseline.type`: `tfidf` or `transformer`.
- `uncertainty.*`: calibration bins and uncertainty bucket count.
- `stability.*`: perturbation set, evaluated subset size, and top-k definition.

Saved artifacts under `output_dir` include:

- `metrics_summary.json`
- `uncertainty_metrics.json`
- `uncertainty_bucket_analysis.jsonl`
- `per_example_predictions.jsonl`
- `stability_metrics.json`
- `stability_by_perturbation.jsonl`
- `stability_examples.jsonl`

## Notes

- This is a reproduction scaffold, not a finished implementation.
- Python entry points live in `src/runners/`.
- Shell automation lives in `manage/`.
- Large data, weights, and outputs should stay out of Git.
