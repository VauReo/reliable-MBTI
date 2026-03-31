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

## Task 6: Candidate Reliability Index (CRI)

The evaluator computes two CRI variants from normalized uncertainty and stability:

- Additive: `CRI_add = 1 - (w_u * U + w_s * (1 - S))`
- Multiplicative: `CRI_mul = (1 - U)^(w_u) * S^(w_s)`

`U` is derived from uncertainty signals (entropy and inverse margin).
`S` is derived from perturbation stability (Task 5), with configurable fallback when stability is unavailable.

### Weighting strategy

- Default mode is `learned` from a fit split of evaluation data.
- Constraints: `w_u >= 0`, `w_s >= 0`, `w_u + w_s = 1`.
- Learned weights and diagnostics are saved into `cri_metrics.json`.

### CRI effectiveness outputs

The evaluation writes:

- `per_example_predictions.jsonl` (includes `cri_additive`, `cri_multiplicative`, `primary_cri`)
- `cri_metrics.json` (weights, correlations, AUC, bucket analysis)
- `cri_comparison.png` (holdout comparison across CRI and baselines)

Metrics include Spearman correlation, holdout AUC, and bucketed correctness.
Comparisons include CRI vs uncertainty-only and stability-only signals.

## Ethics and limitations

- Not for hiring decisions.
- CRI measures reliability of the ranking signal, not candidate quality.
- MBTI labels are personality descriptors and should not be treated as validated predictors of job performance.
- Learned CRI weights can overfit on small samples; always report fit/holdout split and diagnostics.

## Mini model card (course version)

- Model family: TF-IDF + multinomial logistic regression ranker.
- Input: merged MBTI user posts text.
- Output: ranked MBTI list of 16 classes with probabilities.
- Reliability layer: uncertainty + perturbation stability + CRI.
- Intended use: research and Trustworthy AI coursework.
- Out-of-scope use: automated hiring or employment screening decisions.
