from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MBTI_TYPES = (
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP',
)


class MBTIRankerModel:
    """A lightweight TF-IDF + multinomial logistic ranker for MBTI scoring."""

    def __init__(
        self,
        model_id: str = 'tfidf_logreg',
        *,
        max_features: int | None = 50_000,
        min_df: int = 2,
        ngram_range: tuple[int, int] = (1, 2),
        c_value: float = 4.0,
        max_iter: int = 2_000,
        random_state: int = 42,
    ) -> None:
        self.model_id = model_id
        self.pipeline = Pipeline(
            steps=[
                (
                    'tfidf',
                    TfidfVectorizer(
                        max_features=max_features,
                        min_df=min_df,
                        ngram_range=ngram_range,
                        sublinear_tf=True,
                        strip_accents='unicode',
                    ),
                ),
                (
                    'clf',
                    LogisticRegression(
                        solver='lbfgs',
                        C=c_value,
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def describe(self) -> None:
        print(f"[model] configured model_id={self.model_id}")

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> 'MBTIRankerModel':
        self.pipeline.fit(list(texts), [label.strip().upper() for label in labels])
        return self

    def predict_proba(self, texts: Sequence[str]) -> tuple[tuple[str, ...], np.ndarray]:
        probs = self.pipeline.predict_proba(list(texts))
        observed = tuple(str(label).upper() for label in self.pipeline.named_steps['clf'].classes_)
        aligned = np.zeros((len(texts), len(MBTI_TYPES)), dtype=float)
        index_by_label = {label: idx for idx, label in enumerate(observed)}
        for col, label in enumerate(MBTI_TYPES):
            src = index_by_label.get(label)
            if src is not None:
                aligned[:, col] = probs[:, src]
        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        aligned /= row_sums
        return MBTI_TYPES, aligned

    def predict_rankings(self, texts: Sequence[str]) -> list[list[str]]:
        labels, probs = self.predict_proba(texts)
        order = np.argsort(-probs, axis=1)
        return [[labels[idx] for idx in row] for row in order]

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('wb') as handle:
            pickle.dump(self, handle)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> 'MBTIRankerModel':
        with Path(path).open('rb') as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f'Expected pickled {cls.__name__}, got {type(model)!r}')
        return model
