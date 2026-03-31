from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from models.mbti_ranker import MBTI_TYPES


class TransformerMBTIRanker:
    """Small transformer encoder with a classification head over the 16 MBTI types."""

    def __init__(
        self,
        *,
        model_name: str = 'distilbert-base-uncased',
        max_length: int = 256,
        learning_rate: float = 2.0e-5,
        weight_decay: float = 0.01,
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
        num_train_epochs: int = 1,
        warmup_ratio: float = 0.0,
        random_state: int = 42,
        max_train_samples: int | None = None,
        max_eval_samples: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.warmup_ratio = warmup_ratio
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples

        self.label2id = {label: idx for idx, label in enumerate(MBTI_TYPES)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.tokenizer = None
        self.model = None
        self.device = None

    def _load_runtime(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=False)
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(MBTI_TYPES),
                label2id=self.label2id,
                id2label=self.id2label,
                token=False,
            )
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        return torch

    def _prepare_subset(
        self,
        texts: Sequence[str],
        labels: Sequence[str] | None,
        *,
        limit: int | None,
    ) -> tuple[list[str], list[str] | None]:
        pairs = list(zip(texts, labels, strict=True)) if labels is not None else [(text, None) for text in texts]
        if limit is not None and limit > 0 and len(pairs) > limit:
            rng = random.Random(self.random_state)
            pairs = rng.sample(pairs, limit)
        sampled_texts = [text for text, _ in pairs]
        sampled_labels = None if labels is None else [str(label) for _, label in pairs]
        return sampled_texts, sampled_labels

    def _batch_tokenize(self, batch_texts: list[str]):
        assert self.tokenizer is not None
        encoded = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return encoded

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        val_texts: Sequence[str] | None = None,
        val_labels: Sequence[str] | None = None,
    ) -> 'TransformerMBTIRanker':
        torch = self._load_runtime()
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        train_texts, train_labels = self._prepare_subset(texts, labels, limit=self.max_train_samples)
        assert train_labels is not None
        if not train_texts:
            raise ValueError('Transformer baseline received an empty training set.')

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        steps_per_epoch = math.ceil(len(train_texts) / self.train_batch_size)
        total_steps = max(1, steps_per_epoch * self.num_train_epochs)
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        for epoch in range(self.num_train_epochs):
            self.model.train()
            indices = list(range(len(train_texts)))
            random.shuffle(indices)
            epoch_loss = 0.0

            for start in range(0, len(indices), self.train_batch_size):
                batch_indices = indices[start : start + self.train_batch_size]
                batch_texts = [train_texts[idx] for idx in batch_indices]
                batch_labels = torch.tensor(
                    [self.label2id[train_labels[idx].strip().upper()] for idx in batch_indices],
                    dtype=torch.long,
                    device=self.device,
                )
                encoded = self._batch_tokenize(batch_texts)
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_loss += float(loss.detach().cpu().item())

            avg_loss = epoch_loss / max(1, steps_per_epoch)
            print(f'[transformer_baseline] epoch={epoch + 1}/{self.num_train_epochs} loss={avg_loss:.4f}')
            if val_texts is not None and val_labels is not None:
                val_acc = self._accuracy(val_texts, val_labels)
                print(f'[transformer_baseline] epoch={epoch + 1} val_top_1_accuracy={val_acc:.4f}')

        return self

    def _accuracy(self, texts: Sequence[str], labels: Sequence[str]) -> float:
        rankings = self.predict_rankings(texts)
        correct = sum(1 for ranking, gold in zip(rankings, labels, strict=True) if ranking[0] == gold.strip().upper())
        return correct / len(rankings) if rankings else 0.0

    def predict_proba(self, texts: Sequence[str]) -> tuple[tuple[str, ...], np.ndarray]:
        torch = self._load_runtime()
        pred_texts, _ = self._prepare_subset(texts, None, limit=self.max_eval_samples)
        if len(pred_texts) != len(texts):
            pred_texts = list(texts)
        self.model.eval()
        probs_chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(pred_texts), self.eval_batch_size):
                batch_texts = pred_texts[start : start + self.eval_batch_size]
                encoded = self._batch_tokenize(batch_texts)
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                probs_chunks.append(probs)
        stacked = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0, len(MBTI_TYPES)))
        return MBTI_TYPES, stacked

    def predict_rankings(self, texts: Sequence[str]) -> list[list[str]]:
        labels, probs = self.predict_proba(texts)
        order = np.argsort(-probs, axis=1)
        return [[labels[idx] for idx in row] for row in order]

    def save(self, path: str | Path) -> Path:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('Cannot save transformer baseline before fitting it.')
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        metadata = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'train_batch_size': self.train_batch_size,
            'eval_batch_size': self.eval_batch_size,
            'num_train_epochs': self.num_train_epochs,
            'warmup_ratio': self.warmup_ratio,
            'random_state': self.random_state,
        }
        (output_dir / 'baseline_metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return output_dir
