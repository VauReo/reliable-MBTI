from __future__ import annotations

import html
import inspect
import json
import os
import platform
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset.sft_generation import (
    generate_teacher_replies_with_fallback,
    load_jsonl,
    parse_ranked_mbti,
    resolve_path,
)
from executors.base import RunContext
from utils.io import ensure_dir
from tqdm import tqdm

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover
    class TrainerCallback:  # type: ignore[no-redef]
        pass


@dataclass(slots=True)
class ValidationExample:
    label: str
    prompt: str
    source_row_index: int | None = None


class RankingEvalCallback(TrainerCallback):
    def __init__(self, executor: 'SFTExecutor') -> None:
        self.executor = executor
        self.trainer = None
        self.last_evaluated_step = -1

    def bind_trainer(self, trainer: Any) -> None:
        self.trainer = trainer

    def _run_if_due(self, state) -> None:
        if self.trainer is None:
            return
        eval_steps = self.executor.train_cfg['eval_steps']
        if eval_steps <= 0 or state.global_step <= 0:
            return
        if state.global_step % eval_steps != 0:
            return
        if state.global_step == self.last_evaluated_step:
            return
        metrics = self.executor.run_ranking_evaluation(
            model=self.trainer.model,
            step=state.global_step,
            epoch=state.epoch,
        )
        if metrics:
            self.trainer.log(metrics)
            self.executor.maybe_save_best_checkpoints(
                self.trainer,
                metrics=metrics,
                step=state.global_step,
                epoch=state.epoch,
            )
            self.last_evaluated_step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        self._run_if_due(state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.trainer is None or not self.executor.validation_examples:
            return control
        if state.global_step == self.last_evaluated_step:
            return control
        metrics = self.executor.run_ranking_evaluation(
            model=self.trainer.model,
            step=state.global_step,
            epoch=state.epoch,
        )
        if metrics:
            self.trainer.log(metrics)
            self.executor.maybe_save_best_checkpoints(
                self.trainer,
                metrics=metrics,
                step=state.global_step,
                epoch=state.epoch,
            )
            self.last_evaluated_step = state.global_step
        return control


class SFTExecutor:
    def __init__(self, context: RunContext, config: dict[str, Any]) -> None:
        self.context = context
        self.config = config
        self.seed = int(config.get('seed', 42))

        self.model_cfg = self._normalize_model_config(config.get('model', {}))
        self.data_cfg = self._normalize_data_config(config.get('data', {}))
        self.train_cfg = self._normalize_training_config(config.get('training', {}))
        self.lora_cfg = self._normalize_lora_config(
            config.get('lora', {}),
            config.get('training', {}),
        )
        self.eval_cfg = self._normalize_eval_config(config.get('evaluation', {}))

        self.output_dir = self.train_cfg['output_dir']
        ensure_dir(self.output_dir)
        self.prompt_template = self._load_prompt_template(self.data_cfg['prompt_template_path'])
        self.validation_examples = self._prepare_validation_examples()
        self.eval_history_path = self.output_dir / 'ranking_eval_history.jsonl'
        self.eval_generations_dir = self.output_dir / 'eval_generations'
        self.figures_dir = self.output_dir / 'figures'
        self.best_checkpoints_dir = self.output_dir / 'best_checkpoints'
        self.best_metric_state = self._init_best_metric_state()
        self.tokenizer = None
        self.runtime_info: dict[str, Any] = {}

    def run(self) -> None:
        deps = self._load_runtime_dependencies()
        torch = deps['torch']
        Dataset = deps['Dataset']
        TrainingArguments = deps['TrainingArguments']
        AutoModelForCausalLM = deps['AutoModelForCausalLM']
        AutoTokenizer = deps['AutoTokenizer']
        SFTTrainer = deps['SFTTrainer']
        LoraConfig = deps['LoraConfig']
        PeftModel = deps['PeftModel']
        TaskType = deps['TaskType']

        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        model, tokenizer = self._load_model_and_tokenizer(
            torch=torch,
            auto_model_cls=AutoModelForCausalLM,
            auto_tokenizer_cls=AutoTokenizer,
        )
        self.tokenizer = tokenizer
        self.runtime_info = self._collect_runtime_info(torch, model)

        run_type = self.train_cfg['type']
        if run_type == 'eval':
            model = self._load_eval_model(model, PeftModel)
            model = self._prepare_eval_model_device(torch, model)
            self.runtime_info = self._collect_runtime_info(torch, model)
            param_stats = self._parameter_counts(model)
            self._write_runtime_artifacts()
            print(f'[train_sft] experiment={self.context.experiment_name}')
            print(f'[train_sft] output_dir={self.output_dir}')
            print(f'[train_sft] model_source={self.model_cfg["model_source"]}')
            print(f'[train_sft] runtime={self._format_runtime_summary()}')
            print(f'[train_sft] val_rows={len(self.validation_examples)}')
            print(
                '[train_sft] learnable_params='
                f'{param_stats["learnable_params"]:,} / total_params={param_stats["total_params"]:,} '
                f'({param_stats["trainable_ratio"]:.2%} trainable)'
            )
            metrics = self.run_ranking_evaluation(model=model, step=0, epoch=None)
            self._write_training_graphs([])
            print(f'[train_sft] eval metrics={metrics}')
            print(f'[train_sft] evaluation artifacts saved to {self.output_dir}')
            return

        train_dataset = self._build_train_dataset(Dataset, tokenizer)
        training_args = self._build_training_args(TrainingArguments, torch)
        peft_config = self._build_peft_config(LoraConfig, TaskType)
        trainer = self._build_trainer(
            trainer_cls=SFTTrainer,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            training_args=training_args,
            peft_config=peft_config,
        )

        ranking_callback = RankingEvalCallback(self)
        trainer.add_callback(ranking_callback)
        ranking_callback.bind_trainer(trainer)

        param_stats = self._parameter_counts(trainer.model)
        self._write_runtime_artifacts()
        print(f'[train_sft] experiment={self.context.experiment_name}')
        print(f'[train_sft] output_dir={self.output_dir}')
        print(f'[train_sft] model_source={self.model_cfg["model_source"]}')
        print(f'[train_sft] runtime={self._format_runtime_summary()}')
        print(f'[train_sft] train_rows={len(train_dataset)}')
        print(f'[train_sft] val_rows={len(self.validation_examples)}')
        print(
            '[train_sft] learnable_params='
            f'{param_stats["learnable_params"]:,} / total_params={param_stats["total_params"]:,} '
            f'({param_stats["trainable_ratio"]:.2%} trainable)'
        )
        print(
            '[train_sft] logging_steps={logging_steps} eval_steps={eval_steps} '
            'save_steps={save_steps}'.format(**self.train_cfg)
        )

        resume_checkpoint = self.train_cfg['resume_from_checkpoint']
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        self._write_training_graphs(trainer.state.log_history)

        print('[train_sft] training complete')
        print(f'[train_sft] final artifacts saved to {self.output_dir}')

    def _normalize_model_config(self, config: dict[str, Any]) -> dict[str, Any]:
        model_id = str(config.get('student_model_id', '')).strip()
        tokenizer_id = str(config.get('tokenizer_id', '')).strip() or model_id
        local_weights_dir = config.get('local_weights_dir')
        local_weights_path = resolve_path(local_weights_dir) if local_weights_dir else None
        model_source = model_id
        tokenizer_source = tokenizer_id
        if local_weights_path and local_weights_path.exists():
            model_source = str(local_weights_path)
            if not config.get('tokenizer_id'):
                tokenizer_source = str(local_weights_path)

        if not model_source:
            raise ValueError('model.student_model_id must be provided')
        if not tokenizer_source:
            raise ValueError('model.tokenizer_id must be provided')

        return {
            'student_model_id': model_id,
            'tokenizer_id': tokenizer_id,
            'local_weights_dir': local_weights_path,
            'model_source': model_source,
            'tokenizer_source': tokenizer_source,
            'torch_dtype': str(config.get('torch_dtype', 'bfloat16')),
            'device_map': config.get('device_map'),
            'trust_remote_code': self._as_bool(config.get('trust_remote_code', True)),
            'attn_implementation': config.get('attn_implementation'),
        }

    def _normalize_data_config(self, config: dict[str, Any]) -> dict[str, Any]:
        prompt_template_key = config.get('prompt_template_path') or config.get('prompt_template')
        if not prompt_template_key:
            raise ValueError('data.prompt_template_path must be provided')
        train_path = resolve_path(config['train_path'])
        val_path = resolve_path(config['val_path'])
        return {
            'train_path': train_path,
            'val_path': val_path,
            'prompt_template_path': resolve_path(prompt_template_key),
            'train_max_samples': self._optional_int(config.get('train_max_samples')),
            'val_max_samples': self._optional_int(config.get('val_max_samples')),
        }

    def _normalize_training_config(self, config: dict[str, Any]) -> dict[str, Any]:
        output_dir_raw = config.get('output_dir', self.context.output_dir)
        report_to_raw = config.get('report_to', [])
        run_type = str(config.get('type', 'train')).strip().lower()
        if run_type not in {'train', 'eval'}:
            raise ValueError('training.type must be either "train" or "eval"')
        if report_to_raw is None:
            report_to = []
        elif isinstance(report_to_raw, str):
            report_to = [] if report_to_raw.lower() == 'none' else [report_to_raw]
        else:
            report_to = list(report_to_raw)

        return {
            'output_dir': resolve_path(output_dir_raw),
            'type': run_type,
            'num_train_epochs': float(config.get('num_train_epochs', config.get('epochs', 1))),
            'per_device_train_batch_size': int(
                config.get('per_device_train_batch_size', config.get('batch_size', 1))
            ),
            'gradient_accumulation_steps': int(
                config.get('gradient_accumulation_steps', config.get('grad_accum_steps', 1))
            ),
            'learning_rate': float(config.get('learning_rate', 2.0e-5)),
            'weight_decay': float(config.get('weight_decay', 0.0)),
            'warmup_ratio': float(config.get('warmup_ratio', 0.03)),
            'lr_scheduler_type': str(config.get('lr_scheduler_type', 'constant_with_warmup')),
            'max_seq_length': int(config.get('max_seq_length', 4096)),
            'gradient_checkpointing': self._as_bool(
                config.get('gradient_checkpointing', True)
            ),
            'bf16': self._as_bool(config.get('bf16', True)),
            'fp16': self._as_bool(config.get('fp16', False)),
            'logging_steps': max(1, int(config.get('logging_steps', 10))),
            'eval_steps': max(1, int(config.get('eval_steps', 50))),
            'save_steps': max(1, int(config.get('save_steps', 50))),
            'save_total_limit': int(config.get('save_total_limit', 2)),
            'max_steps': int(config.get('max_steps', -1)),
            'report_to': report_to,
            'resume_from_checkpoint': config.get('resume_from_checkpoint'),
        }

    def _normalize_lora_config(
        self,
        config: dict[str, Any],
        training_config: dict[str, Any],
    ) -> dict[str, Any]:
        enabled = config.get('enabled', training_config.get('use_lora', True))
        target_modules = config.get(
            'target_modules',
            [
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
        )
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        return {
            'enabled': self._as_bool(enabled),
            'r': int(config.get('r', 16)),
            'alpha': int(config.get('alpha', config.get('lora_alpha', 32))),
            'dropout': float(config.get('dropout', config.get('lora_dropout', 0.05))),
            'target_modules': list(target_modules),
            'bias': str(config.get('bias', 'none')),
            'task_type': str(config.get('task_type', 'CAUSAL_LM')),
        }

    def _normalize_eval_config(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            'top_k': int(config.get('top_k', 3)),
            'batch_size': max(1, int(config.get('batch_size', 4))),
            'max_new_tokens': int(config.get('max_new_tokens', 512)),
            'max_prompt_tokens': self._optional_int(config.get('max_prompt_tokens')),
            'max_samples': self._optional_int(config.get('max_samples')),
            'shuffle': self._as_bool(config.get('shuffle', True)),
            'save_generations': self._as_bool(config.get('save_generations', True)),
            'save_generation_count': max(0, int(config.get('save_generation_count', 3))),
        }

    def _load_runtime_dependencies(self) -> dict[str, Any]:
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig, PeftModel, TaskType
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from trl import SFTTrainer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Student SFT training requires torch, datasets, peft, trl, and accelerate. '
                'Install them with: uv sync --extra train'
            ) from exc

        return {
            'torch': torch,
            'Dataset': Dataset,
            'LoraConfig': LoraConfig,
            'PeftModel': PeftModel,
            'TaskType': TaskType,
            'AutoModelForCausalLM': AutoModelForCausalLM,
            'AutoTokenizer': AutoTokenizer,
            'TrainingArguments': TrainingArguments,
            'SFTTrainer': SFTTrainer,
        }

    def _load_model_and_tokenizer(self, *, torch, auto_model_cls, auto_tokenizer_cls):
        tokenizer = auto_tokenizer_cls.from_pretrained(
            self.model_cfg['tokenizer_source'],
            trust_remote_code=self.model_cfg['trust_remote_code'],
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        model_kwargs: dict[str, Any] = {
            'trust_remote_code': self.model_cfg['trust_remote_code'],
        }
        torch_dtype = self._torch_dtype_from_string(torch, self.model_cfg['torch_dtype'])
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype
        if self.model_cfg['device_map'] is not None:
            model_kwargs['device_map'] = self.model_cfg['device_map']
        if self.model_cfg['attn_implementation']:
            model_kwargs['attn_implementation'] = self.model_cfg['attn_implementation']

        model = auto_model_cls.from_pretrained(
            self.model_cfg['model_source'],
            **model_kwargs,
        )
        if self.train_cfg['gradient_checkpointing'] and hasattr(model, 'config'):
            model.config.use_cache = False
        if self.train_cfg['gradient_checkpointing'] and hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        return model, tokenizer

    def _load_eval_model(self, model, peft_model_cls):
        checkpoint = self.train_cfg['resume_from_checkpoint']
        if not checkpoint:
            print('[train_sft] no eval checkpoint provided, using base model weights')
            return model
        checkpoint_path = resolve_path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Eval checkpoint not found: {checkpoint_path}')
        print(f'[train_sft] loading eval checkpoint={checkpoint_path}')
        return peft_model_cls.from_pretrained(model, str(checkpoint_path))

    def _prepare_eval_model_device(self, torch, model):
        if not torch.cuda.is_available():
            return model
        device_map = getattr(model, 'hf_device_map', None)
        if device_map:
            return model
        target_device = torch.device('cuda:0')
        print(f'[train_sft] moving eval model to {target_device}')
        return model.to(target_device)

    def _build_train_dataset(self, dataset_cls, tokenizer):
        rows = load_jsonl(self.data_cfg['train_path'])
        if not rows:
            raise ValueError(f'No SFT rows found in {self.data_cfg["train_path"]}')

        max_samples = self.data_cfg['train_max_samples']
        if max_samples is not None:
            rows = rows[:max_samples]

        train_records: list[dict[str, str]] = []
        for row in rows:
            messages = row.get('messages')
            if not isinstance(messages, list) or not messages:
                raise ValueError('Each SFT train row must contain a non-empty messages list')
            train_records.append({'text': self._render_chat_example(messages, tokenizer)})

        return dataset_cls.from_list(train_records)

    def _build_training_args(self, training_arguments_cls, torch):
        use_bf16 = self.train_cfg['bf16'] and torch.cuda.is_available()
        use_fp16 = self.train_cfg['fp16'] and torch.cuda.is_available() and not use_bf16
        return training_arguments_cls(
            output_dir=str(self.output_dir),
            run_name=self.context.experiment_name,
            seed=self.seed,
            num_train_epochs=self.train_cfg['num_train_epochs'],
            per_device_train_batch_size=self.train_cfg['per_device_train_batch_size'],
            gradient_accumulation_steps=self.train_cfg['gradient_accumulation_steps'],
            learning_rate=self.train_cfg['learning_rate'],
            weight_decay=self.train_cfg['weight_decay'],
            warmup_ratio=self.train_cfg['warmup_ratio'],
            lr_scheduler_type=self.train_cfg['lr_scheduler_type'],
            logging_strategy='steps',
            logging_steps=self.train_cfg['logging_steps'],
            logging_first_step=True,
            save_strategy='steps',
            save_steps=self.train_cfg['save_steps'],
            save_total_limit=self.train_cfg['save_total_limit'],
            max_steps=self.train_cfg['max_steps'],
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=self.train_cfg['gradient_checkpointing'],
            remove_unused_columns=False,
            report_to=self.train_cfg['report_to'],
        )

    def _build_peft_config(self, lora_config_cls, task_type_cls):
        if not self.lora_cfg['enabled']:
            return None
        task_type = getattr(task_type_cls, self.lora_cfg['task_type'].upper(), None)
        if task_type is None:
            task_type = task_type_cls.CAUSAL_LM
        return lora_config_cls(
            r=self.lora_cfg['r'],
            lora_alpha=self.lora_cfg['alpha'],
            lora_dropout=self.lora_cfg['dropout'],
            target_modules=self.lora_cfg['target_modules'],
            bias=self.lora_cfg['bias'],
            task_type=task_type,
        )

    def _build_trainer(
        self,
        *,
        trainer_cls,
        model,
        tokenizer,
        train_dataset,
        training_args,
        peft_config,
    ):
        signature = inspect.signature(trainer_cls.__init__)
        trainer_kwargs: dict[str, Any] = {
            'model': model,
            'args': training_args,
            'train_dataset': train_dataset,
        }
        if 'dataset_text_field' in signature.parameters:
            trainer_kwargs['dataset_text_field'] = 'text'
        if 'max_seq_length' in signature.parameters:
            trainer_kwargs['max_seq_length'] = self.train_cfg['max_seq_length']
        if 'packing' in signature.parameters:
            trainer_kwargs['packing'] = False
        if peft_config is not None and 'peft_config' in signature.parameters:
            trainer_kwargs['peft_config'] = peft_config
        if 'processing_class' in signature.parameters:
            trainer_kwargs['processing_class'] = tokenizer
        elif 'tokenizer' in signature.parameters:
            trainer_kwargs['tokenizer'] = tokenizer
        return trainer_cls(**trainer_kwargs)

    def _prepare_validation_examples(self) -> list[ValidationExample]:
        val_path = self.data_cfg['val_path']
        if not val_path.exists():
            print(f'[train_sft] validation path not found, skipping: {val_path}')
            return []

        rows = load_jsonl(val_path)
        indexed_rows = list(enumerate(rows))
        if self.eval_cfg['shuffle']:
            rng = random.Random(self.seed)
            rng.shuffle(indexed_rows)

        max_samples = self.eval_cfg['max_samples']
        if max_samples is None:
            max_samples = self.data_cfg['val_max_samples']
        if max_samples is not None:
            indexed_rows = indexed_rows[:max_samples]

        examples: list[ValidationExample] = []
        for idx, row in indexed_rows:
            label = str(row.get('label', '')).strip().upper()
            prompt = self._build_eval_prompt(row)
            if not label or not prompt:
                continue
            examples.append(
                ValidationExample(
                    label=label,
                    prompt=prompt,
                    source_row_index=idx,
                )
            )
        return examples

    def _build_eval_prompt(self, row: dict[str, Any]) -> str | None:
        text = str(row.get('text') or row.get('text_clean') or '').strip()
        if text:
            return self.prompt_template.format(posts=text)

        messages = row.get('messages')
        if isinstance(messages, list):
            for message in messages:
                if str(message.get('role', '')).lower() == 'user':
                    content = str(message.get('content', '')).strip()
                    if content:
                        return content
        return None

    def run_ranking_evaluation(self, *, model, step: int, epoch: float | None) -> dict[str, float]:
        if not self.validation_examples or self.tokenizer is None:
            return {}

        print(
            f'[train_sft] step={step}: running ranking validation '
            f'on {len(self.validation_examples)} samples'
        )

        was_training = bool(getattr(model, 'training', False))
        if hasattr(model, 'eval'):
            model.eval()

        total = len(self.validation_examples)
        parsed = 0
        top1 = 0
        gold_in_top_k = 0
        top_k = self.eval_cfg['top_k']
        batch_size = self.eval_cfg['batch_size']
        saved_generations: list[dict[str, Any]] = []

        eval_bar = tqdm(
            total=total,
            desc=f'eval step={step}',
            unit='sample',
            dynamic_ncols=True,
            leave=False,
        )
        try:
            for start in range(0, total, batch_size):
                batch = self.validation_examples[start : start + batch_size]
                prompts = [example.prompt for example in batch]
                outputs = generate_teacher_replies_with_fallback(
                    model=model,
                    tokenizer=self.tokenizer,
                    user_prompts=prompts,
                    max_new_tokens=self.eval_cfg['max_new_tokens'],
                    max_prompt_tokens=self.eval_cfg['max_prompt_tokens'],
                )
                for example, output in zip(batch, outputs, strict=True):
                    ranked = parse_ranked_mbti(output)
                    is_top_1 = bool(ranked) and ranked[0] == example.label
                    in_top_k = example.label in ranked[:top_k]
                    if ranked:
                        parsed += 1
                    if is_top_1:
                        top1 += 1
                    if in_top_k:
                        gold_in_top_k += 1
                    if len(saved_generations) < self.eval_cfg['save_generation_count']:
                        saved_generations.append(
                            {
                                'label': example.label,
                                'source_row_index': example.source_row_index,
                                'prompt': example.prompt,
                                'assistant_raw': output,
                                'ranked_mbti': ranked,
                                'top_1_correct': is_top_1,
                                f'gold_in_top_{top_k}': in_top_k,
                            }
                        )
                    eval_bar.update(1)
                    if eval_bar.n:
                        eval_bar.set_postfix({
                            'parse': f'{parsed / eval_bar.n:.3f}',
                            'top1': f'{top1 / eval_bar.n:.3f}',
                            f'top{top_k}': f'{gold_in_top_k / eval_bar.n:.3f}',
                        })
        finally:
            eval_bar.close()
            if was_training and hasattr(model, 'train'):
                model.train()

        metrics = {
            'ranking_eval_examples': float(total),
            'ranking_eval_parse_rate': parsed / total if total else 0.0,
            'ranking_eval_top_1_accuracy': top1 / total if total else 0.0,
            f'ranking_eval_gold_in_top_{top_k}': gold_in_top_k / total if total else 0.0,
        }
        saved_generations_path = self._save_eval_generations(
            step=step,
            epoch=epoch,
            examples=saved_generations,
        )
        self._append_eval_history(
            {
                'step': step,
                'epoch': epoch,
                'examples': total,
                'parsed_examples': parsed,
                'top_1_correct': top1,
                f'gold_in_top_{top_k}_count': gold_in_top_k,
                'saved_generations_path': str(saved_generations_path) if saved_generations_path else None,
                'metrics': metrics,
            }
        )
        return metrics


    def _init_best_metric_state(self) -> dict[str, dict[str, Any]]:
        top_k = self.eval_cfg['top_k']
        metric_dirs = {
            'ranking_eval_top_1_accuracy': 'ranking_eval_top_1_accuracy',
            f'ranking_eval_gold_in_top_{top_k}': f'ranking_eval_gold_in_top_{top_k}',
        }
        return {
            metric_name: {
                'alias': alias,
                'best_value': None,
            }
            for metric_name, alias in metric_dirs.items()
        }

    def maybe_save_best_checkpoints(
        self,
        trainer,
        *,
        metrics: dict[str, float],
        step: int,
        epoch: float | None,
    ) -> None:
        if self.tokenizer is None:
            return
        for metric_name, state in self.best_metric_state.items():
            metric_value = self._coerce_float(metrics.get(metric_name))
            if metric_value is None:
                continue
            best_value = state['best_value']
            if best_value is not None and metric_value <= best_value:
                continue
            state['best_value'] = metric_value
            self._save_best_metric_checkpoint(
                trainer,
                metric_name=metric_name,
                metric_alias=state['alias'],
                metric_value=metric_value,
                step=step,
                epoch=epoch,
            )

    def _save_best_metric_checkpoint(
        self,
        trainer,
        *,
        metric_name: str,
        metric_alias: str,
        metric_value: float,
        step: int,
        epoch: float | None,
    ) -> None:
        target_dir = self.best_checkpoints_dir / metric_alias
        if target_dir.exists():
            shutil.rmtree(target_dir)
        ensure_dir(target_dir)
        trainer.save_model(str(target_dir))
        self.tokenizer.save_pretrained(str(target_dir))
        metadata = {
            'metric_name': metric_name,
            'metric_alias': metric_alias,
            'metric_value': metric_value,
            'step': step,
            'epoch': epoch,
        }
        (target_dir / 'best_checkpoint.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + '\\n',
            encoding='utf-8',
        )
        print(
            f'[train_sft] saved best checkpoint for {metric_name}='
            f'{metric_value:.6f} to {target_dir}'
        )

    def _collect_runtime_info(self, torch, model) -> dict[str, Any]:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        gpu_names = [torch.cuda.get_device_name(idx) for idx in range(device_count)] if cuda_available else []
        current_device = int(torch.cuda.current_device()) if cuda_available and device_count else None
        current_gpu_name = gpu_names[current_device] if current_device is not None and current_device < len(gpu_names) else None
        model_device_map = getattr(model, 'hf_device_map', None)
        return {
            'hostname': platform.node(),
            'pid': os.getpid(),
            'cuda_available': cuda_available,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_device_count': device_count,
            'cuda_current_device': current_device,
            'cuda_current_device_name': current_gpu_name,
            'gpu_names': gpu_names,
            'model_devices': self._collect_model_devices(model),
            'model_device_map': self._json_ready(model_device_map) if model_device_map else None,
        }

    def _collect_model_devices(self, model) -> list[str]:
        device_map = getattr(model, 'hf_device_map', None)
        if isinstance(device_map, dict) and device_map:
            seen: list[str] = []
            for device in device_map.values():
                label = str(device)
                if label not in seen:
                    seen.append(label)
            return seen

        devices: list[str] = []
        try:
            for param in model.parameters():
                label = str(param.device)
                if label not in devices:
                    devices.append(label)
                if len(devices) >= 8:
                    break
        except Exception:
            return []
        return devices

    def _format_runtime_summary(self) -> str:
        if not self.runtime_info:
            return 'unknown'
        model_devices = self.runtime_info.get('model_devices') or []
        device_label = 'cuda' if self.runtime_info.get('cuda_available') else 'cpu'
        return (
            f"device={device_label} "
            f"visible={self.runtime_info.get('cuda_visible_devices')!r} "
            f"count={self.runtime_info.get('cuda_device_count')} "
            f"model_devices={model_devices}"
        )

    def _append_eval_history(self, entry: dict[str, Any]) -> None:
        ensure_dir(self.eval_history_path.parent)
        with self.eval_history_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _save_eval_generations(
        self,
        *,
        step: int,
        epoch: float | None,
        examples: list[dict[str, Any]],
    ) -> Path | None:
        if not self.eval_cfg['save_generations'] or not examples:
            return None
        ensure_dir(self.eval_generations_dir)
        path = self.eval_generations_dir / f'step_{step:06d}.json'
        payload = {
            'step': step,
            'epoch': epoch,
            'saved_examples': len(examples),
            'examples': examples,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        print(f'[train_sft] saved {len(examples)} eval generations to {path}')
        return path

    def _write_runtime_artifacts(self) -> None:
        snapshot = {
            'experiment_name': self.context.experiment_name,
            'seed': self.seed,
            'model': self._json_ready(self.model_cfg),
            'data': self._json_ready(self.data_cfg),
            'training': self._json_ready(self.train_cfg),
            'lora': self._json_ready(self.lora_cfg),
            'evaluation': self._json_ready(self.eval_cfg),
            'runtime': self._json_ready(self.runtime_info),
            'validation_examples': len(self.validation_examples),
        }
        config_path = self.output_dir / 'resolved_sft_config.json'
        config_path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + '\\n',
            encoding='utf-8',
        )
        runtime_path = self.output_dir / 'runtime_info.json'
        runtime_path.write_text(
            json.dumps(self._json_ready(self.runtime_info), indent=2, ensure_ascii=False) + '\\n',
            encoding='utf-8',
        )


    def _write_training_graphs(self, log_history: list[dict[str, Any]]) -> None:
        loss_points = self._extract_log_series(log_history, 'loss')
        lr_points = self._extract_log_series(log_history, 'learning_rate')
        eval_series = self._extract_eval_series()
        eval_map = {name: points for name, points in eval_series}

        charts: list[str] = []
        png_series: list[tuple[str, list[tuple[float, float]], str]] = []
        if loss_points:
            charts.append(self._render_svg_chart('Training Loss', loss_points, '#2563eb'))
            png_series.append(('training_loss', loss_points, 'Training Loss'))
        if lr_points:
            charts.append(self._render_svg_chart('Learning Rate', lr_points, '#dc2626'))
            png_series.append(('learning_rate', lr_points, 'Learning Rate'))
        for name, points in eval_series:
            charts.append(self._render_svg_chart(name, points, '#059669'))
            png_series.append((self._chart_slug(name), points, name))

        if not charts:
            return

        ensure_dir(self.figures_dir)
        output_path = self.figures_dir / 'training_curves.html'
        output_path.write_text(self._render_training_graphs_page(charts), encoding='utf-8')
        print(f'[train_sft] saved training curves to {output_path}')
        self._write_training_graph_pngs(png_series)
        self._write_training_summary_png(
            lr_points=lr_points,
            loss_points=loss_points,
            parse_points=eval_map.get('ranking eval parse rate', []),
            top1_points=eval_map.get('ranking eval top 1 accuracy', []),
            topk_points=eval_map.get('ranking eval gold in top 3', []),
            top_k=self.eval_cfg['top_k'],
        )

    def _write_training_graph_pngs(
        self,
        chart_series: list[tuple[str, list[tuple[float, float]], str]],
    ) -> None:
        plt = self._load_matplotlib_pyplot()
        if plt is None:
            return

        ensure_dir(self.figures_dir)
        for slug, points, title in chart_series:
            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
            ax.plot(xs, ys, linewidth=2.5)
            self._style_matplotlib_axes(ax, title=title, ylabel=title)
            fig.tight_layout()
            output_path = self.figures_dir / f'{slug}.png'
            fig.savefig(output_path)
            plt.close(fig)


    def _write_training_summary_png(
        self,
        *,
        lr_points: list[tuple[float, float]],
        loss_points: list[tuple[float, float]],
        parse_points: list[tuple[float, float]],
        top1_points: list[tuple[float, float]],
        topk_points: list[tuple[float, float]],
        top_k: int,
    ) -> None:
        plt = self._load_matplotlib_pyplot()
        if plt is None:
            return
        if not any([lr_points, loss_points, parse_points, top1_points, topk_points]):
            return

        ensure_dir(self.figures_dir)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
        panels = [
            (axes[0, 0], lr_points, 'Learning Rate', '#dc2626'),
            (axes[0, 1], loss_points, 'Training Loss', '#2563eb'),
            (axes[1, 0], parse_points, 'Parse Rate', '#059669'),
        ]
        for ax, points, title, color in panels:
            if points:
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                ax.plot(xs, ys, linewidth=2.5, color=color)
            self._style_matplotlib_axes(ax, title=title)

        ax = axes[1, 1]
        if top1_points:
            ax.plot(
                [point[0] for point in top1_points],
                [point[1] for point in top1_points],
                linewidth=2.5,
                color='#7c3aed',
                label='Top-1 Accuracy',
            )
        if topk_points:
            ax.plot(
                [point[0] for point in topk_points],
                [point[1] for point in topk_points],
                linewidth=2.5,
                color='#ea580c',
                label=f'Gold in Top-{top_k}',
            )
        self._style_matplotlib_axes(ax, title='Ranking Accuracy')
        if top1_points or topk_points:
            ax.legend()

        fig.tight_layout()
        output_path = self.figures_dir / 'training_summary.png'
        fig.savefig(output_path)
        plt.close(fig)

    def _style_matplotlib_axes(self, ax, *, title: str, ylabel: str | None = None) -> None:
        ax.set_title(title)
        ax.set_xlabel('Step')
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(which='major', linestyle='--', linewidth=0.8, alpha=0.45)

    def _load_matplotlib_pyplot(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            return plt
        except ImportError:
            print('[train_sft] matplotlib not available, skipping PNG graph export')
            return None

    def _chart_slug(self, name: str) -> str:
        slug = name.strip().lower().replace(' ', '_')
        slug = ''.join(ch for ch in slug if ch.isalnum() or ch == '_')
        return slug or 'chart'

    def _extract_log_series(
        self,
        log_history: list[dict[str, Any]],
        key: str,
    ) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for entry in log_history:
            if key not in entry:
                continue
            step = self._coerce_float(entry.get('step'))
            value = self._coerce_float(entry.get(key))
            if step is None or value is None:
                continue
            points.append((step, value))
        return points

    def _extract_eval_series(self) -> list[tuple[str, list[tuple[float, float]]]]:
        if not self.eval_history_path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with self.eval_history_path.open(encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        by_metric: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            step = self._coerce_float(row.get('step'))
            metrics = row.get('metrics', {})
            if step is None or not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                numeric = self._coerce_float(value)
                if numeric is None:
                    continue
                by_metric.setdefault(str(key), []).append((step, numeric))

        return [(key.replace('_', ' '), by_metric[key]) for key in sorted(by_metric)]

    def _render_training_graphs_page(self, charts: list[str]) -> str:
        body = '\n'.join(charts)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SFT Training Curves</title>
  <style>
    :root {{
      color-scheme: dark;
      --muted: #94a3b8;
      --text: #e5e7eb;
    }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: linear-gradient(180deg, #020617, #0f172a);
      color: var(--text);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ margin: 0 0 24px; color: var(--muted); }}
    .chart {{
      background: rgba(17, 24, 39, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 18px;
      box-shadow: 0 20px 45px rgba(15, 23, 42, 0.35);
    }}
    .chart-title {{ margin: 0 0 10px; font-size: 18px; }}
    .chart-meta {{ margin: 8px 0 0; color: var(--muted); font-size: 13px; }}
    svg {{ width: 100%; height: auto; display: block; }}
    text {{ fill: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <h1>SFT Training Curves</h1>
  <p>Saved automatically from trainer logs and ranking evaluation history.</p>
  {body}
</body>
</html>
"""

    def _render_svg_chart(
        self,
        title: str,
        points: list[tuple[float, float]],
        color: str,
    ) -> str:
        width = 880
        height = 260
        left = 56
        right = 18
        top = 18
        bottom = 34
        plot_w = width - left - right
        plot_h = height - top - bottom
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        if x_min == x_max:
            x_max = x_min + 1.0
        if y_min == y_max:
            delta = abs(y_min) * 0.05 or 1.0
            y_min -= delta
            y_max += delta

        def sx(x: float) -> float:
            return left + (x - x_min) / (x_max - x_min) * plot_w

        def sy(y: float) -> float:
            return top + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

        polyline = ' '.join(f'{sx(x):.2f},{sy(y):.2f}' for x, y in points)
        y_ticks: list[str] = []
        for idx in range(5):
            frac = idx / 4
            value = y_max - frac * (y_max - y_min)
            y = top + frac * plot_h
            y_ticks.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
                f'stroke="rgba(148,163,184,0.18)" stroke-width="1" />'
                f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{value:.4g}</text>'
            )
        x_labels = (
            f'<text x="{left}" y="{height - 8}" text-anchor="start">step {x_min:.0f}</text>'
            f'<text x="{width - right}" y="{height - 8}" text-anchor="end">step {x_max:.0f}</text>'
        )
        safe_title = html.escape(title)
        return f"""<section class="chart">
  <h2 class="chart-title">{safe_title}</h2>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{safe_title}">
    <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="rgba(15,23,42,0.35)" />
    {''.join(y_ticks)}
    <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="rgba(148,163,184,0.3)" stroke-width="1" />
    <polyline fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{polyline}" />
    {x_labels}
  </svg>
  <p class="chart-meta">{len(points)} points</p>
</section>"""

    def _coerce_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _render_chat_example(self, messages: list[dict[str, Any]], tokenizer) -> str:
        normalized_messages = []
        for message in messages:
            normalized_messages.append(
                {
                    'role': str(message.get('role', 'user')).strip().lower(),
                    'content': str(message.get('content', '')).strip(),
                }
            )
        if hasattr(tokenizer, 'apply_chat_template'):
            return tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        rendered_parts = []
        for message in normalized_messages:
            rendered_parts.append(
                f"{message['role'].upper()}:\n{message['content']}"
            )
        return '\n\n'.join(rendered_parts)


    def _parameter_counts(self, model: Any) -> dict[str, float | int]:
        total_params = 0
        learnable_params = 0
        for param in model.parameters():
            count = param.numel()
            total_params += count
            if param.requires_grad:
                learnable_params += count
        trainable_ratio = (learnable_params / total_params) if total_params else 0.0
        return {
            'learnable_params': learnable_params,
            'total_params': total_params,
            'trainable_ratio': trainable_ratio,
        }

    def _load_prompt_template(self, path: Path) -> str:
        return path.read_text(encoding='utf-8')

    def _torch_dtype_from_string(self, torch, name: str):
        mapping = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        key = str(name or '').strip().lower()
        if not key or key == 'auto':
            return None
        if key not in mapping:
            raise ValueError(f'Unknown torch dtype: {name!r}')
        return mapping[key]

    def _optional_int(self, value: Any) -> int | None:
        if value in (None, '', 'null'):
            return None
        return int(value)

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'on'}
        return bool(value)

    def _json_ready(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: self._json_ready(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._json_ready(item) for item in value]
        return value
