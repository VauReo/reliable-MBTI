from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

from dataset.sft_generation import load_jsonl, resolve_path
from executors.base import RunContext
from executors.sft_executor import RankingEvalCallback, SFTExecutor
from losses.ranking_reward import (
    reward_dimension_similarity,
    reward_format,
    reward_invalid_answer,
    reward_parse_success,
    reward_reciprocal_rank,
    reward_top_1,
    reward_top_k,
)


class GRPOTrainingCallback(RankingEvalCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.executor.append_reward_history(
                {
                    'step': state.global_step,
                    'epoch': state.epoch,
                    'logs': self.executor._json_ready(logs),
                }
            )
        return control


class GRPOExecutor(SFTExecutor):
    def __init__(self, context: RunContext, config: dict[str, Any]) -> None:
        super().__init__(context, config)
        self.grpo_cfg = self._normalize_grpo_config(config.get('grpo', {}), config.get('training', {}))
        self.reward_cfg = self._normalize_reward_config(
            config.get('reward', {}),
            config.get('evaluation', {}),
        )
        self.reward_history_path = self.output_dir / 'reward_history.jsonl'

    def run(self) -> None:
        deps = self._load_runtime_dependencies()
        torch = deps['torch']
        Dataset = deps['Dataset']
        AutoModelForCausalLM = deps['AutoModelForCausalLM']
        AutoTokenizer = deps['AutoTokenizer']
        PeftModel = deps['PeftModel']
        LoraConfig = deps['LoraConfig']
        TaskType = deps['TaskType']
        GRPOConfig = deps['GRPOConfig']
        GRPOTrainer = deps['GRPOTrainer']

        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        eval_override = None
        if self.train_cfg['type'] == 'eval' and self.train_cfg['resume_from_checkpoint']:
            eval_override = self.train_cfg['resume_from_checkpoint']

        model, tokenizer, policy_meta = self._load_policy_model_and_tokenizer(
            torch=torch,
            auto_model_cls=AutoModelForCausalLM,
            auto_tokenizer_cls=AutoTokenizer,
            peft_model_cls=PeftModel,
            policy_source=eval_override,
            trainable=self.train_cfg['type'] == 'train',
        )
        self.tokenizer = tokenizer
        self.runtime_info = self._collect_runtime_info(torch, model)

        if self.train_cfg['type'] == 'eval':
            model = self._prepare_eval_model_device(torch, model)
            self.runtime_info = self._collect_runtime_info(torch, model)
            param_stats = self._parameter_counts(model)
            self._write_runtime_artifacts()
            print(f'[train_grpo] experiment={self.context.experiment_name}')
            print(f'[train_grpo] output_dir={self.output_dir}')
            print(f'[train_grpo] policy_source={policy_meta["effective_policy_source"]}')
            print(f'[train_grpo] runtime={self._format_runtime_summary()}')
            print(f'[train_grpo] val_rows={len(self.validation_examples)}')
            print(
                '[train_grpo] learnable_params='
                f'{param_stats["learnable_params"]:,} / total_params={param_stats["total_params"]:,} '
                f'({param_stats["trainable_ratio"]:.2%} trainable)'
            )
            metrics = self.run_ranking_evaluation(model=model, step=0, epoch=None)
            self._write_training_graphs([])
            print(f'[train_grpo] eval metrics={metrics}')
            print(f'[train_grpo] evaluation artifacts saved to {self.output_dir}')
            return

        train_dataset = self._build_grpo_dataset(Dataset)
        reward_funcs, reward_weights = self._build_reward_functions()
        grpo_args = self._build_grpo_args(GRPOConfig, torch, reward_weights)
        peft_config = None
        if self.lora_cfg['enabled'] and not policy_meta['policy_is_adapter']:
            peft_config = self._build_peft_config(LoraConfig, TaskType)

        callback = GRPOTrainingCallback(self)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=grpo_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=[callback],
            peft_config=peft_config,
        )
        callback.bind_trainer(trainer)

        self.runtime_info = self._collect_runtime_info(torch, trainer.model)
        param_stats = self._parameter_counts(trainer.model)
        self._write_runtime_artifacts()
        print(f'[train_grpo] experiment={self.context.experiment_name}')
        print(f'[train_grpo] output_dir={self.output_dir}')
        print(f'[train_grpo] policy_source={policy_meta["effective_policy_source"]}')
        print(f'[train_grpo] runtime={self._format_runtime_summary()}')
        print(f'[train_grpo] train_rows={len(train_dataset)}')
        print(f'[train_grpo] val_rows={len(self.validation_examples)}')
        print(
            '[train_grpo] learnable_params='
            f'{param_stats["learnable_params"]:,} / total_params={param_stats["total_params"]:,} '
            f'({param_stats["trainable_ratio"]:.2%} trainable)'
        )
        print(
            '[train_grpo] logging_steps={logging_steps} eval_steps={eval_steps} '
            'save_steps={save_steps}'.format(**self.train_cfg)
        )
        print(
            '[train_grpo] num_generations={num_generations} max_completion_length={max_completion_length} '
            'beta={beta}'.format(**self.grpo_cfg)
        )

        trainer.train(resume_from_checkpoint=self.train_cfg['resume_from_checkpoint'])
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        self._write_training_graphs(trainer.state.log_history)

        print('[train_grpo] training complete')
        print(f'[train_grpo] final artifacts saved to {self.output_dir}')

    def _existing_local_path(self, value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path if path.exists() else None
        resolved = resolve_path(value)
        return resolved if resolved.exists() else None

    def _normalize_model_config(self, config: dict[str, Any]) -> dict[str, Any]:
        policy_init_raw = str(
            config.get('policy_init') or config.get('policy_model_dir') or ''
        ).strip()
        reference_model_id = str(config.get('reference_model_id', '')).strip()
        tokenizer_id = str(config.get('tokenizer_id', '')).strip()
        local_weights_dir = config.get('local_weights_dir')
        local_weights_path = self._existing_local_path(str(local_weights_dir)) if local_weights_dir else None

        policy_source = policy_init_raw
        policy_path = self._existing_local_path(policy_init_raw)
        if policy_path and policy_path.exists():
            policy_source = str(policy_path)

        policy_is_adapter = bool(policy_path and (policy_path / 'adapter_config.json').exists())
        adapter_base_model = None
        if policy_is_adapter:
            adapter_config = json.loads((policy_path / 'adapter_config.json').read_text(encoding='utf-8'))
            adapter_base_model = adapter_config.get('base_model_name_or_path')

        base_model_source = ''
        if local_weights_path and local_weights_path.exists():
            base_model_source = str(local_weights_path)
        elif reference_model_id:
            base_model_source = reference_model_id
        elif adapter_base_model:
            base_model_source = str(adapter_base_model)
        elif policy_source:
            base_model_source = policy_source

        tokenizer_source = tokenizer_id or base_model_source or policy_source
        if not policy_source:
            policy_source = base_model_source
        if not policy_source:
            raise ValueError('model.policy_init or model.reference_model_id must be provided')
        if not base_model_source:
            raise ValueError('Could not infer the GRPO base model source')
        if not tokenizer_source:
            raise ValueError('model.tokenizer_id must be provided or inferable from the model source')

        return {
            'policy_init': policy_init_raw or policy_source,
            'policy_source': policy_source,
            'policy_is_adapter': policy_is_adapter,
            'reference_model_id': reference_model_id,
            'local_weights_dir': local_weights_path,
            'base_model_source': base_model_source,
            'tokenizer_source': tokenizer_source,
            'torch_dtype': str(config.get('torch_dtype', 'bfloat16')),
            'device_map': config.get('device_map'),
            'trust_remote_code': self._as_bool(config.get('trust_remote_code', True)),
            'attn_implementation': config.get('attn_implementation'),
        }

    def _normalize_grpo_config(
        self,
        config: dict[str, Any],
        training_config: dict[str, Any],
    ) -> dict[str, Any]:
        generation_batch_size = config.get('generation_batch_size')
        return {
            'num_generations': int(config.get('num_generations', config.get('group_size', 4))),
            'max_completion_length': int(
                config.get('max_completion_length', config.get('max_new_tokens', 512))
            ),
            'temperature': float(config.get('temperature', 0.7)),
            'top_p': float(config.get('top_p', 0.9)),
            'top_k': int(config.get('top_k', 20)),
            'repetition_penalty': float(config.get('repetition_penalty', 1.0)),
            'beta': float(config.get('beta', 0.02)),
            'num_iterations': int(config.get('num_iterations', 1)),
            'generation_batch_size': (
                None if generation_batch_size in (None, '', 'null') else int(generation_batch_size)
            ),
            'log_completions': self._as_bool(config.get('log_completions', False)),
            'num_completions_to_print': max(1, int(config.get('num_completions_to_print', 2))),
        }

    def _normalize_reward_config(
        self,
        config: dict[str, Any],
        eval_config: dict[str, Any],
    ) -> dict[str, Any]:
        top_k = int(config.get('top_k', eval_config.get('top_k', 3)))
        return {
            'top_k': top_k,
            'format_weight': float(config.get('format_weight', 0.1)),
            'parse_weight': float(config.get('parse_weight', 0.3)),
            'top_1_weight': float(config.get('top_1_weight', config.get('exact_match_weight', 2.0))),
            'top_k_weight': float(config.get('top_k_weight', config.get('ndcg_k_weight', 1.0))),
            'reciprocal_rank_weight': float(
                config.get('reciprocal_rank_weight', config.get('ndcg_weight', 0.5))
            ),
            'dimension_similarity_weight': float(config.get('dimension_similarity_weight', 0.0)),
            'invalid_answer_penalty': float(config.get('invalid_answer_penalty', -0.5)),
            'use_dimension_similarity': self._as_bool(
                config.get('use_dimension_similarity', True)
            ),
        }

    def _load_runtime_dependencies(self) -> dict[str, Any]:
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig, PeftModel, TaskType
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'GRPO training requires torch, datasets, peft, trl, and accelerate. '
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
            'GRPOConfig': GRPOConfig,
            'GRPOTrainer': GRPOTrainer,
        }

    def _load_policy_model_and_tokenizer(
        self,
        *,
        torch,
        auto_model_cls,
        auto_tokenizer_cls,
        peft_model_cls,
        policy_source: str | None,
        trainable: bool,
    ):
        effective_policy_source = str(policy_source or self.model_cfg['policy_source'])
        policy_path = self._existing_local_path(effective_policy_source)
        policy_exists = bool(policy_path and policy_path.exists())
        policy_is_adapter = bool(policy_exists and (policy_path / 'adapter_config.json').exists())

        tokenizer = auto_tokenizer_cls.from_pretrained(
            self.model_cfg['tokenizer_source'],
            trust_remote_code=self.model_cfg['trust_remote_code'],
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

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

        model_load_source = self.model_cfg['base_model_source'] if policy_is_adapter else (
            str(policy_path) if policy_exists else effective_policy_source
        )
        base_model = auto_model_cls.from_pretrained(
            model_load_source,
            **model_kwargs,
        )
        if self.train_cfg['gradient_checkpointing'] and hasattr(base_model, 'config'):
            base_model.config.use_cache = False
        if self.train_cfg['gradient_checkpointing'] and hasattr(base_model, 'enable_input_require_grads'):
            base_model.enable_input_require_grads()

        model = base_model
        if policy_is_adapter:
            model = peft_model_cls.from_pretrained(
                base_model,
                effective_policy_source,
                is_trainable=trainable,
            )
            if self.train_cfg['gradient_checkpointing'] and hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()

        return model, tokenizer, {
            'effective_policy_source': effective_policy_source,
            'policy_is_adapter': policy_is_adapter,
        }

    def _build_grpo_dataset(self, dataset_cls):
        rows = load_jsonl(self.data_cfg['train_path'])
        if not rows:
            raise ValueError(f'No GRPO rows found in {self.data_cfg["train_path"]}')

        max_samples = self.data_cfg['train_max_samples']
        if max_samples is not None:
            rows = rows[:max_samples]

        train_records: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            label = str(row.get('label', '')).strip().upper()
            prompt = self._build_eval_prompt(row)
            if not label or not prompt:
                continue
            train_records.append(
                {
                    'prompt': prompt,
                    'label': label,
                    'source_row_index': idx,
                    'source_dataset': str(row.get('source_dataset', 'unknown')),
                }
            )
        if not train_records:
            raise ValueError(f'No valid GRPO train rows found in {self.data_cfg["train_path"]}')
        return dataset_cls.from_list(train_records)

    def _build_grpo_args(self, grpo_config_cls, torch, reward_weights: list[float]):
        use_bf16 = self.train_cfg['bf16'] and torch.cuda.is_available()
        use_fp16 = self.train_cfg['fp16'] and torch.cuda.is_available() and not use_bf16
        kwargs: dict[str, Any] = {
            'output_dir': str(self.output_dir),
            'run_name': self.context.experiment_name,
            'seed': self.seed,
            'num_train_epochs': self.train_cfg['num_train_epochs'],
            'max_steps': self.train_cfg['max_steps'],
            'per_device_train_batch_size': self.train_cfg['per_device_train_batch_size'],
            'gradient_accumulation_steps': self.train_cfg['gradient_accumulation_steps'],
            'learning_rate': self.train_cfg['learning_rate'],
            'weight_decay': self.train_cfg['weight_decay'],
            'warmup_ratio': self.train_cfg['warmup_ratio'],
            'lr_scheduler_type': self.train_cfg['lr_scheduler_type'],
            'logging_steps': self.train_cfg['logging_steps'],
            'logging_strategy': 'steps',
            'logging_first_step': True,
            'save_strategy': 'steps',
            'save_steps': self.train_cfg['save_steps'],
            'save_total_limit': self.train_cfg['save_total_limit'],
            'eval_strategy': 'no',
            'report_to': self.train_cfg['report_to'],
            'bf16': use_bf16,
            'fp16': use_fp16,
            'gradient_checkpointing': self.train_cfg['gradient_checkpointing'],
            'remove_unused_columns': False,
            'num_generations': self.grpo_cfg['num_generations'],
            'max_completion_length': self.grpo_cfg['max_completion_length'],
            'temperature': self.grpo_cfg['temperature'],
            'top_p': self.grpo_cfg['top_p'],
            'top_k': self.grpo_cfg['top_k'],
            'repetition_penalty': self.grpo_cfg['repetition_penalty'],
            'beta': self.grpo_cfg['beta'],
            'num_iterations': self.grpo_cfg['num_iterations'],
            'reward_weights': reward_weights,
            'log_completions': self.grpo_cfg['log_completions'],
            'num_completions_to_print': self.grpo_cfg['num_completions_to_print'],
        }
        if self.grpo_cfg['generation_batch_size'] is not None:
            kwargs['generation_batch_size'] = self.grpo_cfg['generation_batch_size']
        return grpo_config_cls(**kwargs)

    def _build_reward_functions(self) -> tuple[list[Callable[..., list[float]]], list[float]]:
        reward_funcs: list[Callable[..., list[float]]] = []
        reward_weights: list[float] = []

        def register(name: str, weight: float, fn: Callable[..., list[float]]) -> None:
            if weight == 0.0:
                return
            fn.__name__ = name
            reward_funcs.append(fn)
            reward_weights.append(weight)

        def format_reward(*, completions, **kwargs):
            return [reward_format(completion) for completion in completions]

        def parse_reward(*, completions, **kwargs):
            return [reward_parse_success(completion) for completion in completions]

        def invalid_answer_penalty(*, completions, **kwargs):
            return [reward_invalid_answer(completion) for completion in completions]

        def top_1_reward(*, completions, label, **kwargs):
            return [reward_top_1(self._parse_completion(completion), gold) for completion, gold in zip(completions, label, strict=True)]

        def top_k_reward(*, completions, label, **kwargs):
            top_k = self.reward_cfg['top_k']
            return [
                reward_top_k(self._parse_completion(completion), gold, top_k=top_k)
                for completion, gold in zip(completions, label, strict=True)
            ]

        def reciprocal_rank_reward(*, completions, label, **kwargs):
            return [
                reward_reciprocal_rank(self._parse_completion(completion), gold, k=16)
                for completion, gold in zip(completions, label, strict=True)
            ]

        def dimension_similarity_reward(*, completions, label, **kwargs):
            scores: list[float] = []
            for completion, gold in zip(completions, label, strict=True):
                ranked = self._parse_completion(completion)
                best = ranked[0] if ranked else ''
                scores.append(reward_dimension_similarity(best, gold) if best else 0.0)
            return scores

        register('format_reward', self.reward_cfg['format_weight'], format_reward)
        register('parse_reward', self.reward_cfg['parse_weight'], parse_reward)
        register('invalid_answer_penalty', self.reward_cfg['invalid_answer_penalty'], invalid_answer_penalty)
        register('top_1_reward', self.reward_cfg['top_1_weight'], top_1_reward)
        register('top_k_reward', self.reward_cfg['top_k_weight'], top_k_reward)
        register(
            'reciprocal_rank_reward',
            self.reward_cfg['reciprocal_rank_weight'],
            reciprocal_rank_reward,
        )
        if self.reward_cfg['use_dimension_similarity']:
            register(
                'dimension_similarity_reward',
                self.reward_cfg['dimension_similarity_weight'],
                dimension_similarity_reward,
            )
        if not reward_funcs:
            raise ValueError('At least one non-zero GRPO reward weight must be configured')
        return reward_funcs, reward_weights

    def _parse_completion(self, completion: Any) -> list[str]:
        if isinstance(completion, list):
            text = ''.join(str(part.get('content', '')) for part in completion)
        else:
            text = str(completion)
        from dataset.sft_generation import parse_ranked_mbti
        return parse_ranked_mbti(text)

    def append_reward_history(self, entry: dict[str, Any]) -> None:
        ensure_dir = self.reward_history_path.parent.mkdir
        ensure_dir(parents=True, exist_ok=True)
        with self.reward_history_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\\n')

    def _write_runtime_artifacts(self) -> None:
        snapshot = {
            'experiment_name': self.context.experiment_name,
            'seed': self.seed,
            'model': self._json_ready(self.model_cfg),
            'data': self._json_ready(self.data_cfg),
            'training': self._json_ready(self.train_cfg),
            'lora': self._json_ready(self.lora_cfg),
            'grpo': self._json_ready(self.grpo_cfg),
            'reward': self._json_ready(self.reward_cfg),
            'evaluation': self._json_ready(self.eval_cfg),
            'runtime': self._json_ready(self.runtime_info),
            'validation_examples': len(self.validation_examples),
        }
        config_path = self.output_dir / 'resolved_grpo_config.json'
        config_path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        runtime_path = self.output_dir / 'runtime_info.json'
        runtime_path.write_text(
            json.dumps(self._json_ready(self.runtime_info), indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )

    def _write_training_graphs(self, log_history: list[dict[str, Any]]) -> None:
        series_map = self._extract_numeric_log_series(log_history)
        eval_series = self._extract_eval_series()
        for name, points in eval_series:
            series_map[name] = points
        if not series_map:
            return

        charts: list[str] = []
        png_series: list[tuple[str, list[tuple[float, float]], str]] = []
        for name, points in sorted(series_map.items()):
            if not points:
                continue
            title = self._metric_title(name)
            color = self._metric_color(name)
            charts.append(self._render_svg_chart(title, points, color))
            png_series.append((self._chart_slug(title), points, title))

        ensure_dir = self.figures_dir.mkdir
        ensure_dir(parents=True, exist_ok=True)
        output_path = self.figures_dir / 'training_curves.html'
        output_path.write_text(self._render_training_graphs_page(charts), encoding='utf-8')
        print(f'[train_grpo] saved training curves to {output_path}')
        self._write_training_graph_pngs(png_series)
        self._write_grpo_summary_png(series_map)

    def _extract_numeric_log_series(
        self,
        log_history: list[dict[str, Any]],
    ) -> dict[str, list[tuple[float, float]]]:
        excluded = {'epoch', 'step', 'total_flos'}
        series: dict[str, list[tuple[float, float]]] = {}
        for entry in log_history:
            step = self._coerce_float(entry.get('step'))
            if step is None:
                continue
            for key, value in entry.items():
                if key in excluded:
                    continue
                numeric = self._coerce_float(value)
                if numeric is None:
                    continue
                series.setdefault(key, []).append((step, numeric))
        return series

    def _metric_title(self, name: str) -> str:
        cleaned = name.replace('/', ' ').replace('_', ' ')
        cleaned = ' '.join(cleaned.split())
        return cleaned or 'metric'

    def _metric_color(self, name: str) -> str:
        palette = ['#2563eb', '#dc2626', '#059669', '#7c3aed', '#ea580c', '#0891b2']
        return palette[sum(ord(ch) for ch in name) % len(palette)]

    def _write_grpo_summary_png(self, series_map: dict[str, list[tuple[float, float]]]) -> None:
        plt = self._load_matplotlib_pyplot()
        if plt is None:
            return
        if not series_map:
            return

        learning_rate = series_map.get('learning_rate', [])
        loss_points = series_map.get('loss', [])
        reward_points = self._first_series_matching(series_map, include='reward')
        kl_points = self._first_series_matching(series_map, include='kl')
        top1_points = series_map.get('ranking eval top 1 accuracy', [])
        topk_points = series_map.get(f'ranking eval gold in top {self.eval_cfg["top_k"]}', [])

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
        panels = [
            (axes[0, 0], learning_rate, 'Learning Rate', '#dc2626'),
            (axes[0, 1], loss_points, 'GRPO Loss', '#2563eb'),
            (axes[1, 0], reward_points or kl_points, 'Reward / KL', '#059669'),
        ]
        for ax, points, title, color in panels:
            if points:
                ax.plot([point[0] for point in points], [point[1] for point in points], linewidth=2.5, color=color)
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
                label=f'Gold in Top-{self.eval_cfg["top_k"]}',
            )
        self._style_matplotlib_axes(ax, title='Ranking Accuracy')
        if top1_points or topk_points:
            ax.legend()

        fig.tight_layout()
        output_path = self.figures_dir / 'training_summary.png'
        fig.savefig(output_path)
        plt.close(fig)

    def _first_series_matching(
        self,
        series_map: dict[str, list[tuple[float, float]]],
        *,
        include: str,
    ) -> list[tuple[float, float]]:
        include = include.lower()
        for key, points in series_map.items():
            key_l = key.lower()
            if include in key_l and not key_l.startswith('ranking eval'):
                return points
        return []
