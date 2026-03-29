"""Teacher distillation for SFT JSONL (chat messages + optional GRPO pool)."""

from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from utils.config import ROOT_DIR
from utils.io import ensure_dir


def resolve_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT_DIR / p


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _parse_mbti_bracket_block(block: str) -> list[str]:
    """Parse [TYPE1, TYPE2, ...] when LLMs omit quotes (invalid Python literals)."""
    inner = block.strip()
    if inner.startswith('[') and inner.endswith(']'):
        inner = inner[1:-1]
    out: list[str] = []
    for part in re.split(r'\s*,\s*', inner):
        part = part.strip()
        if not part:
            continue
        s = re.sub(r'[^A-Za-z]', '', part).upper()
        if len(s) == 4 and s.isalpha():
            out.append(s)
    return out


def parse_ranked_mbti(text: str) -> list[str]:
    """Extract ordered MBTI types from a strict <answer>...</answer> block."""
    m = re.search(r'<answer>\s*([\s\S]*?)\s*</answer>', text, re.IGNORECASE)
    if not m:
        return []
    block = m.group(1).strip()
    if not block:
        return []

    lst: Any | None = None
    try:
        lst = ast.literal_eval(block)
    except (SyntaxError, ValueError, TypeError):
        lst = None

    if isinstance(lst, list):
        out: list[str] = []
        for x in lst:
            s = str(x).strip().upper()
            if len(s) == 4 and s.isalpha():
                out.append(s)
        if out:
            return out

    if block.startswith('[') and block.endswith(']'):
        return _parse_mbti_bracket_block(block)

    return _parse_mbti_bracket_block(f'[{block}]')


def label_passes_rejection(
    label: str,
    ranked: list[str],
    *,
    top_k: int,
) -> bool:
    if not ranked:
        return False
    label_u = label.strip().upper()
    if label_u not in ranked:
        return False
    if top_k <= 0:
        return True
    idx = ranked.index(label_u)
    return idx < top_k


@dataclass
class GenerationStats:
    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    grpo_reserve_rows: int = 0
    grpo_from_sft_rejects: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def grpo_total_rows(self) -> int:
        return self.grpo_reserve_rows + self.grpo_from_sft_rejects


def _teacher_accept_postfix(accepted: int, attempted: int) -> str:
    if attempted <= 0:
        return 'accepted=0/0'
    pct = 100.0 * accepted / attempted
    return f'accepted={accepted}/{attempted} ({pct:.1f}%)'


def _dtype_from_string(name: str):
    import torch

    mapping = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    key = (name or 'bfloat16').lower()
    if key not in mapping:
        raise ValueError(f'Unknown torch_dtype {name!r}')
    return mapping[key]


def build_teacher_inputs(
    tokenizer,
    user_prompt: str,
    max_prompt_tokens: int | None = None,
) -> dict:
    messages = [{'role': 'user', 'content': user_prompt}]
    if not hasattr(tokenizer, 'apply_chat_template'):
        raise RuntimeError('Tokenizer must support apply_chat_template (Qwen chat models).')
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tok_kw: dict[str, Any] = {'return_tensors': 'pt'}
    if max_prompt_tokens is not None:
        tok_kw['truncation'] = True
        tok_kw['max_length'] = max_prompt_tokens
    return tokenizer(prompt, **tok_kw)


def _model_input_device(model):
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def generate_teacher_reply(
    *,
    model,
    tokenizer,
    user_prompt: str,
    max_new_tokens: int,
    max_prompt_tokens: int | None = None,
) -> str:
    import torch

    inputs = build_teacher_inputs(tokenizer, user_prompt, max_prompt_tokens=max_prompt_tokens)
    dev = _model_input_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs['input_ids'].shape[1]
    gen_ids = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def generate_teacher_replies_batch(
    *,
    model,
    tokenizer,
    user_prompts: list[str],
    max_new_tokens: int,
    max_prompt_tokens: int | None = None,
) -> list[str]:
    """Left-padded batch generation; returns one assistant string per prompt."""
    import torch

    if not user_prompts:
        return []
    if len(user_prompts) == 1:
        return [
            generate_teacher_reply(
                model=model,
                tokenizer=tokenizer,
                user_prompt=user_prompts[0],
                max_new_tokens=max_new_tokens,
                max_prompt_tokens=max_prompt_tokens,
            )
        ]

    pad_side = getattr(tokenizer, 'padding_side', 'right')
    tokenizer.padding_side = 'left'

    texts: list[str] = []
    for up in user_prompts:
        messages = [{'role': 'user', 'content': up}]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    tok_kw: dict[str, Any] = {
        'padding': True,
        'return_tensors': 'pt',
    }
    if max_prompt_tokens is not None:
        tok_kw['truncation'] = True
        tok_kw['max_length'] = max_prompt_tokens

    encoded = tokenizer(texts, **tok_kw)
    dev = _model_input_device(model)
    encoded = {k: v.to(dev) for k, v in encoded.items()}

    with torch.inference_mode():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_w = encoded['input_ids'].shape[1]
    results: list[str] = []
    for i in range(out.shape[0]):
        gen_ids = out[i, prompt_w:]
        results.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    tokenizer.padding_side = pad_side
    return results


def generate_teacher_replies_with_fallback(
    *,
    model,
    tokenizer,
    user_prompts: list[str],
    max_new_tokens: int,
    max_prompt_tokens: int | None = None,
) -> list[str]:
    """Try batched generation; on CUDA OOM, fall back to one prompt at a time."""
    import torch

    if not user_prompts:
        return []
    try:
        return generate_teacher_replies_batch(
            model=model,
            tokenizer=tokenizer,
            user_prompts=user_prompts,
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if 'out of memory' not in msg and 'cuda out of memory' not in msg:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        out: list[str] = []
        for up in user_prompts:
            out.append(
                generate_teacher_reply(
                    model=model,
                    tokenizer=tokenizer,
                    user_prompt=up,
                    max_new_tokens=max_new_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                )
            )
        return out


def run_generation(config: dict) -> GenerationStats:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'Generating SFT data requires torch and transformers. '
            'Install with: uv sync --extra train'
        ) from exc

    seed = int(config.get('seed', 42))
    random.seed(seed)

    teacher_cfg = config['teacher']
    data_cfg = config['data']
    gen_cfg = config['generation']
    out_cfg = config['outputs']

    model_dir = resolve_path(teacher_cfg['local_model_dir'])
    train_path = resolve_path(data_cfg['train_split_path'])
    template_path = resolve_path(data_cfg['prompt_template_path'])

    max_sft_rows = int(gen_cfg['max_rows_for_sft_pool'])
    shuffle = bool(gen_cfg.get('shuffle_before_split', True))
    top_k = int(gen_cfg.get('rejection_top_k', 5))
    require_in_rank = bool(gen_cfg.get('require_label_in_ranked_list', True))
    batch_size = max(1, int(gen_cfg.get('batch_size', 1)))
    max_prompt_tokens = gen_cfg.get('max_prompt_tokens')
    if max_prompt_tokens is not None:
        max_prompt_tokens = int(max_prompt_tokens)

    sft_out = resolve_path(out_cfg['sft_jsonl'])
    grpo_out = resolve_path(out_cfg['grpo_jsonl'])
    meta_out = resolve_path(out_cfg.get('metadata_json', 'outputs/generate_sft/metadata.json'))
    attempts_raw = out_cfg.get('teacher_attempts_jsonl', 'outputs/generate_sft/teacher_attempts.jsonl')
    if attempts_raw is None or attempts_raw == '':
        attempts_path = None
    else:
        attempts_path = resolve_path(str(attempts_raw))

    dtype = _dtype_from_string(str(teacher_cfg.get('torch_dtype', 'bfloat16')))
    max_new_tokens = int(teacher_cfg.get('max_new_tokens', 1024))
    device_map = teacher_cfg.get('device_map', 'auto')
    trust_remote_code = bool(teacher_cfg.get('trust_remote_code', True))

    template = load_prompt_template(template_path)
    rows = load_jsonl(train_path)
    if not rows:
        raise ValueError(f'No rows in {train_path}')

    indices = list(range(len(rows)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    if max_sft_rows <= 0:
        raise ValueError('generation.max_rows_for_sft_pool must be positive')
    sft_idx = indices[:max_sft_rows]
    grpo_idx = indices[max_sft_rows:]

    grpo_rows = [rows[i] for i in grpo_idx]
    grpo_from_sft_rejects: list[dict[str, Any]] = []

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()


    stats = GenerationStats(grpo_reserve_rows=len(grpo_rows))
    sft_lines: list[dict[str, Any]] = []
    teacher_attempt_logs: list[dict[str, Any]] = []

    print(
        f'[teacher] batch_size={batch_size}'
        + (f', max_prompt_tokens={max_prompt_tokens}' if max_prompt_tokens is not None else '')
    )

    def append_attempt_log(
        pos: int,
        *,
        label_log: str,
        attempted_call: bool,
        accepted_here: bool,
        reject_reason: str | None,
        ranked_out: list[str],
        assistant_raw: str | None,
        gen_err: str | None,
    ) -> None:
        if attempts_path is None:
            return
        gold_in_top = False
        if label_log and ranked_out:
            gold_in_top = label_passes_rejection(label_log, ranked_out, top_k=top_k)
        teacher_attempt_logs.append(
            {
                'source_row_index': pos,
                'label': label_log,
                'attempted_teacher': attempted_call,
                'accepted': accepted_here,
                'reject_reason': reject_reason if not accepted_here else None,
                'ranked_parsed': ranked_out,
                'rejection_top_k': top_k,
                'gold_in_top_k': gold_in_top,
                'assistant_raw': assistant_raw,
                'generate_error': gen_err,
            }
        )

    pbar = tqdm(
        total=len(sft_idx),
        desc='reasoning generation',
        unit='sample',
        dynamic_ncols=True,
    )
    batch_start = 0
    while batch_start < len(sft_idx):
        chunk = sft_idx[batch_start : batch_start + batch_size]
        batch_start += len(chunk)

        pack: list[tuple[int, dict[str, Any], str, str]] = []
        for pos in chunk:
            rec = rows[pos]
            label = str(rec.get('label', '')).strip().upper()
            text = str(rec.get('text', ''))
            if not label or not text:
                stats.errors.append(f'skip row {pos}: missing label or text')
                append_attempt_log(
                    pos,
                    label_log=label,
                    attempted_call=False,
                    accepted_here=False,
                    reject_reason='missing_fields',
                    ranked_out=[],
                    assistant_raw=None,
                    gen_err=None,
                )
                pbar.set_postfix_str(_teacher_accept_postfix(stats.accepted, stats.attempted))
                pbar.update(1)
                continue

            user_prompt = template.format(posts=text)
            stats.attempted += 1
            pack.append((pos, rec, label, user_prompt))

        if not pack:
            continue

        prompts = [p[3] for p in pack]
        results: list[tuple[str | None, str | None]] = []
        try:
            texts = generate_teacher_replies_with_fallback(
                model=model,
                tokenizer=tokenizer,
                user_prompts=prompts,
                max_new_tokens=max_new_tokens,
                max_prompt_tokens=max_prompt_tokens,
            )
            results = [(t, None) for t in texts]
        except Exception:
            for up in prompts:
                try:
                    t = generate_teacher_reply(
                        model=model,
                        tokenizer=tokenizer,
                        user_prompt=up,
                        max_new_tokens=max_new_tokens,
                        max_prompt_tokens=max_prompt_tokens,
                    )
                    results.append((t, None))
                except Exception as exc:  # pragma: no cover
                    results.append((None, repr(exc)))

        if len(results) != len(pack):
            raise RuntimeError(
                f'Teacher batch length mismatch: pack={len(pack)} results={len(results)}'
            )

        for (pos, rec, label, user_prompt), (assistant, gen_err) in zip(pack, results, strict=True):
            assistant_raw: str | None = None
            ranked_out: list[str] = []
            reject_reason: str | None = None
            accepted_here = False
            attempted_call = True
            try:
                if assistant is None:
                    stats.rejected += 1
                    stats.errors.append(f'row {pos}: generate failed: {gen_err}')
                    grpo_from_sft_rejects.append(dict(rec))
                    reject_reason = 'generate_failed'
                    continue

                full_assistant = assistant
                assistant_raw = assistant
                ranked_out = parse_ranked_mbti(assistant)
                if not ranked_out:
                    stats.rejected += 1
                    grpo_from_sft_rejects.append(dict(rec))
                    reject_reason = 'parse_failed'
                    continue
                ok = True
                if require_in_rank:
                    ok = label_passes_rejection(label, ranked_out, top_k=top_k)
                if not ok:
                    stats.rejected += 1
                    grpo_from_sft_rejects.append(dict(rec))
                    reject_reason = 'rejection_sampling'
                    continue

                messages = [
                    {'role': 'user', 'content': user_prompt},
                    {'role': 'assistant', 'content': full_assistant.strip()},
                ]
                out_rec = {
                    'messages': messages,
                    'label': label,
                    'ranked_mbti': ranked_out,
                    'source_row_index': pos,
                }
                sft_lines.append(out_rec)
                stats.accepted += 1
                accepted_here = True
            finally:
                append_attempt_log(
                    pos,
                    label_log=label,
                    attempted_call=attempted_call,
                    accepted_here=accepted_here,
                    reject_reason=reject_reason,
                    ranked_out=ranked_out,
                    assistant_raw=assistant_raw,
                    gen_err=gen_err,
                )
                pbar.set_postfix_str(_teacher_accept_postfix(stats.accepted, stats.attempted))
                pbar.update(1)
    pbar.close()

    if stats.attempted > 0:
        pct = 100.0 * stats.accepted / stats.attempted
        print(
            f'[teacher] reasoning generation: {stats.accepted}/{stats.attempted} accepted ({pct:.1f}%)'
        )
    else:
        print('[teacher] reasoning generation: no teacher generations attempted')

    if attempts_path is not None:
        write_jsonl(attempts_path, teacher_attempt_logs)
        print(f'[teacher] per-attempt log (raw outputs + parsed ranks): {attempts_path}')

    write_jsonl(sft_out, sft_lines)

    stats.grpo_from_sft_rejects = len(grpo_from_sft_rejects)
    grpo_all = grpo_rows + grpo_from_sft_rejects
    write_jsonl(grpo_out, grpo_all)

    meta = {
        'seed': seed,
        'teacher_model_dir': str(model_dir),
        'train_split': str(train_path),
        'max_rows_for_sft_pool': max_sft_rows,
        'shuffle_before_split': shuffle,
        'batch_size': batch_size,
        'max_prompt_tokens': max_prompt_tokens,
        'sft_output': str(sft_out),
        'grpo_output': str(grpo_out),
        'attempted': stats.attempted,
        'accepted': stats.accepted,
        'rejected': stats.rejected,
        'grpo_reserve_rows': stats.grpo_reserve_rows,
        'grpo_from_sft_rejects': stats.grpo_from_sft_rejects,
        'grpo_total_rows': stats.grpo_total_rows,
        'teacher_attempts_jsonl': str(attempts_path) if attempts_path else None,
        'errors': stats.errors[:50],
    }
    ensure_dir(meta_out.parent)
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    return stats
