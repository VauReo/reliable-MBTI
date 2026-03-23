from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from dataset.splits import stratified_train_val_test
from utils.io import ensure_dir

ROOT = Path(__file__).resolve().parents[2]


def _resolve(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def _split_posts(raw: str, delimiter: str = "|||") -> list[str]:
    if not raw:
        return []
    parts = raw.split(delimiter)
    return [p.strip() for p in parts if p.strip()]


def _truncate_post(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _maybe_remove_mbti_mentions(text: str) -> str:
    if not text:
        return text
    pattern = re.compile(r"\b[IE][NS][TF][JP]\b", re.IGNORECASE)
    return pattern.sub("[TYPE]", text)


def build_text_from_posts(
    posts: str,
    *,
    max_posts_per_user: int,
    max_chars_per_post: int | None,
    remove_direct_mbti_mentions: bool,
    lowercase: bool,
) -> str:
    chunks = _split_posts(posts)
    if max_posts_per_user > 0:
        chunks = chunks[:max_posts_per_user]
    out: list[str] = []
    for c in chunks:
        t = _truncate_post(c, max_chars_per_post)
        if remove_direct_mbti_mentions:
            t = _maybe_remove_mbti_mentions(t)
        if lowercase:
            t = t.lower()
        out.append(t)
    return "\n\n".join(out)


def load_kaggle_csvs(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=False))
    df = pd.concat(frames, ignore_index=True)
    if "type" not in df.columns or "posts" not in df.columns:
        raise ValueError(
            f"Expected columns type, posts in Kaggle CSVs under {input_dir}, got {list(df.columns)}"
        )
    return df


def dataframe_to_records(
    df: pd.DataFrame,
    source_dataset: str,
    *,
    create_cleaned_variant: bool,
    preproc: dict,
) -> list[dict]:
    records: list[dict] = []
    max_posts = int(preproc.get("max_posts_per_user", 128))
    max_chars = preproc.get("max_chars_per_post")
    max_chars_i = int(max_chars) if max_chars is not None else None
    remove_mbti = bool(preproc.get("remove_direct_mbti_mentions", False))
    lowercase = bool(preproc.get("lowercase", False))

    for _, row in df.iterrows():
        label = str(row["type"]).strip().upper()
        raw_posts = str(row["posts"])
        text = build_text_from_posts(
            raw_posts,
            max_posts_per_user=max_posts,
            max_chars_per_post=max_chars_i,
            remove_direct_mbti_mentions=remove_mbti,
            lowercase=lowercase,
        )
        rec: dict = {
            "label": label,
            "text": text,
            "source_dataset": source_dataset,
        }
        if create_cleaned_variant:
            rec["text_clean"] = build_text_from_posts(
                raw_posts,
                max_posts_per_user=max_posts,
                max_chars_per_post=max_chars_i,
                remove_direct_mbti_mentions=True,
                lowercase=lowercase,
            )
        records.append(rec)
    return records


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ingest_pandora_jsonl(input_dir: Path, preproc: dict, create_cleaned: bool) -> list[dict]:
    paths = sorted(input_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"Pandora raw dir {input_dir} has no .jsonl files; add JSONL or disable pandora."
        )
    rows: list[dict] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    if not rows:
        raise ValueError(f"No lines read from JSONL under {input_dir}")

    typed: list[tuple[str, str]] = []
    for obj in rows:
        label = obj.get("label") or obj.get("type") or obj.get("mbti")
        if label is None:
            continue
        raw = obj.get("posts")
        if raw is None:
            raw = obj.get("text")
        if raw is None:
            continue
        typed.append((str(label).strip().upper(), str(raw)))

    if not typed:
        raise ValueError(f"No records with (label, text/posts) in {input_dir}")

    df = pd.DataFrame(typed, columns=["type", "posts"])
    return dataframe_to_records(
        df,
        source_dataset="pandora",
        create_cleaned_variant=create_cleaned,
        preproc=preproc,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class DatasetPreprocessor:
    """Ingest raw MBTI sources, write processed JSONL, and stratified train/val/test splits."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self) -> None:
        processed_dir = _resolve(self.config["processed_data_dir"])
        splits_dir = _resolve(self.config["splits_dir"])
        seed = int(self.config["seed"])
        preproc = self.config.get("preprocessing", {})
        split_cfg = self.config.get("splits", {})
        train_r = float(split_cfg.get("train_ratio", 0.8))
        val_r = float(split_cfg.get("val_ratio", 0.1))
        test_r = float(split_cfg.get("test_ratio", 0.1))

        ensure_dir(processed_dir)
        ensure_dir(splits_dir)

        datasets = self.config.get("datasets", {})
        create_clean = bool(preproc.get("create_cleaned_variant", False))

        for name, spec in datasets.items():
            if not spec.get("enabled", False):
                print(f"[prepare_data] skip dataset {name} (disabled)")
                continue
            input_path = _resolve(spec["input_path"])
            output_name = spec["output_name"]
            print(f"[prepare_data] processing {name} from {input_path}")

            if name == "kaggle_mbti":
                raw_df = load_kaggle_csvs(input_path)
                records = dataframe_to_records(
                    raw_df,
                    source_dataset=name,
                    create_cleaned_variant=create_clean,
                    preproc=preproc,
                )
            elif name == "pandora":
                records = ingest_pandora_jsonl(input_path, preproc, create_clean)
            else:
                print(f"[prepare_data] unknown dataset key {name}, skipping")
                continue

            out_processed = processed_dir / output_name
            write_jsonl(out_processed, records)
            print(f"[prepare_data] wrote {out_processed} ({len(records)} rows)")

            df = records_to_dataframe(records)
            try:
                train_df, val_df, test_df = stratified_train_val_test(
                    df,
                    "label",
                    train_ratio=train_r,
                    val_ratio=val_r,
                    test_ratio=test_r,
                    random_state=seed,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Stratified split failed for {name}. "
                    "Each class needs enough examples in train/val/test. "
                    f"Original error: {exc}"
                ) from exc

            ds_split_dir = splits_dir / name
            ensure_dir(ds_split_dir)
            for split_name, part in (
                ("train", train_df),
                ("val", val_df),
                ("test", test_df),
            ):
                split_path = ds_split_dir / f"{split_name}.jsonl"
                write_jsonl(split_path, part.to_dict("records"))
                print(f"[prepare_data] wrote {split_path} ({len(part)} rows)")
