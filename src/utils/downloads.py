from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from huggingface_hub import snapshot_download

from utils.io import ensure_dir


ARCHIVE_SUFFIXES = ('.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2')


def infer_filename(url: str, configured_name: str | None = None) -> str:
    if configured_name:
        return configured_name
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or 'download.bin'


def infer_archive_type(filename: str, configured: str | None = None) -> str:
    if configured and configured != 'auto':
        return configured
    if filename.endswith('.zip'):
        return 'zip'
    if filename.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2')):
        return 'tar'
    return 'none'


def copy_path_contents(source: Path, destination: Path, dry_run: bool = False) -> None:
    ensure_dir(destination)
    if dry_run:
        print(f'[copy] dry-run source={source} -> {destination}')
        return

    if source.is_file():
        target = destination / source.name
        shutil.copy2(source, target)
        return

    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            ensure_dir(target.parent)
            shutil.copy2(child, target)


def download_url(url: str, destination: Path, dry_run: bool = False) -> Path:
    ensure_dir(destination.parent)
    if dry_run:
        print(f'[download] dry-run url={url} -> {destination}')
        return destination
    print(f'[download] downloading {url} -> {destination}')
    urlretrieve(url, destination)
    return destination


def extract_archive(archive_path: Path, target_dir: Path, archive_type: str, dry_run: bool = False) -> None:
    ensure_dir(target_dir)
    if dry_run:
        print(f'[extract] dry-run archive={archive_path} type={archive_type} -> {target_dir}')
        return
    print(f'[extract] extracting {archive_path} -> {target_dir}')
    if archive_type == 'zip':
        with zipfile.ZipFile(archive_path, 'r') as handle:
            handle.extractall(target_dir)
        return
    if archive_type == 'tar':
        with tarfile.open(archive_path, 'r:*') as handle:
            handle.extractall(target_dir)
        return
    raise ValueError(f'Unsupported archive type: {archive_type}')


def materialize_dataset(url: str, target_dir: Path, filename: str, archive_type: str, dry_run: bool = False) -> None:
    ensure_dir(target_dir)
    download_path = target_dir / filename
    download_url(url, download_path, dry_run=dry_run)
    actual_archive_type = infer_archive_type(filename, archive_type)
    if actual_archive_type != 'none':
        extract_archive(download_path, target_dir, actual_archive_type, dry_run=dry_run)


def materialize_kagglehub_dataset(
    dataset_handle: str,
    target_dir: Path,
    sync_mode: str = 'copy',
    dry_run: bool = False,
) -> None:
    ensure_dir(target_dir)
    if dry_run:
        print(f'[kagglehub] dry-run dataset_handle={dataset_handle} -> {target_dir}')
        return

    try:
        import kagglehub
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'kagglehub is required for Kaggle dataset downloads. Install dependencies with '
            '`uv sync` or `uv add kagglehub`.') from exc

    print(f'[kagglehub] downloading dataset_handle={dataset_handle}')
    source_path = Path(kagglehub.dataset_download(dataset_handle))
    print(f'[kagglehub] cached source path: {source_path}')

    if sync_mode != 'copy':
        raise ValueError(f'Unsupported sync mode: {sync_mode}')

    copy_path_contents(source_path, target_dir, dry_run=False)
    print(f'[kagglehub] copied dataset contents into {target_dir}')


def materialize_hf_snapshot(
    model_id: str,
    local_dir: Path,
    revision: str = 'main',
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    ensure_dir(local_dir)
    if dry_run:
        print(f'[hf] dry-run snapshot model_id={model_id} revision={revision} -> {local_dir}')
        if allow_patterns:
            print(f'[hf] allow_patterns={allow_patterns}')
        if ignore_patterns:
            print(f'[hf] ignore_patterns={ignore_patterns}')
        return
    print(f'[hf] downloading model_id={model_id} revision={revision} -> {local_dir}')
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=revision,
        allow_patterns=allow_patterns or None,
        ignore_patterns=ignore_patterns or None,
    )


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
