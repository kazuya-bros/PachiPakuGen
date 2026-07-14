"""Download and verify one pinned See-Through model profile.

The authoritative repository, revision, path, and byte-size table lives in
``see_through_model_requirements.json``. Rust readiness checks consume the same
table, so the downloader cannot redefine what "complete" means.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import Any

# This script is intentionally run in a user-visible console. The Xet backend
# stalled on the supported Windows setup, while standard HTTP keeps a resumable
# .incomplete file. These flags must be set before importing huggingface_hub.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from huggingface_hub import hf_hub_download


SCHEMA_VERSION = 1
HEARTBEAT_SECONDS = 15


def _load_profile_requirements(path: Path, profile: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schemaVersion") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported See-Through model requirement schema")
    repositories = data.get("profiles", {}).get(profile)
    if not isinstance(repositories, list) or not repositories:
        raise RuntimeError(f"No model requirements found for profile: {profile}")

    repo_ids: set[str] = set()
    for repository in repositories:
        repo_id = repository.get("repoId")
        revision = repository.get("revision")
        required_files = repository.get("files")
        if not isinstance(repo_id, str) or not repo_id or repo_id in repo_ids:
            raise RuntimeError(f"Invalid or duplicate model repository: {repo_id}")
        if not isinstance(revision, str) or len(revision) != 40:
            raise RuntimeError(f"Invalid pinned revision: {repo_id}")
        if not isinstance(required_files, list) or not required_files:
            raise RuntimeError(f"No required files listed for: {repo_id}")
        paths: set[str] = set()
        for file in required_files:
            relative = file.get("path")
            size = file.get("size")
            if (
                not isinstance(relative, str)
                or not relative
                or relative in paths
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise RuntimeError(f"Invalid or duplicate model path: {repo_id}/{relative}")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise RuntimeError(f"Invalid model byte size: {repo_id}/{relative}")
            paths.add(relative)
        repo_ids.add(repo_id)
    return repositories


def _relative_file_record(
    path: Path, hf_home: Path, expected_size: int
) -> dict[str, object]:
    absolute = path.absolute()
    relative = Path(os.path.relpath(absolute, hf_home.absolute()))
    if ".." in relative.parts or relative.is_absolute():
        raise RuntimeError(f"Model cache escaped managed HF_HOME: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Model size mismatch: {path} "
            f"(expected {expected_size}, actual {actual_size} bytes)"
        )
    return {"path": relative.as_posix(), "size": expected_size}


def _main_ref_for_snapshot(snapshot: Path) -> Path:
    # <repo cache>/snapshots/<revision> -> <repo cache>/refs/main
    return snapshot.parent.parent / "refs" / "main"


def _write_main_ref(snapshot: Path, revision: str) -> Path:
    main_ref = _main_ref_for_snapshot(snapshot)
    main_ref.parent.mkdir(parents=True, exist_ok=True)
    temporary = main_ref.with_suffix(".tmp")
    temporary.write_text(revision, encoding="utf-8")
    os.replace(temporary, main_ref)
    return main_ref


def _snapshot_for_downloaded_file(model_path: Path, relative_path: str) -> Path:
    """Return the pinned snapshot root without reimplementing HF cache names."""
    snapshot = model_path
    for _ in Path(relative_path).parts:
        snapshot = snapshot.parent
    return snapshot


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _incomplete_bytes(cache_dir: Path, repo_id: str) -> int:
    blobs = cache_dir / f"models--{repo_id.replace('/', '--')}" / "blobs"
    if not blobs.is_dir():
        return 0
    total = 0
    for path in blobs.glob("*.incomplete"):
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _download_file_with_heartbeat(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    cache_dir: Path,
    expected_size: int,
    force_download: bool = False,
) -> Path:
    stop = Event()
    started = monotonic()

    def report_progress() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            partial = _incomplete_bytes(cache_dir, repo_id)
            elapsed = int(monotonic() - started)
            print(
                f"Download heartbeat: {repo_id}/{filename} - "
                f"partial {_format_bytes(partial)} / {_format_bytes(expected_size)}, "
                f"elapsed {elapsed}s",
                flush=True,
            )

    reporter = Thread(target=report_progress, daemon=True)
    reporter.start()
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
            )
        )
    finally:
        stop.set()
        reporter.join(timeout=1)


def _download_repository(
    requirement: dict[str, Any],
    hf_home: Path,
    manifest_files: list[dict[str, object]],
    revisions: dict[str, str],
) -> None:
    repo_id = requirement["repoId"]
    revision = requirement["revision"]
    required_files = requirement["files"]
    cache_dir = hf_home / "hub"

    print(f"Preparing required See-Through model: {repo_id}", flush=True)
    snapshot: Path | None = None
    total = len(required_files)
    for index, file in enumerate(required_files, start=1):
        print(
            f"Model download {index}/{total}: {repo_id}/{file['path']} "
            f"({_format_bytes(file['size'])})",
            flush=True,
        )
        model_path = _download_file_with_heartbeat(
            repo_id=repo_id,
            filename=file["path"],
            revision=revision,
            cache_dir=cache_dir,
            expected_size=file["size"],
        )
        if not model_path.is_file() or model_path.stat().st_size != file["size"]:
            # Hugging Face may reuse an existing pointer without rechecking a blob
            # that was truncated after download. Force only that file to repair it.
            print(f"Repairing incomplete model file: {repo_id}/{file['path']}", flush=True)
            model_path = _download_file_with_heartbeat(
                repo_id=repo_id,
                filename=file["path"],
                revision=revision,
                cache_dir=cache_dir,
                expected_size=file["size"],
                force_download=True,
            )
        candidate_snapshot = _snapshot_for_downloaded_file(model_path, file["path"])
        if not candidate_snapshot.is_dir() or candidate_snapshot.name != revision:
            raise RuntimeError(f"Pinned Hugging Face snapshot is missing: {repo_id}")
        if snapshot is None:
            snapshot = candidate_snapshot
        elif candidate_snapshot != snapshot:
            raise RuntimeError(f"Model files resolved to different snapshots: {repo_id}")
        manifest_files.append(
            _relative_file_record(model_path, hf_home, file["size"])
        )

    if snapshot is None:
        raise RuntimeError(f"Pinned Hugging Face snapshot is missing: {repo_id}")
    main_ref = _write_main_ref(snapshot, revision)
    manifest_files.append(_relative_file_record(main_ref, hf_home, len(revision)))
    revisions[repo_id] = revision


def prepare_models(profile: str, requirements_path: Path, manifest_path: Path) -> None:
    hf_home_text = os.environ.get("HF_HOME", "").strip()
    if not hf_home_text:
        raise RuntimeError("HF_HOME is not set")
    hf_home = Path(hf_home_text)
    hf_home.mkdir(parents=True, exist_ok=True)

    requirements = _load_profile_requirements(requirements_path, profile)
    files: list[dict[str, object]] = []
    revisions: dict[str, str] = {}
    for requirement in requirements:
        _download_repository(requirement, hf_home, files, revisions)

    unique_files = {str(entry["path"]): entry for entry in files}
    if len(unique_files) != len(files):
        raise RuntimeError("Duplicate files were produced for the model manifest")
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "profile": profile,
        "repositories": [requirement["repoId"] for requirement in requirements],
        "revisions": revisions,
        "files": [unique_files[key] for key in sorted(unique_files)],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    print(
        f"See-Through model verification manifest saved: {manifest_path} "
        f"({len(unique_files)} files)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("low-vram", "standard"), required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare_models(args.profile, args.requirements, args.manifest)


if __name__ == "__main__":
    main()
