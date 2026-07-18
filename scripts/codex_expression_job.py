#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Codex-assisted expression generated-parts workflow.

This script is intentionally small and file-based:

1. `prepare <job-dir>` creates `_codex/` folders and a prompt for Codex ImageGen.
2. Codex generates expression part images and places/saves them under `_codex/generated_parts/`.
3. `finish <job-dir>` composites only the detected mouth/eye edit region back onto
   the original image, writes final PNGs, and creates a ZIP.

The script does not call Codex or OpenAI APIs by itself. Codex ImageGen remains a
conversation-side step, while all local image handling is reproducible.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


MOUTH_TARGETS = (
    "mouth-closed",
    "mouth-a",
    "mouth-i",
    "mouth-u",
    "mouth-e",
    "mouth-o",
)
EYE_TARGETS = ("eyes-closed",)
GENERATED_PART_TARGETS = (*MOUTH_TARGETS, *EYE_TARGETS)
FINAL_TARGETS = ("eyes-open", *GENERATED_PART_TARGETS)
SOURCE_NAMES = ("source.png", "original.png", "input.png", "base.png")
REFERENCE_NAMES = ("reference.png", "ref.png", "style-reference.png")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"image read failed: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
        image = np.concatenate([image, alpha], axis=2)
    elif image.shape[2] != 4:
        raise ValueError(f"unsupported channel count: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"png encode failed: {path}")
    encoded.tofile(str(path))


def find_source(job_dir: Path) -> Path:
    for name in SOURCE_NAMES:
        candidate = job_dir / name
        if candidate.is_file():
            return candidate

    ignored = {
        "manifest.json",
        "codex_job.json",
        "codex_request.md",
        *(f"{target}.png" for target in FINAL_TARGETS),
    }
    for candidate in sorted(job_dir.iterdir()):
        if candidate.name in ignored or candidate.name.startswith("_"):
            continue
        if candidate.suffix.lower() in IMAGE_EXTENSIONS and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "source image not found. Put source.png, original.png, input.png, or one image file in the job folder."
    )


def find_reference(job_dir: Path) -> Path | None:
    for name in REFERENCE_NAMES:
        candidate = job_dir / name
        if candidate.is_file():
            return candidate
    return None


def codex_dir(job_dir: Path) -> Path:
    return job_dir / "_codex"


def generated_parts_dir(root: Path) -> Path:
    primary = root / "generated_parts"
    if primary.is_dir():
        return primary
    legacy = root / "donors"
    if legacy.is_dir():
        return legacy
    return primary


def prepare(job_dir: Path) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    source = find_source(job_dir)
    source_image = read_image(source)
    root = codex_dir(job_dir)
    parts_dir = root / "generated_parts"
    masks_dir = root / "masks"
    candidates_dir = root / "candidates"
    for directory in (parts_dir, masks_dir, candidates_dir):
        directory.mkdir(parents=True, exist_ok=True)

    canonical_source = root / "source.png"
    write_png(canonical_source, source_image)

    reference = find_reference(job_dir)
    canonical_reference = None
    if reference is not None:
        canonical_reference = root / "reference.png"
        shutil.copyfile(reference, canonical_reference)

    request_text = build_codex_request(job_dir, canonical_source, canonical_reference)
    (root / "codex_request.md").write_text(request_text, encoding="utf-8")
    (root / "codex_job.json").write_text(
        json.dumps(
            {
                "formatVersion": 1,
                "mode": "codex-generated-parts",
                "status": "waitingForGeneratedParts",
                "source": str(canonical_source),
                "reference": str(canonical_reference) if canonical_reference else None,
                "generatedPartsDirectory": str(parts_dir),
                "expectedGeneratedParts": list(GENERATED_PART_TARGETS),
                "createdAt": now_iso(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"prepared: {job_dir}")
    print(f"source: {canonical_source}")
    if canonical_reference:
        print(f"reference: {canonical_reference}")
    print(f"request: {root / 'codex_request.md'}")
    print(f"save Codex ImageGen generated part PNGs to: {parts_dir}")


def build_codex_request(
    job_dir: Path, source: Path, reference: Path | None
) -> str:
    reference_line = (
        f"- Optional reference image: `{reference}`\n"
        "  Use it only for mouth interior, teeth, tongue, iris, and eyelid rendering style.\n"
        if reference
        else "- Optional reference image: none.\n"
    )
    targets = "\n".join(f"  - `{name}.png`" for name in GENERATED_PART_TARGETS)
    return f"""# Codex ImageGen Generated Parts Request

Use the visible source image as the only edit canvas.

- Source image: `{source}`
{reference_line}- Codex generated parts output directory: `{job_dir / '_codex' / 'generated_parts'}`

Generate full-frame expression part source images. The final app/script will extract only the
mouth or eyelid area and composite it back onto the source image, so drift
outside the edited facial part is less important than clean local alignment.

Hard invariants:

- Preserve character identity, canvas size, pose, camera angle, hair silhouette, outfit, accessories, choker, hands, and background as much as possible.
- Do not edit the nose, neck, chin outline, back hair, clothes, or accessories.
- Keep the mouth centered at the original mouth position.
- For mouth targets, edit only the lips, teeth, tongue, and mouth interior.
- For `eyes-closed`, edit only the eyelids/eyelashes; keep eyebrows unchanged.
- Keep anime linework and palette consistent with the source.

Required generated part filenames:

{targets}

Mouth shapes:

- `mouth-closed`: softly closed lips, no opening.
- `mouth-a`: Japanese A vowel, vertical opening, mouth interior visible.
- `mouth-i`: Japanese I vowel, horizontal clean white teeth strip, minimal dark/pink interior.
- `mouth-u`: Japanese U vowel, small rounded oval, no teeth.
- `mouth-e`: Japanese E vowel, spread opening, upper teeth may be visible.
- `mouth-o`: Japanese O vowel, larger rounded hollow opening, no teeth.
- `eyes-closed`: natural blink, both eyes fully closed, unchanged eyebrows.
"""


def status(job_dir: Path) -> None:
    parts_dir = generated_parts_dir(codex_dir(job_dir))
    missing = [name for name in GENERATED_PART_TARGETS if not (parts_dir / f"{name}.png").is_file()]
    if missing:
        print("missing generated parts:")
        for name in missing:
            print(f"  - {name}.png")
    else:
        print("all generated part files are present")


def finish(job_dir: Path, target_names: Iterable[str], zip_outputs: bool) -> None:
    root = codex_dir(job_dir)
    source_path = root / "source.png"
    if not source_path.is_file():
        source_path = find_source(job_dir)
    source = read_image(source_path)
    write_png(job_dir / "eyes-open.png", source)

    parts_dir = generated_parts_dir(root)
    masks_dir = root / "masks"
    candidates_dir = root / "candidates"
    masks_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    generated_files = [str(job_dir / "eyes-open.png")]
    warnings: list[str] = []
    selected = list(target_names)
    if not selected:
        selected = list(GENERATED_PART_TARGETS)

    for target in selected:
        if target not in GENERATED_PART_TARGETS:
            warnings.append(f"unknown target skipped: {target}")
            continue
        part_path = parts_dir / f"{target}.png"
        if not part_path.is_file():
            warnings.append(f"missing generated part skipped: {part_path}")
            continue

        generated_part = read_image(part_path)
        generated_part = resize_like(generated_part, source)
        write_png(candidates_dir / f"{target}.png", generated_part)

        mask_path = masks_dir / f"{target}.png"
        if mask_path.is_file():
            mask = read_mask(mask_path, source.shape[:2])
        else:
            kind = "eye" if target.startswith("eyes-") else "mouth"
            mask = infer_edit_mask(source, generated_part, kind)
            write_png(mask_path, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA))

        adjusted_part = match_candidate_to_source_boundary(source, generated_part, mask)
        result = composite_inside_mask(source, adjusted_part, mask)
        output_path = job_dir / f"{target}.png"
        write_png(output_path, result)
        generated_files.append(str(output_path))

    manifest = {
        "formatVersion": 1,
        "engine": "codex-imagegen",
        "mode": "codex-generated-parts",
        "source": str(source_path),
        "generatedPartsDirectory": str(parts_dir),
        "maskDirectory": str(masks_dir),
        "targets": selected,
        "generatedFiles": generated_files,
        "warnings": warnings,
        "createdAt": now_iso(),
        "note": "Codex generated parts are created by Codex ImageGen outside this script; this script performs local mask inference and compositing only.",
    }
    (job_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if zip_outputs:
        zip_path = job_dir / f"{job_dir.name}.zip"
        write_zip(zip_path, job_dir, [Path(path) for path in generated_files] + [job_dir / "manifest.json"])
        print(f"zip: {zip_path}")

    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print(f"finished: {job_dir}")


def resize_like(image: np.ndarray, source: np.ndarray) -> np.ndarray:
    h, w = source.shape[:2]
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask_image = read_image(path)
    mask = mask_image[:, :, 3] if mask_image.shape[2] == 4 else mask_image[:, :, 0]
    h, w = size
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask


def infer_edit_mask(source: np.ndarray, generated_part: np.ndarray, kind: str) -> np.ndarray:
    h, w = source.shape[:2]
    source_bgr = source[:, :, :3].astype(np.int16)
    generated_bgr = generated_part[:, :, :3].astype(np.int16)
    diff = np.max(np.abs(source_bgr - generated_bgr), axis=2).astype(np.uint8)

    roi = np.zeros((h, w), dtype=np.uint8)
    if kind == "eye":
        x1, x2 = int(w * 0.18), int(w * 0.82)
        y1, y2 = int(h * 0.18), int(h * 0.46)
        anchor = (w * 0.5, h * 0.32)
        max_components = 6
        dilate_iter = 3
        blur_size = 7
    else:
        x1, x2 = int(w * 0.30), int(w * 0.70)
        y1, y2 = int(h * 0.32), int(h * 0.58)
        anchor = (w * 0.5, h * 0.43)
        max_components = 3
        dilate_iter = 4
        blur_size = 9
    roi[y1:y2, x1:x2] = 255

    roi_values = diff[roi > 0]
    threshold = max(14, int(np.percentile(roi_values, 87))) if roi_values.size else 18
    mask = np.where((diff >= threshold) & (roi > 0), 255, 0).astype(np.uint8)
    mask = cv2.medianBlur(mask, 3)
    mask = keep_nearby_components(mask, anchor, max_components=max_components)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return mask


def keep_nearby_components(
    mask: np.ndarray, anchor: tuple[float, float], max_components: int
) -> np.ndarray:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask

    components = []
    ax, ay = anchor
    image_area = mask.shape[0] * mask.shape[1]
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < max(12, image_area // 50000):
            continue
        cx, cy = centroids[label]
        distance = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
        components.append((distance, -area, label))

    if not components:
        return mask
    components.sort()
    keep = {label for _, _, label in components[:max_components]}
    return np.where(np.isin(labels, list(keep)), 255, 0).astype(np.uint8)


def match_candidate_to_source_boundary(
    source: np.ndarray, generated_part: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    hard = (mask > 16).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    outer = cv2.dilate(hard, kernel, iterations=2)
    inner = cv2.erode(hard, kernel, iterations=1)
    ring = (outer > 0) & (inner == 0)
    if ring.sum() < 20:
        return generated_part

    source_rgb = source[:, :, :3].astype(np.int16)
    generated_rgb = generated_part[:, :, :3].astype(np.int16)
    delta = np.median(source_rgb[ring] - generated_rgb[ring], axis=0)
    if not np.all(np.isfinite(delta)):
        return generated_part

    adjusted = generated_part.copy().astype(np.int16)
    adjusted[:, :, :3] = np.clip(adjusted[:, :, :3] + delta * 0.55, 0, 255)
    return adjusted.astype(np.uint8)


def composite_inside_mask(source: np.ndarray, generated_part: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    result = source.copy().astype(np.float32)
    result[:, :, :3] = source[:, :, :3].astype(np.float32) * (1.0 - alpha) + generated_part[
        :, :, :3
    ].astype(np.float32) * alpha
    result[:, :, 3] = source[:, :, 3]
    return np.clip(result, 0, 255).astype(np.uint8)


def write_zip(zip_path: Path, job_dir: Path, files: list[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, path.relative_to(job_dir))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create a Codex generated-parts job folder")
    prepare_parser.add_argument("job_dir", type=Path)

    status_parser = subparsers.add_parser("status", help="List missing generated part files")
    status_parser.add_argument("job_dir", type=Path)

    finish_parser = subparsers.add_parser("finish", help="Composite generated part files back onto the source")
    finish_parser.add_argument("job_dir", type=Path)
    finish_parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Target name to finish. Repeatable. Default: all generated part targets.",
    )
    finish_parser.add_argument("--no-zip", action="store_true", help="Do not create a ZIP")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "prepare":
            prepare(args.job_dir.resolve())
        elif args.command == "status":
            status(args.job_dir.resolve())
        elif args.command == "finish":
            finish(args.job_dir.resolve(), args.target, not args.no_zip)
        else:
            raise ValueError(f"unknown command: {args.command}")
    except Exception as exc:  # noqa: BLE001 - command-line error reporting
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
