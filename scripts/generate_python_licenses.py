#!/usr/bin/env python3
"""Generate a license inventory for an installed See-Through virtual environment."""

from __future__ import annotations

import argparse
from email.parser import Parser
from html import escape
from pathlib import Path


FALLBACKS = {
    "antlr4-python3-runtime": ("BSD-3-Clause", "https://github.com/antlr/antlr4/blob/master/LICENSE.txt"),
    "Cython": ("Apache-2.0", "https://github.com/cython/cython/blob/master/LICENSE.txt"),
    "live2d-annotators": ("Apache-2.0", "https://github.com/shitagaki-lab/see-through/blob/main/LICENSE"),
    "live2d-common": ("Apache-2.0", "https://github.com/shitagaki-lab/see-through/blob/main/LICENSE"),
    "pycocotools": ("BSD-2-Clause", "https://github.com/cocodataset/cocoapi/blob/master/license.txt"),
    "tokenizers": ("Apache-2.0", "https://github.com/huggingface/tokenizers/blob/main/LICENSE"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True, help="See-Through .venv directory")
    parser.add_argument("--output", type=Path, required=True, help="HTML output path")
    return parser.parse_args()


def find_site_packages(environment: Path) -> Path:
    windows = environment / "Lib" / "site-packages"
    if windows.is_dir():
        return windows
    candidates = sorted((environment / "lib").glob("python*/site-packages"))
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"site-packages not found below {environment}")


def license_files(dist_info: Path) -> list[Path]:
    matches = []
    for path in dist_info.rglob("*"):
        if not path.is_file():
            continue
        parts = {part.lower() for part in path.parts}
        name = path.name.lower()
        if "licenses" in parts or name.startswith(("license", "licence", "copying", "notice")):
            matches.append(path)
    return sorted(matches)


def main() -> int:
    args = parse_args()
    site_packages = find_site_packages(args.environment.resolve())
    rows = []
    missing = []
    for dist_info in sorted(site_packages.glob("*.dist-info")):
        metadata_path = dist_info / "METADATA"
        if not metadata_path.is_file():
            continue
        metadata = Parser().parsestr(metadata_path.read_text("utf-8", errors="replace"))
        name = metadata.get("Name", dist_info.name)
        version = metadata.get("Version", "unknown")
        expression = metadata.get("License-Expression") or metadata.get("License") or ""
        project_urls = metadata.get_all("Project-URL", [])
        home = metadata.get("Home-page") or ""
        if project_urls:
            first_url = project_urls[0]
            home = first_url.split(",", 1)[-1].strip()
        files = license_files(dist_info)
        fallback = FALLBACKS.get(name)
        if not files and fallback:
            expression = expression or fallback[0]
            home = fallback[1]
        elif not files:
            missing.append(f"{name} {version}")
        rows.append((name, version, expression or "not declared", home, files))

    if missing:
        raise RuntimeError("No license file or audited fallback for: " + ", ".join(missing))

    sections = []
    for name, version, expression, home, files in sorted(rows, key=lambda row: row[0].lower()):
        title = f"{name} {version}"
        linked_title = f'<a href="{escape(home, quote=True)}">{escape(title)}</a>' if home else escape(title)
        texts = []
        for path in files:
            relative = path.relative_to(site_packages)
            text = path.read_text("utf-8", errors="replace").strip()
            texts.append(f"<h3>{escape(str(relative))}</h3><pre>{escape(text)}</pre>")
        if not texts:
            texts.append(
                "<p><em>The installed wheel did not contain a separate license file. "
                "The audited upstream license link is provided above; this package is downloaded "
                "into the user-managed See-Through environment and is not bundled in the installer.</em></p>"
            )
        sections.append(
            f"<section><h2>{linked_title}</h2><p>License: <code>{escape(expression)}</code></p>"
            + "\n".join(texts)
            + "</section>"
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>See-Through Python Dependency Licenses</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.5}"
        "section{border-top:1px solid #bbb;padding:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f4f6f8;padding:1rem}</style></head><body>"
        "<h1>See-Through Python Dependency Licenses</h1>"
        "<p>Snapshot of the app-managed See-Through environment. These packages are fetched after "
        "an explicit setup action and are not bundled in the PachiPakuGen installer.</p>"
        + "\n".join(sections)
        + "</body></html>\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} dependency notices to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
