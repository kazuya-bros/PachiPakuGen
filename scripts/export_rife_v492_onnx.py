#!/usr/bin/env python3
"""Export Practical-RIFE v4.9.2 to PachiPakuGen's dynamic DirectML ONNX form.

This script intentionally uses the official Practical-RIFE model archive and
the official repository implementation.  It does not use the previously
bundled yuvraj108c/TensorStack ONNX graph.

The upstream ``model.warplayer.warp`` caches a grid at the first traced size.
For ONNX export only, this script substitutes an equivalent grid made by
resizing a 2 x 2 endpoint grid to the runtime tensor shape.  This keeps the
official IFNet, v4.9.2 weights, ensemble path and scale list while avoiding a
64 x 64 constant that breaks larger DirectML inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings
import zipfile


PRACTICAL_RIFE_REPOSITORY = "https://github.com/hzwer/Practical-RIFE.git"
PRACTICAL_RIFE_COMMIT = "17d8c7a1005b37f4c97bfee04e316aaec7fdc536"

EXPECTED_HASHES = {
    "archive": "f57de4828ae902eec5c1c518bec05edd510f37919b29d5c138cc0d9072b5b63c",
    "train_log/flownet.pkl": "ef91580a020abb7ddfbd3a51573dc395cf2c2a9530ff653ef3f8a1fc6845857f",
    "train_log/IFNet_HDv3.py": "fadb25d8fc3fb6bac52c834356b7b9e27422c9d5ebb060afe4790e2b52cb0f7b",
    "train_log/RIFE_HDv3.py": "5041316615eeb28c1101a764896522ba24316b8c8f6cb0d57358254551fd936d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"  file:     {path}"
        )


def run_git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def prepare_repository(repo_dir: Path | None, work_dir: Path) -> Path:
    if repo_dir is None:
        repo_dir = work_dir / "Practical-RIFE"
        run_git("clone", "--filter=blob:none", PRACTICAL_RIFE_REPOSITORY, str(repo_dir))
        run_git("checkout", "--detach", PRACTICAL_RIFE_COMMIT, cwd=repo_dir)

    repo_dir = repo_dir.resolve()
    if not (repo_dir / ".git").exists():
        raise RuntimeError(f"Not a Practical-RIFE git checkout: {repo_dir}")

    actual_commit = run_git("rev-parse", "HEAD", cwd=repo_dir)
    if actual_commit != PRACTICAL_RIFE_COMMIT:
        raise RuntimeError(
            "Practical-RIFE checkout is not at the audited commit:\n"
            f"  expected: {PRACTICAL_RIFE_COMMIT}\n"
            f"  actual:   {actual_commit}\n"
            "Check out the expected commit or omit --repo-dir to use a temporary clone."
        )
    return repo_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Official Practical-RIFE v4.9.2 ZIP downloaded from the upstream Google Drive link.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        help=(
            "Optional Practical-RIFE checkout at the audited commit. "
            "When omitted, the official repository is cloned into a temporary directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("rife-v4.9.2-dynamic-dml.onnx"),
        help="Destination ONNX path.",
    )
    parser.add_argument(
        "--verify-ort",
        action="store_true",
        help="Run seeded ONNX Runtime CPU checks at 64, 256 and 512 pixels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = args.archive.resolve()
    output = args.output.resolve()

    if not archive.is_file():
        raise FileNotFoundError(archive)
    check_hash(archive, EXPECTED_HASHES["archive"], "Official v4.9.2 archive")

    # Imports are delayed so `--help` and source verification remain usable
    # before the export-only Python dependencies are installed.
    warnings.filterwarnings("ignore")
    import numpy as np
    import onnx
    import torch
    import torch.nn.functional as F

    def dynamic_warp(ten_input, ten_flow):
        """Match upstream align-corners warp without tracing a fixed grid."""

        corner_grid = ten_flow.new_tensor(
            [
                [[[-1.0, 1.0], [-1.0, 1.0]]],
                [[[-1.0, -1.0], [1.0, 1.0]]],
            ]
        ).reshape(1, 2, 2, 2)
        base_grid = F.interpolate(
            corner_grid,
            size=(ten_flow.shape[2], ten_flow.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        normalized_flow = torch.cat(
            [
                ten_flow[:, 0:1] / ((ten_input.shape[3] - 1.0) / 2.0),
                ten_flow[:, 1:2] / ((ten_input.shape[2] - 1.0) / 2.0),
            ],
            1,
        )
        grid = (base_grid + normalized_flow).permute(0, 2, 3, 1)
        return F.grid_sample(
            input=ten_input,
            grid=grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    with tempfile.TemporaryDirectory(prefix="pachipaku-rife-export-") as temporary:
        work_dir = Path(temporary)
        extracted = work_dir / "official-v4.9.2"
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)

        for relative_path, expected in EXPECTED_HASHES.items():
            if relative_path == "archive":
                continue
            check_hash(extracted / relative_path, expected, relative_path)

        repository = prepare_repository(args.repo_dir, work_dir)

        # Prefer the model implementation shipped in the official v4.9.2
        # archive; use the pinned repository for its `model.warplayer` module.
        sys.path.insert(0, str(repository))
        sys.path.insert(0, str(extracted))
        archived_module = importlib.import_module("train_log.IFNet_HDv3")
        # IFNet imports `warp` by value. Replace only that export-time binding;
        # the official network architecture and weights remain unchanged.
        archived_module.warp = dynamic_warp
        IFNet = archived_module.IFNet

        class ExportableRIFE(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = IFNet()

            def forward(self, img0, img1, timestep):
                _, _, merged = self.net(
                    torch.cat((img0, img1), 1),
                    timestep,
                    [8, 4, 2, 1],
                    False,
                    True,
                    True,
                )
                return merged[3]

        # The upstream warplayer module selects CUDA whenever CUDA is
        # available, so the model and sample tensors must use that same device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ExportableRIFE().to(device).eval()
        state = torch.load(
            extracted / "train_log" / "flownet.pkl",
            map_location=device,
            weights_only=False,
        )
        state = {name.replace("module.", ""): value for name, value in state.items()}
        model.net.load_state_dict(state, strict=True)

        torch.manual_seed(1729)
        img0 = torch.rand(1, 3, 64, 64, device=device)
        img1 = torch.rand(1, 3, 64, 64, device=device)
        timestep = torch.tensor([0.37], dtype=torch.float32, device=device)

        with torch.no_grad():
            reference = model(img0, img1, timestep).detach().cpu().numpy()

        output.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            (img0, img1, timestep),
            str(output),
            input_names=["img0", "img1", "timestep"],
            output_names=["output"],
            dynamic_axes={
                "img0": {2: "height", 3: "width"},
                "img1": {2: "height", 3: "width"},
                "output": {2: "height", 3: "width"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

        onnx_model = onnx.load(str(output))
        onnx.checker.check_model(onnx_model)

        if args.verify_ort:
            import onnxruntime as ort

            session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            rng = np.random.default_rng(1729)
            for size in (64, 256, 512):
                feed = {
                    "img0": rng.random((1, 3, size, size), dtype=np.float32),
                    "img1": rng.random((1, 3, size, size), dtype=np.float32),
                    "timestep": np.array([0.37], dtype=np.float32),
                }
                actual = session.run(["output"], feed)[0]
                if actual.shape != (1, 3, size, size) or not np.isfinite(actual).all():
                    raise RuntimeError(f"ORT verification failed at {size} x {size}")
                print(f"ORT CPU {size}x{size}: shape={actual.shape}, finite=True")

    print(f"Device:  {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"ONNX:    {onnx.__version__}")
    print(f"Output:  {output}")
    print(f"Bytes:   {output.stat().st_size}")
    print(f"SHA-256: {sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
