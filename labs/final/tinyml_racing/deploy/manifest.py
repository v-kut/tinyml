"""`manifest.json`: what the artifact bundle contains, and what produced it.

The index over `artifacts/`: names, sizes, SHA-256s, the boundary shapes and
dtypes, the quantization scheme, and the headline error numbers. Written last,
after the files it hashes, so a manifest that exists describes a finished bundle.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tinyml_racing.deploy.artifact import EXPORT_VERSION, ActorExport
from tinyml_racing.deploy.quantize import QuantModel
from tinyml_racing.utils import Run

# Bump when a key changes meaning or disappears. Readers outside this repo are
# the whole point of the file, so its shape is a contract like `actor.npz`'s.
MANIFEST_VERSION = 2

# Read in 1 MiB blocks: `actor.npz` is half a megabyte today, but it carries a
# calibration set that scales with `--n-calibration`.
_BLOCK = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path.open(path, "rb") as fh:
        while block := fh.read(_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def bundle(run: Run) -> list[dict[str, Any]]:
    """Every artifact the run has produced, in the order `build()` writes them.

    Missing files are skipped rather than reported as absent: the caller
    decides what a complete bundle is, and this is the inventory of one.
    """
    described = (
        (run.actor_npz, "exported actor: weights, normalization, calibration, references"),
        (run.header, "int8 weights as a C header, this is what the board runs"),
        (run.actor_onnx, "float32 reference graph, raw observations in, action out"),
        (run.report, "float32 vs int8 vs board over held-out tracks"),
    )
    return [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "what": what,
        }
        for path, what in described
        if path.is_file()
    ]


def _headline(results: dict[str, Any]) -> dict[str, Any]:
    """The two numbers that decide whether the quantized model shipped: how far
    its actions moved, and how much lap reward that cost. The table they come
    from is `report.json`, listed in `files`.
    """
    rows = {row["name"]: row for row in results["models"]}
    int8 = rows.get("int8", {})
    return {
        "n_tracks": results.get("n_tracks"),
        # Which detector those laps ran behind. `int8_reward_delta` is a ratio
        # inside one condition, but the reward it is a fraction of is not, and
        # a headline number nobody can attribute is one nobody can compare.
        "sensor": results.get("sensor"),
        "int8_action_max_error": int8.get("max"),
        "int8_reward_delta": int8.get("reward_delta"),
        "board_verified": "board" in rows,
    }


def write_manifest(
    run: Run, export: ActorExport, model: QuantModel, results: dict[str, Any]
) -> Path:
    """Describe the bundle in `run.artifacts`, and return the manifest path."""
    git = None
    if run.config.is_file():
        git = json.loads(run.config.read_text()).get("git_revision")

    payload: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": {
            "name": run.name,
            "dir": str(run.root),
            "git_revision": git,
            "timesteps": export.num_timesteps,
        },
        "model": {
            "layers": list(model.dims),
            "activation": model.activation,
            "parameters": export.n_params,
            # The signature a caller needs to use `actor.onnx` without reading
            # any of this repo: raw sensor values in, servo commands out.
            "input": {
                "name": "obs",
                "dtype": "float32",
                "shape": ["batch", model.n_in],
                "note": "raw observations; the VecNormalize affine is folded into layer 0",
            },
            "output": {
                "name": "action",
                "dtype": "float32",
                "shape": ["batch", model.n_out],
                "note": "steer, throttle/brake, clipped to [-1, 1]",
            },
            "export_version": EXPORT_VERSION,
        },
        "quantization": {
            "scheme": (
                "post-training int8, symmetric, no zero points; "
                "weights per output channel, activations per tensor"
            ),
            "digest": f"0x{model.digest():08x}",
            # `flash_bytes` is what codegen emits; `deployed_flash_bytes` adds
            # the tanh table `tinyml.h` carries, and `compression` quotes that
            # one, because an optimistic flash budget is the dangerous one.
            "flash_bytes": model.flash_bytes,
            "deployed_flash_bytes": model.deployed_flash_bytes,
            "float_bytes": export.float_bytes,
            "compression": export.float_bytes / model.deployed_flash_bytes,
            # The board reports its own digest at handshake, so a mismatch
            # against this one names a stale flash instead of a bad policy.
            "runs_on": "arduino sketch, tinyml.h",
        },
        "evaluation": _headline(results),
        "files": bundle(run),
    }
    run.manifest.parent.mkdir(parents=True, exist_ok=True)
    run.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return run.manifest
