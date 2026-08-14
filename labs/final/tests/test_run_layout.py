"""The run directory is a contract between the trainer, which writes it over hours,
and whoever picks the run up afterwards to flash a car or grade a submission.
`utils.Run` states that contract once; `deploy/manifest.py` makes the resulting
bundle self-describing. Defended here is the part that goes silently wrong
rather than loudly broken: a checkpoint sweep that deletes the newest file
because it sorted names as text, and a manifest that hashes a file it did not
read.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from tinyml_racing.deploy.manifest import MANIFEST_VERSION, bundle, sha256, write_manifest
from tinyml_racing.deploy.quantize import quantize_model
from tinyml_racing.ml.rl.callbacks import RotatingCheckpointCallback
from tinyml_racing.utils import Run


@pytest.fixture
def run(tmp_path) -> Run:
    return Run(tmp_path / "run_x")


@pytest.fixture
def quant_model(export):
    return quantize_model(export)


@pytest.fixture
def results() -> dict[str, Any]:
    """The shape `evaluate_run` returns, trimmed to what the manifest reads."""
    return {
        "n_tracks": 4,
        "models": [
            {"name": "float32", "max": 0.0, "reward_delta": 0.0},
            {"name": "int8", "max": 0.25, "reward_delta": -0.012},
        ],
    }


def test_every_path_lives_under_the_run(run):
    """No property escapes the run directory, and none collides with another."""
    paths = {
        name: getattr(run, name)
        for name in (
            "config",
            "log",
            "tb",
            "training",
            "checkpoints",
            "snapshot",
            "final_model",
            "best_model",
            "vecnormalize",
            "evaluations",
            "artifacts",
            "actor_npz",
            "actor_onnx",
            "header",
            "report",
            "manifest",
        )
    }
    for name, path in paths.items():
        assert run.root in path.parents, name
    files = [p for n, p in paths.items() if n not in ("tb", "training", "checkpoints", "artifacts")]
    assert len(set(files)) == len(files)


def test_resolve_prefers_the_latest_symlink(tmp_path):
    root = tmp_path / "runs"
    (root / "run_a").mkdir(parents=True)
    (root / "run_b").mkdir()
    (root / "latest").symlink_to("run_a", target_is_directory=True)
    assert Run.resolve(None, root).root == (root / "run_a").resolve()


def test_resolve_falls_back_to_the_newest_run(tmp_path):
    """A run whose symlink was never written still resolves, by mtime.

    Not by name: `--run-name` makes the names arbitrary, and the one that
    sorted last (a run called `verify`) had none of the stage files the
    newest run did, so every tool reported the wrong run's gaps.
    """
    root = tmp_path / "runs"
    (root / "verify").mkdir(parents=True)
    newest = root / "run_20260102_000000"
    newest.mkdir()
    os.utime(root / "verify", (1_000_000, 1_000_000))
    assert Run.resolve(None, root).root.name == "run_20260102_000000"


def test_resolve_reports_a_dangling_latest(tmp_path):
    """Deleting the run `latest` points at must not silently resolve another."""
    root = tmp_path / "runs"
    (root / "run_a").mkdir(parents=True)
    (root / "latest").symlink_to("run_gone", target_is_directory=True)
    with pytest.raises(FileNotFoundError, match="run_gone"):
        Run.resolve(None, root)


def test_resolve_refuses_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        Run.resolve(tmp_path / "nope")


def _checkpoints(dir_path, steps):
    for n in steps:
        (dir_path / f"ppo_racing_{n}_steps.zip").write_bytes(b"z")
        (dir_path / f"ppo_racing_vecnormalize_{n}_steps.pkl").write_bytes(b"p")


def test_rotation_keeps_the_newest_by_step_count(run):
    """The trap is lexicographic order: as text, `_1000000_` sorts before
    `_999999_`, so a name sort deletes the newest checkpoint and keeps the
    oldest, and the run only finds out when it tries to resume.
    """
    run.checkpoints.mkdir(parents=True)
    steps = [100_000, 999_999, 1_000_000, 9_000_000, 10_000_000]
    _checkpoints(run.checkpoints, steps)

    cb = RotatingCheckpointCallback(
        save_freq=1, save_path=str(run.checkpoints), name_prefix="ppo_racing", keep=2
    )
    cb._prune()

    assert sorted(p.name for p in run.checkpoints.glob("*.zip")) == [
        "ppo_racing_10000000_steps.zip",
        "ppo_racing_9000000_steps.zip",
    ]
    # The normalization statistics rotate with the policy they belong to; a
    # checkpoint without them cannot be resumed from.
    assert sorted(p.name for p in run.checkpoints.glob("*.pkl")) == [
        "ppo_racing_vecnormalize_10000000_steps.pkl",
        "ppo_racing_vecnormalize_9000000_steps.pkl",
    ]


def test_rotation_refuses_to_keep_nothing(run):
    with pytest.raises(ValueError, match="at least 1"):
        RotatingCheckpointCallback(save_freq=1, save_path=str(run.checkpoints), keep=0)


def test_manifest_describes_the_files_that_exist(run, export, quant_model, results):
    run.artifacts.mkdir(parents=True)
    run.actor_npz.write_bytes(b"npz")
    run.header.write_text("// header\n")
    # `actor.onnx` and `report.json` deliberately absent: the inventory is of
    # what the bundle has, not of what a complete one would have.
    payload = json.loads(write_manifest(run, export, quant_model, results).read_text())

    assert payload["manifest_version"] == MANIFEST_VERSION
    listed = {entry["name"]: entry for entry in payload["files"]}
    assert set(listed) == {"actor.npz", "model.h"}
    assert listed["actor.npz"]["sha256"] == sha256(run.actor_npz)
    assert listed["actor.npz"]["bytes"] == 3
    assert [entry["name"] for entry in bundle(run)] == ["actor.npz", "model.h"]


def test_manifest_carries_the_io_signature_and_headline(run, export, quant_model, results):
    run.artifacts.mkdir(parents=True)
    payload = json.loads(write_manifest(run, export, quant_model, results).read_text())

    assert payload["model"]["layers"] == list(quant_model.dims)
    assert payload["model"]["input"]["shape"] == ["batch", quant_model.n_in]
    assert payload["model"]["output"]["shape"] == ["batch", quant_model.n_out]
    assert payload["run"]["timesteps"] == export.num_timesteps
    assert payload["quantization"]["digest"] == f"0x{quant_model.digest():08x}"
    assert payload["evaluation"]["int8_action_max_error"] == 0.25
    assert payload["evaluation"]["board_verified"] is False
