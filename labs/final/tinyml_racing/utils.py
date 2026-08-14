"""Run-directory conventions.

`Run` is the single definition of what a training run writes where, and
`Run.resolve` is how the deploy tools find the newest one without a path
argument (the `latest` symlink `train.link_latest` maintains).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

RUNS_ROOT = Path("data/runs")


@dataclass(frozen=True)
class Run:
    """Every path a run owns, split by audience.

    config.json  env + training settings, and the git SHA
    train.log    tb/  working record
    training/    the optimizer's state; only `snapshot.pt` leaves it
    artifacts/   the deliverable: actor.npz, actor.onnx, model.h,
                 report.json, manifest.json
    """

    root: Path

    @classmethod
    def create(cls, base_dir: str | Path = RUNS_ROOT, run_name: str | None = None) -> Run:
        """A fresh run directory, named for the wall clock unless told otherwise.

        An explicit `run_name` that already exists is an error rather than a
        resumption: the second run would interleave its `config.json`, `train.log`
        and `artifacts/` with the first's, and `Run.resolve`'s mtime ordering would
        then hand the deploy tools a hybrid. The generated name cannot collide
        below one-second resolution, so only the explicit path is refused.
        """
        # Aware, then rendered in local time: the name is read by a human
        # sitting at the machine that produced it.
        named = run_name is not None
        run_name = run_name or f"run_{datetime.now(UTC).astimezone():%Y%m%d_%H%M%S}"
        root = Path(base_dir) / run_name
        if named and root.exists():
            raise FileExistsError(
                f"run directory already exists: {root}; pass a different --run-name "
                "or delete it, so two runs cannot share one directory"
            )
        root.mkdir(parents=True, exist_ok=not named)
        return cls(root)

    @classmethod
    def resolve(cls, run_dir: str | Path | None = None, runs_root: str | Path = RUNS_ROOT) -> Run:
        """An explicit directory, the `latest` symlink the trainer maintains, or
        the newest run under `runs_root`.
        """
        if run_dir is not None:
            path = Path(run_dir)
            if not path.is_dir():
                raise FileNotFoundError(f"run directory does not exist: {path}")
            return cls(path)

        root = Path(runs_root)
        latest = root / "latest"
        if latest.is_dir():
            return cls(latest.resolve())
        if latest.is_symlink():
            # `is_dir()` follows the link, so a dangling one lands here. Falling
            # through would resolve a *different* run and report its missing
            # stages instead of this.
            raise FileNotFoundError(
                f"{latest} points at {latest.readlink()}, which no longer exists, "
                "pass --run-dir, or re-link it at the run you meant"
            )

        # Newest by mtime, not by name: `--run-name` makes names arbitrary, and
        # sorting them put a run called `verify` after every timestamped one.
        candidates = [p for p in root.glob("*") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(
                f"no run directories found in {root.resolve()}, "
                "start a training run first, or pass --run-dir"
            )
        return cls(max(candidates, key=lambda p: p.stat().st_mtime))

    @property
    def name(self) -> str:
        return self.root.resolve().name

    # --- what was trained -------------------------------------------------
    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def log(self) -> Path:
        return self.root / "train.log"

    @property
    def tb(self) -> Path:
        return self.root / "tb"

    # --- training state ---------------------------------------------------
    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def checkpoints(self) -> Path:
        return self.training / "checkpoints"

    @property
    def snapshot(self) -> Path:
        """The newest policy this run can deploy, in `ml/snapshot.py`'s format.

        Every stage overwrites it and PPO republishes it as it goes;
        `deploy/export.py` reads this file and nothing else.
        """
        return self.training / "snapshot.pt"

    @property
    def pretrain(self) -> Path:
        """The cloned policy PPO started from, kept so the warm start can be
        scored against what PPO made of it.
        """
        return self.training / "pretrain.pt"

    @property
    def ppo(self) -> Path:
        """What PPO finished with, distinct from `snapshot.pt`, which
        distillation overwrites.
        """
        return self.training / "ppo.pt"

    @property
    def student(self) -> Path:
        """The distilled student, when that stage ran."""
        return self.training / "student.pt"

    @property
    def final_model(self) -> Path:
        return self.training / "final_model.zip"

    @property
    def best(self) -> Path:
        """The best-scoring policy `EvalCallback` saw, in snapshot format.

        Written by `BestSnapshotCallback` rather than by SB3, whose own
        `best_model.zip` carries no `VecNormalize` statistics and so cannot be
        turned back into something deployable. Distillation teaches from this
        when it exists; absent (no evaluation ran) the last policy is it.
        """
        return self.training / "best.pt"

    @property
    def vecnormalize(self) -> Path:
        return self.training / "vecnormalize.pkl"

    @property
    def evaluations(self) -> Path:
        """`EvalCallback`'s periodic scores, again named by the callback."""
        return self.training / "evaluations.npz"

    # --- the deployment bundle --------------------------------------------
    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def actor_npz(self) -> Path:
        """The exported actor: weights, normalization, calibration, references."""
        return self.artifacts / "actor.npz"

    @property
    def actor_onnx(self) -> Path:
        """The same function as a portable float32 graph."""
        return self.artifacts / "actor.onnx"

    @property
    def header(self) -> Path:
        """The int8 C header that is compiled into the sketch, what ships."""
        return self.artifacts / "model.h"

    @property
    def report(self) -> Path:
        """Float32 vs int8 vs board, over shared held-out tracks."""
        return self.artifacts / "report.json"

    @property
    def manifest(self) -> Path:
        return self.artifacts / "manifest.json"


def save_json(data: dict[str, Any], filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as f:
        json.dump(data, f, indent=4, default=str)


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    console: Any | None = None,
) -> None:
    """Configure the root logger once; only entry points call this.

    `console` is a `rich.console.Console` when a live display is running:
    records then print above the bars instead of through them. The file
    handler keeps the full format either way, `train.log` is the record.
    """
    if console is None:
        stream: logging.Handler = logging.StreamHandler()
        stream.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATEFMT))
    else:
        from rich.logging import RichHandler

        stream = RichHandler(
            console=console, show_path=False, omit_repeated_times=False, markup=False
        )
        stream.setFormatter(logging.Formatter("%(message)s"))

    handlers: list[logging.Handler] = [stream]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(log_file)
        file.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATEFMT))
        handlers.append(file)
    logging.basicConfig(level=level, handlers=handlers, force=True)
