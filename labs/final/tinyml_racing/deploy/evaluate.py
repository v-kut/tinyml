"""Drive the float32 actor and its int8 compilation around the same held-out
tracks, and optionally the physical board alongside them.

Action-space error is a proxy: two models with the same MAE can lap identically
or crash on opposite corners, so the rows are laps over the seeds `EvalCallback`
held out. `clean_sensor` picks the detector; clean is the comparable default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tinyml_racing import progress
from tinyml_racing.deploy.artifact import ActorExport
from tinyml_racing.deploy.quantize import (
    action_error,
    clipping_rate,
    float_actor,
    quantize_model,
)
from tinyml_racing.ml.config_io import env_config_from_dict
from tinyml_racing.ml.rollout import closed_loop, eval_seeds
from tinyml_racing.sim.car import CarParams
from tinyml_racing.utils import Run


def load_env_config(run: Run):
    with Path.open(run.config) as fh:
        return env_config_from_dict(json.load(fh)["env"])


def evaluate_run(
    run: Run,
    n_tracks: int = 32,
    max_steps: int | None = None,
    board=None,
    clean_sensor: bool = True,
) -> dict[str, Any]:
    """Compare float32, int8 and (optionally) the board over shared tracks.

    `board` is an open `deploy.board.Board`. Including it turns this from
    "does the quantized model still drive" into "does the *hardware*
    still drive", which is the claim the demo actually makes.
    """
    export = ActorExport.load(run.actor_npz)
    env_cfg = load_env_config(run).evaluation_variant(clean_sensor=clean_sensor)
    seeds = eval_seeds(env_cfg, n_tracks)
    # The limit the env would truncate at anyway, rather than a second step
    # budget derived elsewhere. `closed_loop` builds its envs with the default
    # car, so this is its `dt`.
    if max_steps is None:
        max_steps = env_cfg.episode_steps(CarParams().dt)

    model = quantize_model(export)

    # `deployed_flash_bytes`, not `flash_bytes`: the sketch also carries
    # `tinyml.h`'s tanh table, and `manifest.compression` divides by the same
    # figure, so the table's `bytes`/`x` columns cannot drift from the manifest.
    # The baseline clips at the deployed model's own bound, so the two rows
    # differ by quantization error and nothing else.
    variants = [
        ("float32", float_actor(export, model.clip_action), export.float_bytes),
        ("int8", model.act, model.deployed_flash_bytes),
    ]
    if board is not None:
        variants.append(("board", board.act, model.deployed_flash_bytes))

    # One JSON object per model, read back by string key in `_COLUMNS`:
    # heterogeneous by design, so `Any` is the honest value type here.
    rows: list[dict[str, Any]] = []
    with progress.stage("evaluate", len(variants) * len(seeds)) as bar:
        for name, act, n_bytes in variants:
            row: dict[str, Any] = {"name": name, "bytes": int(n_bytes)}
            row.update(
                action_error(np.stack([act(o) for o in export.reference_in]), export.reference_out)
            )
            row.update(
                closed_loop(
                    act,
                    env_cfg,
                    seeds,
                    max_steps,
                    on_lap=lambda i, name=name: bar.advance(
                        1.0, note=f"{name}: track {i}/{len(seeds)}"
                    ),
                )
            )
            rows.append(row)

    base = rows[0]
    for row in rows:
        row["reward_delta"] = (
            (row["reward"] - base["reward"]) / abs(base["reward"]) if base["reward"] else 0.0
        )
        row["compression"] = base["bytes"] / row["bytes"] if row["bytes"] else 0.0

    return {
        "run_dir": str(run.root),
        "arch": model.arch,
        "activation": model.activation,
        "num_timesteps": export.num_timesteps,
        "n_tracks": int(n_tracks),
        "sensor": "clean" if clean_sensor else "noisy",
        "max_steps": int(max_steps),
        "seeds": seeds,
        "clipping_rate": clipping_rate(export),
        "models": rows,
    }


# (header, result key, value format); every column is right-aligned to one width.
_COLUMNS = (
    ("model", "name", ""),
    ("bytes", "bytes", "d"),
    ("x", "compression", ".2f"),
    ("mae", "mae", ".5f"),
    ("max", "max", ".5f"),
    ("reward", "reward", ".1f"),
    ("steps", "steps", ".0f"),
    ("crash", "crash_rate", ".0%"),
    ("v_max", "max_speed", ".2f"),
    ("dreward", "reward_delta", "+.1%"),
)


def format_table(results: dict[str, Any]) -> str:
    """The rows, under a caption naming the laps they came from.

    The sensor belongs on the table rather than beside it: `reward` and `crash`
    mean different things behind a clean detector and behind the training one.
    """
    caption = (
        f"{results['n_tracks']} held-out tracks, {results['sensor']} sensor, "
        f"{results['max_steps']} steps max"
    )
    head = "".join(f"{title:>10s}" for title, _, _ in _COLUMNS)
    rows = [
        "".join(f"{format(row[key], fmt):>10s}" for _, key, fmt in _COLUMNS)
        for row in results["models"]
    ]
    return "\n".join([caption, head, "-" * len(head), *rows])


def write_report(results: dict[str, Any], run: Run) -> Path:
    """The evaluation record, in the run's artifact bundle.

    One file, whether or not a board was attached: the board is a third row in
    the same table, not a second report.
    """
    run.report.parent.mkdir(parents=True, exist_ok=True)
    run.report.write_text(json.dumps(results, indent=2))
    return run.report
