"""The config tree as JSON, and the git SHA recorded beside it.

`config.json` is written once per run and read back much later by the viewer,
the exporter and `deploy/evaluate.py`, so the rebuild half is tolerant: unknown
keys are dropped and missing ones default. Both directions live here, together.
"""

import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Typeshed's "any dataclass" protocol, which is what `fields()` accepts. Only
    # a bound, and PEP 695 evaluates those lazily, so it costs nothing at runtime.
    from _typeshed import DataclassInstance

from tinyml_racing.ml.config import RacingEnvConfig, TrainConfig


def git_revision() -> str | None:
    """Short SHA of the working tree, or None outside a git checkout.

    Recorded with every run so a stale reward curve is traceable to its code.
    """
    # Located through `shutil.which` and run by absolute path, so a `git`
    # dropped earlier in PATH than the real one cannot answer for it.
    git_path = shutil.which("git")
    if git_path is None:
        return None

    try:
        # An argv list, never a shell. `check=False` is explicit: a non-zero
        # exit is what "not a checkout" looks like, and is inspected below.
        out = subprocess.run(  # noqa: S603
            [git_path, "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None

    if out.returncode != 0:
        return None

    revision = out.stdout.strip()
    return revision or None


def dataclass_to_mapping(config: Any) -> dict[str, Any]:
    """A dataclass tree as nested JSON-safe dicts.

    Plain `asdict`, named so it sits beside its inverse: every payload here that
    stores a config stores this shape, and nothing else reads it back.
    """
    return asdict(config)


def dataclass_from_mapping[T: DataclassInstance](cls: type[T], data: Mapping[str, Any]) -> T:
    """Rebuild a config from parsed JSON, keeping only the fields the class still
    declares, so a run recorded by older code still opens.

    `cls` is instantiated bare first: those defaults fill in what the payload
    does not carry, and they are also what says which fields are tuples.
    """
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        default = getattr(defaults, f.name)
        if is_dataclass(default):
            kwargs[f.name] = dataclass_from_mapping(type(default), data[f.name])
        else:
            # JSON has no tuples, so a tuple field round-trips as a list.
            kwargs[f.name] = tuple(data[f.name]) if isinstance(default, tuple) else data[f.name]
    return cls(**kwargs)


def config_to_dict(env_cfg: RacingEnvConfig, train_cfg: TrainConfig) -> dict[str, Any]:
    """The whole `config.json` payload for one run."""
    return {
        "env": dataclass_to_mapping(env_cfg),
        "train": dataclass_to_mapping(train_cfg),
        "git_revision": git_revision(),
    }


def env_config_from_dict(data: Mapping[str, Any]) -> RacingEnvConfig:
    """Rebuild an env config from `config.json`'s `env` block.

    Tolerant of fields added since the run was written, because the viewer must
    still open old runs.
    """
    return dataclass_from_mapping(RacingEnvConfig, data)
