"""Policy snapshot protocol: the trainer writes, the export stage reads.

A snapshot carries the weights, the architecture and spaces needed to rebuild
the network without `PPO.load`, the `VecNormalize` statistics the weights are a
function of, and a version stamp. Publishing is atomic; see the README.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyml_racing.ml.config import PolicyKwargs, RacingEnvConfig
from tinyml_racing.ml.config_io import env_config_from_dict

# 3: no `metrics` bag. 2 replaced version 1's flat `net_arch` with
# `{"pi": [...], "vf": [...]}`. Each step is a bump, not a defaulted key,
# because the difference is not recoverable from the payload.
SNAPSHOT_VERSION = 3


def _atomic(path: Path, write: Callable[[Path], object]) -> Path:
    """Land `path` by rename, so a reader sees the whole previous file or the
    whole new one and never a partial write. Every snapshot write goes through
    here: the stage's own file, and the deliverable the deploy tools poll.
    """
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    Path.replace(tmp, path)
    return path


@dataclass(frozen=True)
class ObsNorm:
    """`VecNormalize`'s observation transform, carried as data.

    The weights are a function of *normalized* observations, so these four
    numbers travel with them everywhere: into `snapshot.pt`, into `actor.npz`,
    and back out of a cloned policy to seed the trainer that inherits it.
    """

    mean: np.ndarray
    var: np.ndarray
    clip_obs: float
    epsilon: float = 1e-8

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # float32 throughout. `mean` and `var` are float64 because that is how
        # `VecNormalize` accumulates and stores them; promoting the observation
        # to match would allocate two float64 temporaries per control step.
        normalized = (np.asarray(obs, dtype=np.float32) - self._mean32) / self._scale32
        return np.clip(normalized, -self.clip_obs, self.clip_obs, out=normalized)

    @cached_property
    def _mean32(self) -> np.ndarray:
        return np.asarray(self.mean, dtype=np.float32)

    @cached_property
    def _scale32(self) -> np.ndarray:
        """`sqrt(var + epsilon)`, narrowed after the fact.

        The addition and the root stay at float64: `epsilon` is 1e-8 against
        variances that can be smaller still, and rounding them together at
        float32 first turns a near-constant channel into a divide-by-zero.
        """
        return np.sqrt(np.asarray(self.var, dtype=np.float64) + self.epsilon).astype(np.float32)

    @classmethod
    def from_venv(cls, venv) -> ObsNorm | None:
        """The statistics of the `VecNormalize` in `venv`'s wrapper stack, if any.

        Walks the stack rather than reading `venv.obs_rms`: `clip_obs` and
        `epsilon` live on the same wrapper, and reproducing the trainer's
        transform needs all four together.
        """
        while venv is not None:
            rms = getattr(venv, "obs_rms", None)
            if rms is not None and getattr(venv, "norm_obs", False):
                return cls(
                    mean=np.asarray(rms.mean, dtype=np.float64),
                    var=np.asarray(rms.var, dtype=np.float64),
                    clip_obs=float(venv.clip_obs),
                    epsilon=float(venv.epsilon),
                )
            venv = getattr(venv, "venv", None)
        return None

    @classmethod
    def fit(cls, obs: np.ndarray, clip_obs: float, epsilon: float = 1e-8) -> ObsNorm:
        """The transform `VecNormalize` converges on over `obs`.

        What the regression stage normalizes with, and, because PPO inherits
        both the weights and these statistics, what its `obs_rms` is seeded
        from, so the cloned policy keeps seeing the distribution it was fit on.
        """
        sample = np.asarray(obs, dtype=np.float64)
        return cls(
            mean=sample.mean(axis=0),
            var=sample.var(axis=0),
            clip_obs=float(clip_obs),
            epsilon=epsilon,
        )


@dataclass(frozen=True)
class SnapshotPayload:
    """The `snapshot.pt` schema, stated once for both directions.

    `to_dict` and `from_dict` are inverses on purpose: the writer and the reader
    used to hold a copy of this schema each, free to drift.
    """

    policy_state_dict: dict[str, torch.Tensor]
    policy_class: str
    policy_kwargs: PolicyKwargs
    observation_space: Any
    action_space: Any
    env_config: RacingEnvConfig
    num_timesteps: int
    # All four are None when the trainer ran without `VecNormalize` in the stack.
    obs_mean: np.ndarray | None = None
    obs_var: np.ndarray | None = None
    clip_obs: float | None = None
    epsilon: float | None = None
    version: int = SNAPSHOT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policy_state_dict": self.policy_state_dict,
            "policy_class": self.policy_class,
            "policy_kwargs": self.policy_kwargs.as_kwargs(),
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "env_config": asdict(self.env_config),
            "num_timesteps": self.num_timesteps,
            "obs_mean": self.obs_mean,
            "obs_var": self.obs_var,
            "clip_obs": self.clip_obs,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SnapshotPayload:
        version = payload.get("version")
        if version != SNAPSHOT_VERSION:
            raise ValueError(
                f"snapshot version {version!r} is not supported "
                f"(this build reads version {SNAPSHOT_VERSION})"
            )
        return cls(
            policy_state_dict=payload["policy_state_dict"],
            policy_class=payload["policy_class"],
            policy_kwargs=PolicyKwargs(
                pi_arch=tuple(payload["policy_kwargs"]["net_arch"]["pi"]),
                vf_arch=tuple(payload["policy_kwargs"]["net_arch"]["vf"]),
                log_std_init=float(payload["policy_kwargs"]["log_std_init"]),
            ),
            observation_space=payload["observation_space"],
            action_space=payload["action_space"],
            env_config=env_config_from_dict(payload["env_config"]),
            num_timesteps=int(payload["num_timesteps"]),
            obs_mean=payload.get("obs_mean"),
            obs_var=payload.get("obs_var"),
            clip_obs=payload.get("clip_obs"),
            epsilon=payload.get("epsilon"),
            # Only SNAPSHOT_VERSION survives the gate above, which is the default.
        )


def save_snapshot(
    path: str | Path,
    policy,
    env_cfg: RacingEnvConfig,
    policy_kwargs: PolicyKwargs,
    num_timesteps: int,
    obs_norm: ObsNorm | None = None,
) -> Path:
    """Atomically publish a policy in the format every stage hands off in.

    `obs_norm` is the caller's: the supervised stages have no vec-env to dig it
    out of.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = SnapshotPayload(
        policy_state_dict={k: v.detach().to("cpu").clone() for k, v in policy.state_dict().items()},
        policy_class=f"{type(policy).__module__}.{type(policy).__qualname__}",
        policy_kwargs=policy_kwargs,
        observation_space=policy.observation_space,
        action_space=policy.action_space,
        env_config=env_cfg,
        num_timesteps=int(num_timesteps),
    )
    if obs_norm is not None:
        payload = replace(
            payload,
            obs_mean=np.asarray(obs_norm.mean, dtype=np.float64),
            obs_var=np.asarray(obs_norm.var, dtype=np.float64),
            clip_obs=float(obs_norm.clip_obs),
            epsilon=float(obs_norm.epsilon),
        )
    return _atomic(path, lambda tmp: torch.save(payload.to_dict(), tmp))


def publish_snapshot(
    stage_path: str | Path,
    deliverable: str | Path,
    policy,
    env_cfg: RacingEnvConfig,
    policy_kwargs: PolicyKwargs,
    num_timesteps: int,
    obs_norm: ObsNorm | None = None,
) -> Path:
    """Write a stage's policy to its own file and onto the run's deliverable.

    Every stage owes both: `training/<stage>.pt` is the lasting record a later
    stage cannot overwrite, `training/snapshot.pt` is whatever the run currently
    offers the deploy tools. The second write lands by rename, never by copy.
    """
    stage_path = save_snapshot(stage_path, policy, env_cfg, policy_kwargs, num_timesteps, obs_norm)
    # Copied rather than serialized twice, so the two files are bit-identical by
    # construction.
    _atomic(Path(deliverable), lambda tmp: shutil.copyfile(stage_path, tmp))
    return stage_path


@dataclass
class Snapshot:
    """A loaded snapshot, ready to drive an environment."""

    num_timesteps: int
    env_config: RacingEnvConfig
    policy: Any
    # None when the producing stage normalized nothing, in which case the
    # weights are a function of raw observations. `deploy/export.py` refuses
    # that case rather than shipping an input transform it cannot reconstruct.
    norm: ObsNorm | None

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Show the policy what it was trained on, see `ObsNorm`."""
        return obs.astype(np.float32) if self.norm is None else self.norm(obs)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """The action the trainer's inference path would take, from a raw observation."""
        action, _ = self.policy.predict(self.normalize_obs(obs), deterministic=deterministic)
        return action


def _resolve_policy_class(dotted: str):
    module_name, _, cls_name = dotted.rpartition(".")
    import importlib

    return getattr(importlib.import_module(module_name), cls_name)


def load_snapshot(path: str | Path) -> Snapshot:
    """Read a snapshot and rebuild a ready-to-run policy on the CPU.

    `weights_only=False` is required because the payload stores Gymnasium space
    objects; the file is produced by this project's own trainer.
    """
    payload = SnapshotPayload.from_dict(
        torch.load(Path(path), map_location="cpu", weights_only=False)
    )

    policy_cls = _resolve_policy_class(payload.policy_class)
    policy = policy_cls(
        observation_space=payload.observation_space,
        action_space=payload.action_space,
        lr_schedule=lambda _: 0.0,
        **payload.policy_kwargs.as_kwargs(),
    )
    policy.load_state_dict(payload.policy_state_dict)
    policy.to("cpu")
    policy.set_training_mode(False)

    norm = None
    if payload.obs_mean is not None and payload.obs_var is not None:
        norm = ObsNorm(
            mean=payload.obs_mean,
            var=payload.obs_var,
            # Statistics without a clip is not something this project writes;
            # "no clip" is the faithful reading of a payload recording none.
            clip_obs=float(payload.clip_obs) if payload.clip_obs is not None else float("inf"),
            epsilon=float(payload.epsilon or 1e-8),
        )

    return Snapshot(
        num_timesteps=payload.num_timesteps,
        env_config=payload.env_config,
        policy=policy,
        norm=norm,
    )
