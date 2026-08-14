"""The `actor.npz` schema: the one definition of what crosses the training ->
deployment boundary.

Separate from `deploy/export.py`, which writes it: the writer needs torch and
SB3, and no reader may. `load` refuses any `version` it does not know, a
drifted scale convention quantizes, compiles and flashes, and only the car sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["EXPORT_VERSION", "ActorExport", "DenseLayer"]

# Bump on any change to key names, dtypes or the meaning of a field: folding
# normalization differently, reordering `w{i}`, switching a scale convention. Adding a
# field readers can default is not a bump.
EXPORT_VERSION = 1


@dataclass(frozen=True)
class DenseLayer:
    """One fully-connected layer.

    `w` is (in_features, out_features), transposed relative to how torch
    stores `nn.Linear`, because every consumer downstream thinks in "column j is
    output j". The transpose happens once, in `export.extract_actor`.
    """

    w: np.ndarray  # (n_in, n_out) float32
    b: np.ndarray  # (n_out,) float32


@dataclass(frozen=True)
class ActorExport:
    """The exported actor plus everything needed to reproduce its inference path.

    The `VecNormalize` statistics are not optional decoration: the weights were
    trained on normalized observations, so without them the model is meaningless
    (see `ml/snapshot.py`). `quantize.py` folds them into the first layer.
    """

    layers: tuple[DenseLayer, ...]
    activation: str
    obs_mean: np.ndarray  # (obs_dim,) float32, VecNormalize running mean
    obs_var: np.ndarray  # (obs_dim,) float32, VecNormalize running variance
    clip_obs: float
    epsilon: float
    calibration: np.ndarray  # (n_calib, obs_dim) float32, on-policy observations
    reference_in: np.ndarray  # (n_ref, obs_dim) float32
    reference_out: np.ndarray  # (n_ref, n_act) float32
    num_timesteps: int
    version: int = EXPORT_VERSION

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def obs_dim(self) -> int:
        return int(self.layers[0].w.shape[0])

    @property
    def n_params(self) -> int:
        return sum(layer.w.size + layer.b.size for layer in self.layers)

    @property
    def float_bytes(self) -> int:
        """Flash an unquantized actor would occupy as a C header of floats."""
        return 4 * self.n_params

    def shape(self) -> str:
        """`64 -> 32 -> 16 -> 2`, for the log line and the report."""
        widths = [self.obs_dim] + [int(layer.w.shape[1]) for layer in self.layers]
        return " -> ".join(str(w) for w in widths)

    def save(self, path: str | Path) -> Path:
        """Write `actor.npz`: flat keys, 0-d numpy scalars for the metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Mixed on purpose: arrays for the tensors, 0-d numpy scalars for the
        # metadata, which is the layout the flat key names imply.
        payload: dict[str, Any] = {
            "version": np.int64(self.version),
            "n_layers": np.int64(self.n_layers),
            "activation": np.array(self.activation),
            "obs_mean": np.asarray(self.obs_mean, dtype=np.float32),
            "obs_var": np.asarray(self.obs_var, dtype=np.float32),
            "clip_obs": np.float32(self.clip_obs),
            "epsilon": np.float32(self.epsilon),
            "num_timesteps": np.int64(self.num_timesteps),
            "calibration": np.asarray(self.calibration, dtype=np.float32),
            "reference_in": np.asarray(self.reference_in, dtype=np.float32),
            "reference_out": np.asarray(self.reference_out, dtype=np.float32),
        }
        for i, layer in enumerate(self.layers):
            payload[f"w{i}"] = np.asarray(layer.w, dtype=np.float32)
            payload[f"b{i}"] = np.asarray(layer.b, dtype=np.float32)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ActorExport:
        """Read `actor.npz`, coercing the 0-d arrays `np.load` returns for scalars.

        An unreadable version is fatal here rather than downstream: the next stage
        quantizes whatever it is handed and the stage after that flashes it.
        """
        with np.load(path, allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files}
        stamp = payload.get("version")
        version = int(stamp) if stamp is not None else None
        if version != EXPORT_VERSION:
            raise ValueError(
                f"{path}: actor export version {version!r} is not supported "
                f"(this build reads version {EXPORT_VERSION}); re-export the run "
                "(`tinyml-build` re-exports unless --reuse-export is passed)"
            )
        return cls(
            layers=tuple(
                DenseLayer(w=payload[f"w{i}"], b=payload[f"b{i}"])
                for i in range(int(payload["n_layers"]))
            ),
            activation=str(payload["activation"]),
            obs_mean=payload["obs_mean"],
            obs_var=payload["obs_var"],
            clip_obs=float(payload["clip_obs"]),
            epsilon=float(payload["epsilon"]),
            calibration=payload["calibration"],
            reference_in=payload["reference_in"],
            reference_out=payload["reference_out"],
            num_timesteps=int(payload["num_timesteps"]),
            # Only EXPORT_VERSION survives the gate above, which is the default.
        )
