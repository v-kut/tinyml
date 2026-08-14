"""One synthetic `ActorExport` builder, shared by every deploy-side suite."""

from __future__ import annotations

import numpy as np
import pytest

from tinyml_racing.deploy.artifact import ActorExport, DenseLayer


def make_export(
    arch: tuple[int, ...] = (6, 4, 2),
    *,
    activation: str = "tanh",
    seed: int = 0,
    head_scale: float | None = None,
    calib_rows: int = 32,
    reference_rows: int = 8,
) -> ActorExport:
    """A synthetic actor with the shape and scale of a real exported one.

    Deliberately not a trained policy: these suites are about arithmetic, and a
    fixture that needs 400k PPO steps to exist is a fixture nobody runs. The
    observation statistics are non-trivial on purpose, so folding normalization
    into layer 0 is actually exercised. `head_scale` over-scales the output
    layer, which is how an unbounded head reaches the action clip, with a tame
    head the clip is dead weight no test can miss.
    """
    rng = np.random.default_rng(seed)
    obs_mean = rng.normal(0.3, 0.2, arch[0]).astype(np.float32)
    obs_var = (0.5 + rng.random(arch[0])).astype(np.float32)
    scales = [float(arch[i]) ** -0.5 for i in range(len(arch) - 1)]
    if head_scale is not None:
        scales[-1] = head_scale
    layers = tuple(
        DenseLayer(
            w=rng.normal(0.0, scale, (arch[i], arch[i + 1])).astype(np.float32),
            b=rng.normal(0.0, 0.1, arch[i + 1]).astype(np.float32),
        )
        for i, scale in enumerate(scales)
    )
    calib = rng.normal(0.2, 0.4, (calib_rows, arch[0])).astype(np.float32)
    return ActorExport(
        layers=layers,
        activation=activation,
        obs_mean=obs_mean,
        obs_var=obs_var,
        clip_obs=10.0,
        epsilon=1e-8,
        calibration=calib,
        reference_in=calib[:reference_rows],
        reference_out=np.zeros((reference_rows, arch[-1]), np.float32),
        num_timesteps=400_000,
    )


@pytest.fixture
def export() -> ActorExport:
    return make_export()
