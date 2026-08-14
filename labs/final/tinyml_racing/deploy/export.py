"""Export the trained actor out of PyTorch as plain arrays: the seam between
training and deployment.

Only this module imports torch and Stable-Baselines3; every reader of the `.npz`
goes through `deploy/artifact.py`, which is why the schema lives there. What
crosses: the dense stack, `VecNormalize` stats, calibration and reference pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from tinyml_racing import progress
from tinyml_racing.deploy.artifact import ActorExport, DenseLayer
from tinyml_racing.deploy.quantize import ACTIVATION
from tinyml_racing.ml.env import RacingEnv
from tinyml_racing.ml.rollout import iter_rollout
from tinyml_racing.ml.snapshot import load_snapshot
from tinyml_racing.utils import Run

logger = logging.getLogger(__name__)


def extract_actor(policy) -> list[DenseLayer]:
    """Pull the actor's dense stack out of an SB3 `ActorCriticPolicy`.

    Layers come back in evaluation order, transposed to (in_features,
    out_features), which is what every consumer downstream expects; torch stores
    `nn.Linear` the other way round. Anything between the Linears that is not
    the activation the kernel implements is refused here rather than three
    stages later: a Flatten or a Dropout would otherwise be exported as if the
    board could run it.
    """
    from torch import nn

    layers: list[DenseLayer] = []
    modules = [*policy.mlp_extractor.policy_net, policy.action_net]

    for module in modules:
        if isinstance(module, nn.Linear):
            layers.append(
                DenseLayer(
                    w=module.weight.detach().cpu().numpy().T,
                    b=module.bias.detach().cpu().numpy(),
                )
            )
        elif str(type(module).__name__).lower() != ACTIVATION:
            raise ValueError(
                f"{type(module).__name__} in the actor trunk is not what the deployed "
                f"kernel implements; it evaluates {ACTIVATION!r} between the Linears"
            )

    if not layers:
        raise ValueError("no Linear layers found in the policy trunk")
    return layers


def collect_calibration(snapshot, n_obs: int, seed: int) -> np.ndarray:
    """Gather observations the policy actually encounters while driving.

    Post-training quantization derives its int8 scales from this set, so it has to
    come from on-policy rollouts: ranges fitted on states the deployed car never
    visits show up as biased steering rather than as an obvious failure.

    Every sample is a `Frame.obs`, the array `iter_rollout` handed to the policy, and
    `iter_rollout` owns the seeded reset. Resetting or re-observing here would
    calibrate on a sweep the policy never acted on.
    """
    env = RacingEnv(config=snapshot.env_config, seed=seed)
    collected: list[np.ndarray] = []
    episode = 0

    # The episode cap bounds a policy that crashes on its first step: each pass
    # would then contribute one observation and the quota would never fill.
    with progress.stage("calibrate", n_obs) as bar:
        while len(collected) < n_obs and episode <= 100:
            collected.extend(
                frame.obs
                for frame in iter_rollout(
                    env, snapshot.act, max_steps=n_obs - len(collected), seed=seed + episode
                )
            )
            episode += 1
            bar.set(len(collected), note=f"{episode} episodes")

    env.close()
    # Loud rather than short: every activation scale is fitted on this set, so a
    # quietly truncated calibration ships wrong scales to the board.
    if len(collected) < n_obs:
        raise RuntimeError(
            f"calibration reached only {len(collected)} of {n_obs} observations in "
            f"{episode} episodes; the policy cannot survive long enough to calibrate "
            "on. Train it further or lower --n-calibration."
        )
    return np.asarray(collected[:n_obs], dtype=np.float32)


def export_actor(
    run: Run,
    n_calibration: int = 2048,
    n_reference: int = 256,
    seed: int = 7,
) -> Path:
    snapshot = load_snapshot(run.snapshot)
    # Without the `VecNormalize` statistics the weights describe a function of
    # normalized observations that nothing downstream can reconstruct; failing
    # here names the run instead of dying in the writer on `np.asarray(None)`.
    if snapshot.norm is None:
        raise ValueError(
            f"{run.snapshot} carries no VecNormalize observation statistics, so the "
            "exported actor would be meaningless (see ml/snapshot.py). Retrain with "
            "VecNormalize in the vec-env stack."
        )
    layers = extract_actor(snapshot.policy)

    calibration = collect_calibration(snapshot, n_calibration, seed)
    reference_in = calibration[:n_reference]
    # The deterministic action mean, clipped to the action space exactly
    # as `policy.predict` does, the device clips too, so this is what
    # the compiled model has to reproduce, end to end.
    reference_out = np.stack([snapshot.act(o) for o in reference_in]).astype(np.float32)

    export = ActorExport(
        layers=tuple(layers),
        activation=ACTIVATION,
        obs_mean=np.asarray(snapshot.norm.mean, dtype=np.float32),
        obs_var=np.asarray(snapshot.norm.var, dtype=np.float32),
        clip_obs=float(snapshot.norm.clip_obs),
        epsilon=float(snapshot.norm.epsilon),
        calibration=calibration,
        reference_in=reference_in,
        reference_out=reference_out,
        num_timesteps=int(snapshot.num_timesteps),
    )
    out = export.save(run.actor_npz)

    logger.info(
        "actor %s (%s), %d params @ %d steps",
        export.shape(),
        export.activation,
        export.n_params,
        export.num_timesteps,
    )
    logger.info("calibration %s, reference %s -> %s", calibration.shape, reference_in.shape, out)
    return out
