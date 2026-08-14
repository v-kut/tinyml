"""Supervised fit of an SB3 policy to a teacher's actions.

Not a stand-in for PPO's network but a real `ActorCriticPolicy` at the run's
architecture, so the result is a state dict PPO loads whole. The actor is fit to
the teacher's actions, the critic to its returns in `VecNormalize` units, and
`log_std` stays out of the optimizer, exploration is the next trainer's choice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter

from tinyml_racing import progress
from tinyml_racing.ml.config import CLIP_OBS, PolicyKwargs, RacingEnvConfig, RegressionConfig
from tinyml_racing.ml.env import RacingEnv
from tinyml_racing.ml.regression.dataset import Dataset
from tinyml_racing.ml.snapshot import ObsNorm

logger = logging.getLogger(__name__)

# Weight on the critic's loss, matching PPO's own `vf_coef`: the heads share no
# parameters, so this only balances how much of the step each gets, and PPO's
# value means the critic arrives trained the way PPO would keep training it.
VF_COEF = 0.5


@dataclass(frozen=True)
class FitResult:
    """A fitted policy and the numbers that say whether it is worth keeping."""

    policy: ActorCriticPolicy
    obs_norm: ObsNorm
    train_mse: float
    # On held-out *episodes*, layout and spawn included. Cloning 200k correlated
    # states overfits readily and this gap is the only thing that says so, but
    # only because the split is by episode, never by row (see `_episode_split`).
    val_mse: float


def build_policy(
    env_cfg: RacingEnvConfig, policy_kwargs: PolicyKwargs, seed: int
) -> ActorCriticPolicy:
    """An untrained policy of exactly the class and shape PPO builds.

    The spaces come from a real `RacingEnv`, not from two `Box` literals: `env.py`
    owns the observation layout. Constructing one is cheap, because the track
    pool is built lazily on a first `reset` this never calls.
    """
    probe = RacingEnv(config=env_cfg)
    try:
        torch.manual_seed(seed)
        return ActorCriticPolicy(
            observation_space=probe.observation_space,
            action_space=probe.action_space,
            # No schedule: this stage runs its own optimizer, and the one SB3
            # attaches here is replaced by PPO when it takes the weights over.
            lr_schedule=lambda _: 0.0,
            **policy_kwargs.as_kwargs(),
        )
    finally:
        probe.close()


def _features(policy: ActorCriticPolicy, obs: torch.Tensor) -> torch.Tensor:
    """Trunk input for either head.

    `extract_features` returns a `(pi, vf)` pair only when the heads own separate
    extractors; `build_policy` keeps SB3's shared one, so it is always a single
    tensor, narrowed here once instead of at both call sites.
    """
    features = policy.extract_features(obs)
    if not isinstance(features, torch.Tensor):
        raise TypeError("this stage requires a policy with a shared features extractor")
    return features


def _action_mean(policy: ActorCriticPolicy, obs: torch.Tensor) -> torch.Tensor:
    """The deterministic action, the actor's half of the trunk only.

    `policy.forward` samples and returns a log probability with it; the mean is
    what the deployed network computes. The value head is not computed alongside
    because the two heads are fit on different rows here.
    """
    latent_pi = policy.mlp_extractor.forward_actor(_features(policy, obs))
    return cast("torch.Tensor", policy.action_net(latent_pi))


def _state_value(policy: ActorCriticPolicy, obs: torch.Tensor) -> torch.Tensor:
    """The critic's estimate, flattened to the shape of a target vector."""
    latent_vf = policy.mlp_extractor.forward_critic(_features(policy, obs))
    return cast("torch.Tensor", policy.value_net(latent_vf)).flatten()


def _episode_split(
    dones: np.ndarray, val_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices for (train, validation), holding out whole episodes.

    A uniform permutation of rows is not a split on a 50 Hz trajectory: a held-out
    frame's 20 ms neighbours sit in the training set centimetres away. Whole
    episodes move over in `rng` order until `val_frac` of the *rows* are covered.
    """
    bounds = np.flatnonzero(dones) + 1  # one past each episode's last row
    starts = np.concatenate(([0], bounds))[:-1]
    target = int(len(dones) * val_frac)

    is_val = np.zeros(len(dones), dtype=bool)
    rows = 0
    # `held` is the loop index because it counts episodes already taken: at the
    # top of iteration k exactly k of them are in the validation set.
    for held, episode in enumerate(rng.permutation(len(starts))):
        # Never every episode: the fit needs rows, and `_action_mse` already
        # reports nan for an empty validation set.
        if rows >= target or held >= len(starts) - 1:
            break
        is_val[starts[episode] : bounds[episode]] = True
        rows += int(bounds[episode] - starts[episode])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def fit_policy(
    dataset: Dataset,
    env_cfg: RacingEnvConfig,
    policy_kwargs: PolicyKwargs,
    cfg: RegressionConfig,
    *,
    value_rows: tuple[np.ndarray, np.ndarray] | None = None,
    label: str = "regression",
    tb_dir: Path | None = None,
) -> FitResult:
    """Fit a fresh policy to `dataset`, returning it with its normalization.

    `value_rows` is `(raw observations, returns)` for the critic, from its own
    noise-free pass because `dataset`'s rewards value the DART-perturbed driver.
    None trains the actor alone; the per-epoch curve goes to TensorBoard.
    """
    # Split first and once, and fit the normalization from the training rows only:
    # it is part of the model, so statistics that saw the held-out episodes leak
    # into the one number that reports overfitting.
    rng = np.random.default_rng(cfg.seed)
    train_rows, val_rows = _episode_split(dataset.dones, cfg.val_frac, rng)
    obs_norm = ObsNorm.fit(dataset.obs[train_rows], CLIP_OBS)

    policy = build_policy(env_cfg, policy_kwargs, cfg.seed)
    policy.set_training_mode(True)

    obs = torch.as_tensor(obs_norm(dataset.obs))
    actions = torch.as_tensor(dataset.actions, dtype=torch.float32)
    train_idx = torch.as_tensor(train_rows)
    val_idx = torch.as_tensor(val_rows)
    # Normalized with the actor's statistics: the critic shares the trunk's
    # input space, and a value head fit on a differently-scaled observation is
    # a value head PPO cannot use.
    critic_obs = None if value_rows is None else torch.as_tensor(obs_norm(value_rows[0]))
    critic_targets = (
        None if value_rows is None else torch.as_tensor(value_rows[1], dtype=torch.float32)
    )

    # `log_std` excluded by name, not by having no gradient: Adam skipping
    # `grad is None` would make that the optimizer's decision, not this stage's.
    trained = [p for name, p in policy.named_parameters() if name != "log_std"]
    optimizer = torch.optim.Adam(trained, lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    no_rows = torch.empty(0, dtype=torch.long)
    train_mse = float("nan")
    val_mse = float("nan")
    writer = SummaryWriter(str(tb_dir)) if tb_dir is not None else None

    try:
        with progress.stage(f"{label} fit", cfg.epochs) as bar:
            for epoch in range(cfg.epochs):
                shuffled = train_idx[torch.randperm(len(train_idx))]
                batches = [
                    shuffled[start : start + cfg.batch_size]
                    for start in range(0, len(shuffled), cfg.batch_size)
                ]
                # The critic's rows are a different trajectory, cut into as many
                # pieces as the actor has batches: every row is still seen once
                # per epoch, and the heads share no parameters.
                value_batches = (
                    list(torch.tensor_split(torch.randperm(len(critic_targets)), len(batches)))
                    if critic_targets is not None
                    else [no_rows] * len(batches)
                )
                total = 0.0
                for batch, value_batch in zip(batches, value_batches, strict=True):
                    action_loss = loss_fn(_action_mean(policy, obs[batch]), actions[batch])
                    loss = action_loss
                    if critic_obs is not None and critic_targets is not None and len(value_batch):
                        predicted = _state_value(policy, critic_obs[value_batch])
                        loss = loss + VF_COEF * loss_fn(predicted, critic_targets[value_batch])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # The action MSE alone, weighted by batch size: the critic
                    # term is in reward units and would make this incomparable
                    # between a pretraining run and a distillation one.
                    total += action_loss.detach().item() * len(batch)

                train_mse = total / max(len(shuffled), 1)
                val_mse = _action_mse(policy, obs, actions, val_idx)
                if writer is not None:
                    writer.add_scalar("regression/train_mse", train_mse, epoch)
                    writer.add_scalar("regression/val_mse", val_mse, epoch)
                bar.advance(note=f"train {train_mse:.5f} · val {val_mse:.5f}")
                # An epoch boundary is the only safe stop: the policy is coherent
                # and the two numbers reported are the kept weights'.
                if bar.skipped:
                    break
    finally:
        if writer is not None:
            writer.close()

    policy.set_training_mode(False)
    return FitResult(policy=policy, obs_norm=obs_norm, train_mse=train_mse, val_mse=val_mse)


@torch.no_grad()
def _action_mse(
    policy: ActorCriticPolicy, obs: torch.Tensor, actions: torch.Tensor, idx: torch.Tensor
) -> float:
    if len(idx) == 0:
        return float("nan")
    policy.set_training_mode(False)
    mean_actions = _action_mean(policy, obs[idx])
    policy.set_training_mode(True)
    return float(nn.functional.mse_loss(mean_actions, actions[idx]))
