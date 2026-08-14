"""The supervised pipeline's contracts: episode splits, leakage, frame straddle, units.

Every assertion here defends a fix whose regression the suite could not see. The
pipeline passed unchanged through a batch of real behaviour changes (the split
stopped being a row permutation, `ObsNorm` stopped seeing held-out episodes, `Frame`
grew a pre-step `obs`, `reward_clip` changed units, `collect` started clipping
unconditionally) and no test noticed. These are the tests that would have.

Nothing trains: one epoch over a few hundred synthetic rows, and two `collect` passes
over one pinned layout.
"""

import inspect

import numpy as np
import pytest

from tinyml_racing.ml.config import (
    CLIP_OBS,
    PolicyKwargs,
    RacingEnvConfig,
    RegressionConfig,
)
from tinyml_racing.ml.env import RacingEnv
from tinyml_racing.ml.regression import fit
from tinyml_racing.ml.regression.dataset import Dataset, collect
from tinyml_racing.ml.rl.ppo import reward_clip
from tinyml_racing.ml.rollout import iter_rollout
from tinyml_racing.sim.car import CarParams

# Asymmetric, and tiny in both heads: the fits below only need a policy that
# constructs and converges quickly, but the two arches being different keeps
# `fit_policy`'s value-head path honest about which one it is sizing.
POLICY_KWARGS = PolicyKwargs(pi_arch=(8, 8), vf_arch=(12, 6))


def pinned(**kw) -> RacingEnvConfig:
    """One layout, always the same one. A fresh instance per call: `RacingEnv`
    writes `cfg.fixed_track_seed` onto the config it is handed and the dataclass
    is not frozen, so a shared constant would leak between tests.
    """
    return RacingEnvConfig(n_tracks=1, fixed_track_seed=42, **kw)


def dones_from_lengths(lengths: list[int]) -> np.ndarray:
    """A `Dataset.dones` mask for consecutive episodes of the given lengths."""
    dones = np.zeros(sum(lengths), dtype=bool)
    dones[np.cumsum(lengths) - 1] = True
    return dones


def episode_slices(lengths: list[int]) -> list[slice]:
    ends = np.cumsum(lengths)
    return [slice(int(end) - length, int(end)) for length, end in zip(lengths, ends, strict=True)]


# --------------------------------------------------------------------------
# _episode_split
# --------------------------------------------------------------------------

# Deliberately uneven, the way a real teacher pass is: an early crash and a full
# lap differ ~20-fold, and the row-counted target has to cope with that.
LENGTHS = [7, 3, 11, 5, 20, 2, 9, 4]


@pytest.mark.parametrize("seed", range(8))
def test_episode_split_never_cuts_an_episode_in_half(seed):
    """The regression: `rng.permutation(len(dataset))` over the rows.

    A `Dataset` is one 50 Hz trajectory, so a row held out that way leaves its 20 ms
    neighbours in the training set and `val_mse` measures interpolation between
    near-duplicates. The independent unit is the episode, with its own layout and
    spawn. A permutation of 61 rows almost surely straddles a boundary, so this fails
    for any seed the moment a row shuffle returns.
    """
    dones = dones_from_lengths(LENGTHS)
    train, val = fit._episode_split(dones, 0.3, np.random.default_rng(seed))

    is_val = np.zeros(len(dones), dtype=bool)
    is_val[val] = True
    for episode, rows in enumerate(episode_slices(LENGTHS)):
        side = is_val[rows]
        assert side.all() or not side.any(), (
            f"episode {episode} (rows {rows.start}:{rows.stop}) straddles the split"
        )
    # A partition, not two overlapping samples: `fit_policy` indexes both.
    assert len(train) + len(val) == len(dones)
    assert set(train.tolist()).isdisjoint(val.tolist())


def test_episode_split_leaves_training_rows_for_a_single_episode_dataset():
    """A one-episode dataset must still be fittable.

    "Whole episodes until `val_frac` of the rows are covered" reads as if it would
    take the only episode and hand the optimizer nothing. The guard is the
    `held >= len(starts) - 1` half of the break, exercised nowhere else, and a short
    debug run with one long episode is where it bites.
    """
    dones = dones_from_lengths([40])
    train, val = fit._episode_split(dones, 0.05, np.random.default_rng(0))
    assert len(train) == 40
    assert len(val) == 0


def test_episode_split_always_keeps_at_least_one_episode_for_training():
    """`val_frac = 1.0` must not empty the training set either.

    The same guard from the other side: the row target is unreachable, so only the
    episode counter stops the loop.
    """
    dones = dones_from_lengths([6, 9])
    train, val = fit._episode_split(dones, 1.0, np.random.default_rng(0))
    assert len(train) > 0
    assert len(val) > 0
    assert {len(train), len(val)} == {6, 9}


def test_episode_split_is_reproducible_from_its_generator():
    """`cfg.seed` pins the split, or two runs of one config report incomparable
    `val_mse` and a regression cannot be bisected. Also pins that the split consumes
    `rng` rather than a module default another caller could advance.
    """
    dones = dones_from_lengths(LENGTHS)
    first = fit._episode_split(dones, 0.3, np.random.default_rng(11))
    second = fit._episode_split(dones, 0.3, np.random.default_rng(11))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_episode_split_holds_out_the_requested_row_fraction():
    """`val_frac` is a fraction of rows, as its config comment says.

    On equal-length episodes the target is a whole number of them, so the count is
    exactly 200. The calibration case; the bound below separates rows from episodes.
    """
    dones = dones_from_lengths([10] * 100)
    train, val = fit._episode_split(dones, 0.2, np.random.default_rng(3))
    assert len(val) == 200
    assert len(train) == 800


@pytest.mark.parametrize("seed", range(8))
def test_episode_split_counts_rows_not_episodes(seed):
    """Holding out 30% of the episodes is not holding out 30% of the rows.

    Episode lengths vary ~20-fold between an early crash and a full lap, so over the
    2/40 alternation below, 30% of episodes is anywhere from 3% to 57% of rows. The
    row-counted rule has a closed-form bound: the loop stops at the first iteration
    where the running count reached `target`, so
    `target <= held rows < target + max_episode_length` for every seed.
    """
    lengths = [2 if i % 2 else 40 for i in range(60)]
    dones = dones_from_lengths(lengths)
    target = int(len(dones) * 0.3)
    _, val = fit._episode_split(dones, 0.3, np.random.default_rng(seed))
    assert target <= len(val) < target + max(lengths)


# --------------------------------------------------------------------------
# ObsNorm leakage
# --------------------------------------------------------------------------


def synthetic_dataset(lengths: list[int], obs_dim: int) -> Dataset:
    """Episodes constant within and far apart between, so any two subsets differ
    visibly in mean and variance. That separation is what makes a leaked statistic
    measurable rather than a rounding difference.
    """
    obs, actions = [], []
    for episode, length in enumerate(lengths):
        obs.append(np.full((length, obs_dim), 100.0 * episode, dtype=np.float32))
        actions.append(np.full((length, 2), 0.1 * episode, dtype=np.float32))
    n = sum(lengths)
    return Dataset(
        obs=np.concatenate(obs),
        actions=np.concatenate(actions),
        rewards=np.zeros(n, dtype=np.float32),
        dones=dones_from_lengths(lengths),
        episodes=len(lengths),
        crashes=0,
    )


def test_obs_norm_is_fitted_on_training_rows_only(monkeypatch):
    """Leakage regression: `ObsNorm.fit(dataset.obs, ...)` over every row.

    Normalization is part of the model, so statistics that saw the held-out episodes
    leak into the one number reporting overfitting. The split is stubbed to a known one
    so the expectation is an independently chosen row set, not a re-run of production.
    """
    lengths = [12, 12, 12, 12]
    dataset = synthetic_dataset(lengths, RacingEnvConfig().obs_dim)
    rows = [np.arange(s.start, s.stop) for s in episode_slices(lengths)]
    # Episodes 1 and 3 held out: whatever the fit sees, it is not everything.
    train_rows = np.concatenate([rows[0], rows[2]])
    val_rows = np.concatenate([rows[1], rows[3]])
    monkeypatch.setattr(fit, "_episode_split", lambda dones, frac, rng: (train_rows, val_rows))

    result = fit.fit_policy(
        dataset,
        pinned(),
        POLICY_KWARGS,
        RegressionConfig(epochs=1, batch_size=64, val_frac=0.5, seed=0),
    )

    train_mean = dataset.obs[train_rows].mean(axis=0, dtype=np.float64)
    train_var = dataset.obs[train_rows].var(axis=0, dtype=np.float64)
    np.testing.assert_allclose(result.obs_norm.mean, train_mean, rtol=1e-12)
    np.testing.assert_allclose(result.obs_norm.var, train_var, rtol=1e-12)
    # And the check that stops the above from passing vacuously: the whole
    # dataset's statistics are a long way from the training rows'.
    whole_mean = dataset.obs.mean(axis=0, dtype=np.float64)
    whole_var = dataset.obs.var(axis=0, dtype=np.float64)
    assert np.abs(whole_mean - train_mean).min() > 10.0
    assert np.abs(whole_var - train_var).min() > 100.0
    assert result.obs_norm.clip_obs == CLIP_OBS


# --------------------------------------------------------------------------
# Frame.obs
# --------------------------------------------------------------------------

ROLLOUT_SEED = 7
ROLLOUT_STEPS = 25


@pytest.fixture
def straddle():
    """One rollout, plus the observations `act` was handed.

    Copied at the call, because `iter_rollout` passes the very array it stores in
    `Frame.obs`, so an identity comparison would hold either way.
    """
    cfg = pinned(max_episode_steps=200)
    env = RacingEnv(config=cfg, seed=0)
    seen: list[np.ndarray] = []

    def act(obs):
        seen.append(np.array(obs, copy=True))
        return np.array([0.0, 0.3], dtype=np.float32)

    frames = list(iter_rollout(env, act, ROLLOUT_STEPS, seed=ROLLOUT_SEED))
    env.close()
    assert len(frames) == len(seen) >= 10, "the rollout must survive long enough to compare"
    return cfg, frames, seen


def test_frame_obs_is_the_observation_act_was_handed(straddle):
    """`Frame.obs` is pre-step, `state`/`scan` are post-step.

    `Frame` had no `obs`, so callers reconstructed the (observation, action) pair from
    the next frame, silently off by one. A regression filling `obs` from the step's
    return value would still be the right shape and dtype, which is why the second
    assertion compares against the frame's own successor.
    """
    _, frames, seen = straddle
    for i, frame in enumerate(frames):
        np.testing.assert_array_equal(frame.obs, seen[i])
    for i in range(len(frames) - 1):
        # `seen[i + 1]` *is* frame i's post-step observation: `iter_rollout`
        # hands it straight to the next `act`.
        assert not np.array_equal(frames[i].obs, seen[i + 1]), (
            f"frame {i}'s obs is the post-step observation, not the pre-step one"
        )


def test_first_frame_obs_is_the_reset_observation(straddle):
    """The pre-step observation of step 0 is what `reset(seed=...)` returned.

    From a second environment rather than `seen[0]`, since `reset(seed=S)` must
    reproduce track, spawn and observation regardless of history. Also the one
    assertion that catches `iter_rollout` dropping the reset observation.
    """
    cfg, frames, _ = straddle
    reference = RacingEnv(config=pinned(max_episode_steps=cfg.max_episode_steps), seed=0)
    expected, _ = reference.reset(seed=ROLLOUT_SEED)
    reference.close()
    np.testing.assert_array_equal(frames[0].obs, expected)


def test_frame_scan_is_the_post_step_scan(straddle):
    """The other half of the straddle, which makes `obs` load-bearing.

    The viewer draws the rays the car is looking at now, so `scan` is the post-step
    reading, the ray block the next `act` call receives. Pinning both sides leaves a
    change nowhere to hide.
    """
    cfg, frames, seen = straddle
    n_rays = cfg.lidar.n_rays
    for i in range(len(frames) - 1):
        np.testing.assert_array_equal(frames[i].scan, seen[i + 1][:n_rays])


# --------------------------------------------------------------------------
# reward_clip
# --------------------------------------------------------------------------


def test_reward_clip_takes_no_arguments_and_is_in_normalized_units():
    """The raw/normalized unit confusion, and the signature change that fixed it.

    `VecNormalize` clips `reward / sqrt(ret_rms.var + eps)`, so the bound counts
    standard deviations of the return accumulator. Sized from the off-track penalty in
    raw units it was an order of magnitude past anything normalized reward reaches, so
    clipping was off while `reward/clip_frac` looked configured. The upper bound says
    "normalized"; the argument count says `train._value_targets` still matches.
    """
    assert inspect.signature(reward_clip).parameters == {}
    clip = reward_clip()
    assert isinstance(clip, float)
    assert np.isfinite(clip)
    # Single digits to low tens. Above `1.0` it can admit the crash penalty at
    # its worst; below ~50 it can still bind on one.
    assert 1.0 < clip < 50.0
    raw_units = RacingEnvConfig().off_track_seconds / CarParams().dt
    assert clip * 10.0 < raw_units, "the bound is back in raw reward units"


# --------------------------------------------------------------------------
# Dataset.reward_accumulator / returns_to_go
# --------------------------------------------------------------------------

GAMMA = 0.5  # not a plausible discount, but it makes powers of two exact


def hand_dataset(rewards: list[float], dones: list[bool]) -> Dataset:
    n = len(rewards)
    return Dataset(
        obs=np.zeros((n, 1), dtype=np.float32),
        actions=np.zeros((n, 2), dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
        episodes=int(sum(dones)),
        crashes=0,
    )


def test_reward_accumulator_is_a_forward_discounted_sum_reset_at_each_done():
    """`ret_rms` is seeded from this, so the wrong recursion mis-scales every reward
    PPO sees.

    `VecNormalize._update_reward` accumulates forward, `R <- gamma*R + r` zeroed after
    a boundary, and divides rewards by that std. The reward-to-go looks
    interchangeable and is not, so the values are written out at `gamma = 1/2`:

        ep 1: r = 1, 2   ->  1,  1/2*1 + 2 = 2.5
        ep 2: r = 3, 4   ->  3,  1/2*3 + 4 = 5.5

    The reset is the part that matters: without it the third entry is
    `1/2*2.5 + 3 = 4.25` and every later one carries the previous episode.
    """
    dataset = hand_dataset([1.0, 2.0, 3.0, 4.0], [False, True, False, True])
    np.testing.assert_allclose(dataset.reward_accumulator(GAMMA), [1.0, 2.5, 3.0, 5.5], rtol=1e-12)


def test_returns_to_go_are_scaled_clipped_and_never_bootstrapped_across_a_done():
    """Value targets in PPO's units, with no credit crossing a boundary.

    `reward_scale` and `clip` are what `VecNormalize` does before PPO sees a reward,
    and the recursion runs backward per episode with no bootstrap. At `gamma = 1/2`,
    `reward_scale = 2`, `clip = 10` over `r = 1, 2, 3, 4`, boundary after the second:

        scaled:  1/2, 1, 3/2, 2
        ep 2:    out[3] = 2,           out[2] = 3/2 + 1/2*2   = 5/2
        ep 1:    out[1] = 1 (done),    out[0] = 1/2 + 1/2*1   = 1

    A bootstrap across the boundary gives `out[1] = 1 + 1/2*5/2 = 9/4`, and a critic
    fit on that values the next episode's spawn as if this car drove there.
    """
    dataset = hand_dataset([1.0, 2.0, 3.0, 4.0], [False, True, False, True])
    np.testing.assert_allclose(
        dataset.returns_to_go(GAMMA, 2.0, 10.0), [1.0, 1.0, 2.5, 2.0], rtol=1e-12
    )


def test_returns_to_go_of_one_episode_are_independent_of_the_next():
    """The same no-bootstrap contract as an independence property.

    Rewriting the second episode's rewards must leave the first episode's targets
    bit-identical. This survives a change to the recursion's shape; the literal above
    pins one arithmetic path.
    """
    dones = [False, True, False, True]
    first = hand_dataset([1.0, 2.0, 3.0, 4.0], dones).returns_to_go(GAMMA, 2.0, 10.0)
    second = hand_dataset([1.0, 2.0, -90.0, 60.0], dones).returns_to_go(GAMMA, 2.0, 10.0)
    np.testing.assert_array_equal(first[:2], second[:2])


def test_returns_to_go_clips_in_normalized_units_before_discounting():
    """`clip` is PPO's `clip_reward`, applied per reward, not to the return.

    Clipping the accumulated return would fit a value head past a bound PPO enforces,
    predicting returns the trainer can never collect. At `reward_scale = 1`,
    `clip = 3/2` over `r = 5, -5`:

        scaled:  3/2, -3/2  (both saturated)
        out[1] = -3/2,  out[0] = 3/2 + 1/2*(-3/2) = 3/4
    """
    dataset = hand_dataset([5.0, -5.0], [False, True])
    np.testing.assert_allclose(dataset.returns_to_go(GAMMA, 1.0, 1.5), [0.75, -1.5], rtol=1e-12)


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------


@pytest.mark.parametrize("noise_std", [0.0, 0.3])
def test_collect_clips_the_executed_action_with_or_without_noise(monkeypatch, noise_std):
    """The clip used to be inside `if noise_std > 0.0`.

    A noise-free pass, which is what `train._value_targets` collects for the critic,
    then handed the env whatever the teacher returned, with the action space upheld
    only because `step` and `control` happen to clamp too. The recorded rewards were
    then those of an action the dataset does not describe, and invisible from the
    `Dataset`, since the label stays the teacher's. So the executed action is read
    where it is executed.
    """
    executed: list[np.ndarray] = []
    real_step = RacingEnv.step

    def recording_step(self, action):
        executed.append(np.asarray(action, dtype=np.float64).copy())
        return real_step(self, action)

    monkeypatch.setattr(RacingEnv, "step", recording_step)

    # Well outside the action space in both channels and both signs, so the clip
    # has to bind on every step no matter what the noise does.
    def teacher(_env, _obs):
        return np.array([3.0, -2.5], dtype=np.float32)

    dataset = collect(teacher, pinned(), 40, noise_std=noise_std, seed=0)

    assert len(executed) == 40
    demanded = np.stack(executed)
    assert (demanded >= -1.0).all() and (demanded <= 1.0).all(), (
        f"collect executed an action outside [-1, 1]: {demanded.min()} .. {demanded.max()}"
    )
    # The clip bound, so the bound above is not vacuous.
    np.testing.assert_allclose(demanded[:, 0], 1.0)
    np.testing.assert_allclose(demanded[:, 1], -1.0)
    # ...and the DART label is still the teacher's own, unclipped: clipping the
    # recording too would train the student to demand the rail rather than to
    # reproduce the expert.
    np.testing.assert_allclose(dataset.actions, np.array([[3.0, -2.5]] * 40, dtype=np.float32))
