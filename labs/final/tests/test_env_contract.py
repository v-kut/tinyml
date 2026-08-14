import math
from dataclasses import replace

import numpy as np
import pytest

from tinyml_racing.ml.config import EVAL_SEED_RANGE, PPOConfig, RacingEnvConfig
from tinyml_racing.ml.env import RacingEnv
from tinyml_racing.ml.rl.callbacks import _STEP_MEANS
from tinyml_racing.sim.car import CarParams, proprioception
from tinyml_racing.sim.lidar import ray_angles

CFG = RacingEnvConfig(n_tracks=2, max_episode_steps=50)


def pinned(**kw) -> RacingEnvConfig:
    """One layout, always the same one, on a config instance of its own.

    `RacingEnvConfig` is unfrozen and `CFG` is module-level, so anything that
    assigns to the config it was handed (as `watch.py` does to pin a layout)
    writes through to every later user. A fresh one per test costs nothing.
    """
    return RacingEnvConfig(n_tracks=1, fixed_track_seed=42, **kw)


@pytest.fixture
def env():
    e = RacingEnv(config=CFG, seed=0)
    # None of the env's seeds reach `action_space`, which carries its own RNG.
    # Unseeded, a sampled action that trips a NaN is a one-off nobody can repeat.
    e.action_space.seed(0)
    yield e
    e.close()


@pytest.mark.parametrize("dt", [0.05, 0.02, 0.01, 0.005])
def test_credit_windows_are_durations_not_step_counts(dt):
    """`gamma` and `gae_lambda` must mean the same *time* at any control rate.

    As literals they did not: 0.995 and 0.95 were 10 s and 1 s at dt 0.05, and 4 s
    and 0.4 s by dt 0.02, shorter than the 3.6 s this car needs to brake from top
    speed. This fails if either goes back to a literal.
    """
    cfg = PPOConfig()
    kwargs = cfg.as_kwargs(dt)
    horizon = -dt / math.log(kwargs["gamma"])
    window = dt / (1.0 - kwargs["gae_lambda"])
    assert horizon == pytest.approx(cfg.discount_s, rel=1e-6)
    assert window == pytest.approx(cfg.credit_s, rel=1e-6)
    # The value horizon has to outlast the manoeuvre it is meant to teach.
    assert horizon > CarParams().top_speed / CarParams().brake_accel_max


def test_credit_windows_accept_per_step_overrides():
    kwargs = PPOConfig(gamma=0.9, gae_lambda=0.8).as_kwargs(0.02)
    assert (kwargs["gamma"], kwargs["gae_lambda"]) == (0.9, 0.8)
    assert "discount_s" not in kwargs and "credit_s" not in kwargs


def test_reset_matches_declared_observation_space(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (CFG.obs_dim,)
    assert obs.dtype == np.float32
    assert env.observation_space.shape == obs.shape
    # Gymnasium's membership test pins dtype as well as shape, and rejects NaN,
    # which compares false against both bounds.
    assert env.observation_space.contains(obs)
    # `reset` has nothing to report yet; the keys callers read are enumerated in
    # `test_step_info_carries_the_metrics_callbacks_log`.
    assert isinstance(info, dict), "Gymnasium's reset contract is (obs, info)"


def test_observation_is_always_finite(env):
    # VecNormalize propagates one NaN into its running statistics and poisons
    # every later observation.
    obs, _ = env.reset(seed=1)
    assert np.isfinite(obs).all()
    for _ in range(30):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all()
        if terminated or truncated:
            obs, _ = env.reset()


def test_the_scan_is_normalized_and_is_the_observation_verbatim(env):
    """The ray block of the observation is the scan `step` recorded, not a
    rescaling and not a second sweep.

    The viewer draws `last_scan` to show what the policy reacted to, so the two
    must be the same numbers. Calling `_observe()` again cannot check it: that
    assigns `last_scan` first, and the noisy sensor makes a second cast a
    different sample anyway.
    """
    env.reset(seed=2)
    obs, *_ = env.step(np.zeros(2, dtype=np.float32))
    scan = env.last_scan
    assert scan.shape == (CFG.lidar.n_rays,)
    assert scan.dtype == np.float32, "a float64 scan would round on its way into the obs"
    # Normalization happens in the sensor, so no scaling survives in between.
    assert (scan >= 0.0).all() and (scan <= 1.0).all()
    np.testing.assert_array_equal(obs[: CFG.lidar.n_rays], scan)


def test_the_scan_difference_is_the_change_between_consecutive_sweeps(env):
    """The second ray block is `scan_t - scan_(t-1)`, in that order.

    Pinned against the sweeps `step` recorded, since a fresh cast is a different
    sample under noise. Reading `last_scan` across two steps also catches the sign
    convention: `prev - now` is as plausible to write and inverts every closure
    rate the block carries.
    """
    n = CFG.lidar.n_rays
    env.reset(seed=3)
    action = np.array([0.1, 0.8], dtype=np.float32)
    env.step(action)
    # Deliberately not copied: `_observe` must rebind `last_scan`. Writing into
    # the old array in place would alias `now` and `prev` and zero the difference.
    prev = env.last_scan
    obs, *_ = env.step(action)
    now = env.last_scan
    delta = obs[n : 2 * n]
    assert np.abs(delta).max() > 0.0, "a moving car changes what it sees; 0 == 0 proves nothing"
    np.testing.assert_array_equal(delta, now - prev)


def test_the_noisy_evaluation_variant_holds_out_layouts_and_keeps_the_detector():
    """The only condition the differenced-sweep block can be measured in.

    A clean sensor is the right default for comparing driving, and the one
    condition where that block can only cost width, since the dropouts it rides
    over are gone. So the noisy variant must be a sensor change and nothing else:
    same seed range, same cleared `fixed_track_seed`, and a scan that really
    differs on the same layout, spawn and actions.
    """
    cfg = RacingEnvConfig(n_tracks=1, fixed_track_seed=42, max_episode_steps=20)
    clean = cfg.evaluation_variant()
    noisy = cfg.evaluation_variant(clean_sensor=False)

    assert cfg.lidar.dropout_prob > 0.0, "a default run trains against a noisy detector"
    assert (clean.lidar.noise_std, clean.lidar.dropout_prob) == (0.0, 0.0)
    assert noisy.lidar == cfg.lidar, "the noisy variant is the training sensor, unchanged"
    # Everything that makes a score held-out is shared with the clean variant. A
    # noisy number over different layouts would measure the layouts.
    assert replace(noisy, lidar=clean.lidar) == clean
    assert noisy.fixed_track_seed is None
    assert noisy.track_seed_range == EVAL_SEED_RANGE

    # And it reaches the env, not only the config: same layout, spawn and
    # actions, so only the detector can differ.
    scans = []
    for variant in (clean, noisy):
        e = RacingEnv(config=replace(variant, n_tracks=1, fixed_track_seed=7), seed=7)
        e.reset(seed=7)
        for _ in range(10):
            e.step(np.array([0.0, 0.5], dtype=np.float32))
        assert e.last_scan is not None
        scans.append(e.last_scan.copy())
        e.close()
    assert not np.array_equal(scans[0], scans[1]), (
        "a noisy evaluation must actually see a noisy sweep"
    )


def test_the_scan_difference_never_straddles_a_reset(env):
    """Zero on an episode's first frame, whether or not one ran before it.

    A reset teleports the car, so every bearing changes for reasons no closure
    rate describes. The mid-run case is the one that regresses: the sweep history
    outlives its episode unless `reset` clears it, and `__init__` alone makes the
    first episode look correct.
    """
    n = CFG.lidar.n_rays
    obs, _ = env.reset(seed=4)
    assert not np.any(obs[n : 2 * n]), "first episode: nothing to difference against"
    for _ in range(5):
        env.step(np.array([0.1, 0.8], dtype=np.float32))
    obs, _ = env.reset(seed=5)
    assert not np.any(obs[n : 2 * n]), "a reset must not difference across the boundary"
    assert obs[-1] == 0.0, "nor carry the previous episode's throttle"


@pytest.mark.parametrize(
    ("commanded", "reported"), [(0.8, 0.8), (-0.6, -0.6), (1.5, 1.0), (-1.5, -1.0)]
)
def test_the_last_channel_reports_the_throttle_that_produced_the_state(commanded, reported):
    """The observation describes a state and reports the pedal that got the car
    there: post-clip, because that is what drove it, and signed, because braking
    is the half nothing else observes.

    `proprioception` carries the achieved steering angle, so the steering command
    is already visible through the slew limit. Throttle has no such channel, which
    is why this block exists.
    """
    e = RacingEnv(config=pinned(), seed=0)
    obs, _ = e.reset(seed=6)
    assert obs[-1] == 0.0
    for _ in range(3):
        obs, _, terminated, truncated, _ = e.step(np.array([0.2, commanded], dtype=np.float32))
        assert obs[-1] == pytest.approx(reported), "the pedal the car was actually driven with"
        if terminated or truncated:
            break
    e.close()


# Every shape the composition knobs admit, blocks off as well as on. The point is
# not that each runs but that `obs_dim` and `_observe` agree about all of them.
COMPOSITIONS = [
    {},
    {"scan_history": 1},
    {"scan_history": 1, "throttle_history": 0},
    {"scan_history": 4},
    {"cross_track_history": 1},
    {"throttle_history": 3, "cross_track_history": 2},
    {"scan_history": 1, "throttle_history": 0, "cross_track_history": 1},
]


@pytest.mark.parametrize(
    "composition",
    COMPOSITIONS,
    ids=["-".join(f"{k}={v}" for k, v in c.items()) or "defaults" for c in COMPOSITIONS],
)
def test_obs_dim_is_the_width_observe_actually_builds(composition):
    """`obs_dim` is a claim, and everything downstream believes it.

    `observation_space`, the dataset buffers, `VecNormalize`'s statistics and the
    exported first layer are all sized from it. A miscounted width is not a
    narrower observation but a network that disagrees with its own run, surfacing
    as a shape error inside SB3 or an exported net fed a misaligned vector.
    """
    cfg = pinned(**composition)
    e = RacingEnv(config=cfg, seed=0)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (cfg.obs_dim,)
    assert e.observation_space.contains(obs)
    for _ in range(4):
        obs, _, terminated, truncated, _ = e.step(np.array([0.1, 0.9], dtype=np.float32))
        assert obs.shape == (cfg.obs_dim,)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            obs, _ = e.reset()
    e.close()


def test_switching_a_block_off_removes_it_rather_than_zeroing_it():
    """Off means gone, not held at a constant zero.

    A zeroed block costs the same flash and multiplies on the device, and
    `VecNormalize` would divide it by a variance no observation built. The bare
    configuration is exactly the sweep plus four proprioceptive scalars.
    """
    cfg = pinned(scan_history=1, throttle_history=0, cross_track_history=0)
    n = cfg.lidar.n_rays
    assert cfg.obs_dim == n + 4
    e = RacingEnv(config=cfg, seed=0)
    e.reset(seed=0)
    obs, *_ = e.step(np.array([0.3, 0.9], dtype=np.float32))
    np.testing.assert_array_equal(obs[:n], e.last_scan)
    np.testing.assert_allclose(obs[n:], proprioception(e.state, e.params), rtol=1e-6)
    e.close()


def test_stacking_deeper_than_two_gives_successive_differences():
    """`scan_history=k` is one sweep then `k-1` differences, newest pair first.

    Not `k` raw sweeps, which would share one int8 `absmax` and spend its range on
    the part that did not move; and not `k-1` differences against the newest
    sweep, which would make the second block the sum of the first two.
    """
    k = 3
    cfg = pinned(scan_history=k)
    n = cfg.lidar.n_rays
    e = RacingEnv(config=cfg, seed=0)
    e.reset(seed=7)
    action = np.array([0.1, 0.8], dtype=np.float32)
    sweeps = []
    for _ in range(k):
        obs, *_ = e.step(action)
        assert e.last_scan is not None
        sweeps.append(e.last_scan)
    np.testing.assert_array_equal(obs[:n], sweeps[-1])
    np.testing.assert_array_equal(obs[n : 2 * n], sweeps[-1] - sweeps[-2])
    np.testing.assert_array_equal(obs[2 * n : 3 * n], sweeps[-2] - sweeps[-3])
    e.close()


def test_the_cross_track_block_is_the_signed_offset_the_shaping_grades():
    """One block, two contracts: the magnitude the reward shapes on, and the sign
    that says which way to steer back.

    The magnitude must agree with `info["cross_track"]`, the quantity
    `_cross_track_potential` is a function of; if they disagree the policy is
    still graded on something it cannot see. The sign must be a real sign:
    `project` once returned an unsigned residual, and a non-negative block would
    look plausible while carrying nothing new.
    """
    e = RacingEnv(config=pinned(cross_track_history=1), seed=0)
    e.reset(seed=8)
    half = float(e.track.wall_offset.min())
    seen = []
    for _ in range(200):
        obs, _, terminated, truncated, info = e.step(np.array([0.25, 0.7], dtype=np.float32))
        block = float(obs[-1])
        assert abs(block) <= 1.0, "saturated at the corridor edge, like the potential"
        # `info` carries the magnitude in metres; the block is signed and scaled.
        assert abs(block) == pytest.approx(min(info["cross_track"] / half, 1.0), abs=1e-6)
        seen.append(block)
        if terminated or truncated:
            obs, _ = e.reset()
    e.close()
    seen = np.array(seen)
    assert (seen > 0).any() and (seen < 0).any(), (
        "the block never changed sign, so it is a magnitude wearing a minus key"
    )


def test_the_cross_track_sign_says_which_side_of_the_line_the_car_is_on():
    """Positive is left of the racing line's direction of travel.

    Pinned against geometry, not a recorded sign: the car is placed either side of
    a known point, facing along the line, and the block must disagree in sign and
    agree in magnitude. A flipped convention trains a policy to steer away.
    """
    e = RacingEnv(config=pinned(cross_track_history=1), seed=0)
    e.reset(seed=4)
    line = e.track.racing_line
    assert line is not None
    at = 60
    along = line[at + 1] - line[at]
    along /= np.linalg.norm(along)
    left = np.array([-along[1], along[0]])
    theta = float(np.arctan2(along[1], along[0]))

    readings = {}
    for name, side in (("left", +1.0), ("right", -1.0)):
        spot = line[at] + side * 2.0 * left
        e._state = replace(e.state, x=float(spot[0]), y=float(spot[1]), theta=theta)
        e._crosses.clear()
        _, cross = e.progress.project((e.state.x, e.state.y))
        e._crosses.extend([e._normalized_cross(cross)] * e.cfg.cross_track_history)
        readings[name] = float(e._observe()[-1])
    e.close()

    assert readings["left"] > 0.0 > readings["right"], f"sign convention flipped: {readings}"
    assert readings["left"] == pytest.approx(-readings["right"], rel=1e-3)


def test_the_cross_track_history_never_straddles_a_reset():
    """Seeded from the spawn's own offset, not zero and not the last episode.

    Zero would claim a car spawned 3 m off the line was on it, then jerk; the
    previous episode's offset belongs to another layout. Either way the difference
    across the boundary, the only reason to carry a history, is a fiction.
    """
    k = 3
    e = RacingEnv(config=pinned(cross_track_history=k), seed=0)
    obs, _ = e.reset(seed=5)
    half = float(e.track.wall_offset.min())
    _, cross = e.progress.project((e.state.x, e.state.y))
    block = obs[-k:]
    # Every slot holds the same spawn reading, so differences across the block are
    # exactly zero on the first frame.
    np.testing.assert_allclose(block, block[0], atol=1e-7)
    assert abs(block[0]) == pytest.approx(min(abs(cross) / half, 1.0), abs=1e-6)
    assert np.sign(block[0]) == np.sign(cross) or cross == 0.0

    for _ in range(6):
        e.step(np.array([0.6, 0.8], dtype=np.float32))
    obs, _ = e.reset(seed=6)
    fresh = obs[-k:]
    np.testing.assert_allclose(fresh, fresh[0], atol=1e-7)
    e.close()


def test_the_ray_fan_is_focused_forward_by_the_ratio_it_promises():
    """`ray_focus` is an edge/centre spacing ratio, and the fan must honour it.

    Easy to get backwards: `atan` instead of `tan` spreads the centre and crowds
    the shoulders, the opposite of the point, and looks plausible in a plot.
    """
    focus, half_fov = 4.0, math.radians(120.0)
    angles = ray_angles(61, 240.0, focus)
    gaps = np.diff(angles)

    assert (gaps > 0).all(), "bearings must be monotone"
    np.testing.assert_allclose(angles, -angles[::-1], atol=1e-12)
    np.testing.assert_allclose([angles[0], angles[-1]], [-half_fov, half_fov])
    # Densest at the centre, sparsest at both edges, and the realized ratio
    # approaches `focus` from below: 61 rays sample the profile, not its endpoints.
    np.testing.assert_allclose(gaps.min(), gaps[len(gaps) // 2], rtol=1e-9)
    np.testing.assert_allclose(gaps.max(), gaps[0], rtol=1e-9)
    np.testing.assert_allclose(gaps[0], gaps[-1], rtol=1e-9)
    assert focus * 0.9 < gaps.max() / gaps.min() < focus


def test_unit_ray_focus_is_the_uniform_fan():
    angles = ray_angles(60, 180.0, 1.0)
    np.testing.assert_allclose(angles, np.linspace(-math.pi / 2, math.pi / 2, 60), atol=1e-12)


def test_same_seed_reproduces_the_same_episode():
    def rollout(seed):
        e = RacingEnv(config=CFG, seed=seed)
        obs, _ = e.reset(seed=seed)
        trace = [obs]
        for _ in range(20):
            obs, _, term, trunc, _ = e.step(np.array([0.2, 0.5], dtype=np.float32))
            trace.append(obs)
            if term or trunc:
                break
        e.close()
        return np.asarray(trace)

    np.testing.assert_array_equal(rollout(5), rollout(5))


def test_reset_with_the_same_seed_reproduces_the_episode_on_a_reused_env():
    """`reset(seed=S)` must give the same episode whatever the env did before.

    The test above builds a fresh env per rollout, so it only compares first
    resets and cannot see history leak. It did leak: the pool was built lazily
    from the generator `reset` re-seeds, so the first reset spent 64 draws on
    layouts and later ones spent none, giving the same seed a different track and
    spawn. Reusing one env across an intervening episode is what catches it.
    """
    e = RacingEnv(config=RacingEnvConfig(n_tracks=8, max_episode_steps=50), seed=0)
    e.action_space.seed(0)

    def snapshot():
        obs, _ = e.reset(seed=7)
        spawn = (e.state.x, e.state.y, e.state.theta, e.state.vx)
        return e.track.seed, spawn, obs

    first_seed, first_spawn, first_obs = snapshot()
    # An intervening episode differing in every way the streams could couple:
    # another layout, another spawn, and a run of noise draws of its own length.
    e.reset(seed=21)
    assert e.track.seed != first_seed, "the intervening episode must move the pool draw"
    for _ in range(20):
        _, _, terminated, truncated, _ = e.step(e.action_space.sample())
        if terminated or truncated:
            break
    again_seed, again_spawn, again_obs = snapshot()
    e.close()

    assert again_seed == first_seed, "same seed, different layout"
    assert again_spawn == first_spawn, "same seed, different spawn"
    np.testing.assert_array_equal(again_obs, first_obs)


def test_actions_are_clipped_not_rejected(env):
    env.reset(seed=3)
    # SB3 can emit slightly out-of-bounds actions; the env must absorb them.
    obs, reward, *_ = env.step(np.array([9.0, -9.0], dtype=np.float32))
    assert np.isfinite(obs).all() and np.isfinite(reward)
    assert abs(env.state.steer) <= env.params.max_steer + 1e-9


def test_step_info_carries_the_metrics_callbacks_log(env):
    """`TrainingMetricsCallback` reads `info` by key: a missing one is a `KeyError`
    thousands of steps in, a renamed one is a TensorBoard channel that silently
    stops being written. The callback's own table is imported, not restated.
    """
    env.reset(seed=4)
    _, _, _, _, info = env.step(np.array([-0.5, -0.8], dtype=np.float32))
    assert {"lap_progress", "track_position", "on_track", "speed", "off_track"} <= set(info)
    assert set(_STEP_MEANS) <= set(info), "a metric the callback logs has no key in `info`"
    assert 0.0 <= info["track_position"] <= 1.0
    # Both channels are logged as rollout means, so both are magnitudes. Signed,
    # alternating flat-out and hard braking would average to a coast, and equal
    # left and right steering would report as straight.
    assert info["action_steer"] == pytest.approx(0.5)
    assert info["action_throttle_abs"] == pytest.approx(0.8)


def test_lap_progress_counts_distance_covered_not_position_on_the_lap():
    """Progress is what the car did, not where the reset put it.

    Spawns are drawn anywhere on the lap, so raw arc length carries a U(0, 1)
    offset: a fresh episode reported 0.56 on average and a crash scored its spawn.
    Both the eval table and `episode/lap_progress` read this.
    """
    e = RacingEnv(config=pinned(), seed=0)
    starts, covered = [], []
    for _ in range(8):
        e.reset()
        _, _, _, _, first = e.step(np.zeros(2, dtype=np.float32))
        starts.append(first["track_position"])
        # One step at a standstill has covered essentially nothing.
        assert abs(first["lap_progress"]) < 1e-3, "progress must start at zero"
        before = np.array([e.state.x, e.state.y])
        distance = 0.0
        for _ in range(200):
            _, _, term, trunc, info = e.step(np.array([0.0, 0.7], dtype=np.float32))
            now = np.array([e.state.x, e.state.y])
            distance += float(np.linalg.norm(now - before))
            before = now
            if term or trunc:
                break
        covered.append((info["lap_progress"], distance / e.progress.total_length))
    e.close()
    assert np.ptp(starts) > 0.3, "spawns must vary, or this proves nothing"
    for reported, actual in covered:
        # Along the racing line rather than through the air, so they agree to
        # within the cornering difference.
        assert reported == pytest.approx(actual, rel=0.15, abs=1e-3)


def test_leaving_the_track_terminates_with_the_penalty():
    # Full lock at full throttle leaves the corridor whichever way the layout
    # turns: ~8.6 m minimum radius against a 6.5 m half-width, before understeer.
    # So this pins the termination contract, not one seed's geometry.
    e = RacingEnv(config=pinned(), seed=0)
    e.reset(seed=0)
    # The penalty is a duration of flat-out progress, which pays <= 1 per step, so
    # it is worth `off_track_seconds / dt` reward units.
    cost = e.cfg.off_track_seconds / e.params.dt
    assert cost > 1.0, "a penalty worth under one step of progress is not a deterrent"
    # Slack is one step of every bounded per-step term: progress, the steering
    # cost, and the shaping potential.
    slack = 1.0 + e.cfg.steer_rate_penalty + e.cfg.cross_track_weight
    action = np.array([1.0, 1.0], dtype=np.float32)
    for _ in range(400):
        _, reward, terminated, truncated, info = e.step(action)
        if terminated:
            assert info["off_track"] is True
            assert reward <= -cost + slack
            break
        assert not truncated, "expected termination before truncation"
    else:
        pytest.fail("car never left the track at full lock and full throttle")
    e.close()


def test_cross_track_shaping_cannot_change_the_return_by_more_than_its_weight():
    """The shaping is potential-based, so it moves only *when* credit lands.

    `Phi` depends on state alone and the reward carries its difference, so an
    episode telescopes to `Phi(end) - Phi(start)`, bounded by the weight however
    long it ran. The weight cannot change the dynamics, so the same seed and
    actions give the same trajectory and the returns differ by at most that bound.
    A per-step cost instead of a difference fails by a factor of episode length.
    """
    actions = [np.array([0.3, 0.6], dtype=np.float32)] * 300

    def episode(weight):
        e = RacingEnv(config=pinned(cross_track_weight=weight), seed=0)
        e.reset(seed=0)
        total, steps = 0.0, 0
        for a in actions:
            _, reward, term, trunc, _ = e.step(a)
            total += reward
            steps += 1
            if term or trunc:
                break
        e.close()
        return total, steps

    shaped, n_shaped = episode(5.0)
    plain, n_plain = episode(0.0)
    assert n_shaped == n_plain > 20, "shaping must not change the trajectory"
    assert abs(shaped - plain) <= 5.0 + 1e-6


def test_truncates_at_the_step_limit():
    e = RacingEnv(config=pinned(max_episode_steps=5), seed=0)
    e.reset(seed=0)
    flags = [e.step(np.array([0.0, 0.0], dtype=np.float32))[3] for _ in range(5)]
    assert flags[-1] is True and not any(flags[:-1])
    e.close()


def test_a_crash_on_the_last_permitted_step_terminates_and_does_not_truncate():
    """Gymnasium's two flags are mutually exclusive, and the distinction pays: SB3
    bootstraps `V(s')` into a *truncated* step's target, so a crash landing on the
    step limit would have the wall's value added back and its penalty refunded.
    Terminate-only, truncate-only and stall-only are covered nearby; the overlap
    was not.

    The crash step is measured under a budget long enough not to interfere, then
    the identical rollout is replayed with the budget cut to that step, so the
    overlap is forced rather than hoped for.
    """
    lock = np.array([1.0, 1.0], dtype=np.float32)

    def rollout(max_episode_steps):
        """`(step the episode ended on, terminated, truncated)`."""
        e = RacingEnv(config=pinned(max_episode_steps=max_episode_steps), seed=0)
        e.reset(seed=0)
        try:
            for i in range(1, max_episode_steps + 1):
                _, _, terminated, truncated, _ = e.step(lock)
                if terminated or truncated:
                    return i, terminated, truncated
        finally:
            e.close()
        pytest.fail("full lock at full throttle never left the track")

    n, terminated, truncated = rollout(1000)
    assert (terminated, truncated) == (True, False), "the unconstrained crash must terminate"
    assert 1 < n < 1000

    # The same rollout, ending on the step the budget also expires on.
    assert rollout(n) == (n, True, False)
    # One step earlier the same rollout truncates instead, which is what makes the
    # case above a real overlap rather than a budget that never binds.
    assert rollout(n - 1) == (n - 1, False, True)


def test_a_parked_car_is_truncated_and_not_terminated():
    """Standing still pays exactly zero (no progress, no steering change, no
    shaping), so nothing in the reward ends the episode and the car sits out the
    budget. Truncated, not terminated: PPO bootstraps the value it cut off, so
    reclaiming the steps says nothing about the policy.
    """
    e = RacingEnv(config=pinned(stall_seconds=1.0), seed=0)
    e.reset(seed=0)
    e.state.vx = 0.0
    window = round(1.0 / e.params.dt)
    brake = np.array([0.0, -1.0], dtype=np.float32)
    stalled = None
    for i in range(1, window + 1):
        _, reward, terminated, truncated, info = e.step(brake)
        assert not terminated, "a stall is not a crash and carries no penalty"
        if truncated:
            stalled = (i, reward, info)
            break
    assert stalled is not None, f"never stalled in {window} steps"
    stalled_at, last_reward, last_info = stalled
    assert stalled_at == window, f"stalled at {stalled_at}, expected {window} steps"
    assert last_info["stalled"] and last_reward == pytest.approx(0.0)
    e.close()


def test_a_driving_car_never_reads_as_stalled():
    """The window measures failure to cover ground, so anything driving, including
    the crawl out of a hairpin, keeps resetting it. Full throttle here leaves the
    track long before the window elapses.
    """
    e = RacingEnv(config=pinned(stall_seconds=1.0), seed=0)
    e.reset(seed=0)
    for _ in range(round(1.0 / e.params.dt)):
        _, _, terminated, _, info = e.step(np.array([0.0, 1.0], dtype=np.float32))
        assert not info["stalled"]
        if terminated:
            break
    e.close()


def test_stall_detection_can_be_switched_off():
    e = RacingEnv(config=pinned(stall_seconds=0.0, max_episode_steps=200), seed=0)
    e.reset(seed=0)
    e.state.vx = 0.0
    flags = [e.step(np.array([0.0, -1.0], dtype=np.float32))[3] for _ in range(199)]
    assert not any(flags)
    e.close()


def test_progress_reward_does_not_spike_when_crossing_the_line():
    """Arc length is measured from the start/finish line, so a completed lap drops
    `s` from `total_length` back to ~0. Unwrapped, that step reads as about -2700
    reward units against a per-step scale of 1: one sample that dwarfs an episode
    and teaches the policy to park short of the line.

    Driving there takes a competent policy and thousands of steps, and pinning
    steer at 0 walls the car a few hundred metres in, so the branch had no
    coverage. The bookkeeping is driven directly instead: the car is placed on the
    racing line a quarter metre short, fast enough to pass it in one step, with
    arc-length state derived the way `reset` derives it.
    """
    e = RacingEnv(config=pinned(), seed=0)
    e.reset(seed=0)
    lut = e.progress
    # `n-1 -> 0` is the closing segment, and the only one arc length wraps on.
    tangent = lut.path[0] - lut.path[-1]
    tangent /= np.linalg.norm(tangent)
    start = lut.path[0] - 0.25 * tangent
    # 40 m/s covers 0.8 m per step, so the line is behind the car after one step
    # and the rollover must be absorbed in a single sample.
    e._state = replace(
        e.state,
        x=float(start[0]),
        y=float(start[1]),
        theta=float(np.arctan2(tangent[1], tangent[0])),
        vx=40.0,
        vy=0.0,
        r=0.0,
        steer=0.0,
    )
    s_before, cross = lut.project((e.state.x, e.state.y))
    e._s_prev, e._prev_steer = s_before, 0.0
    e._potential = e._cross_track_potential(cross)
    assert s_before > lut.total_length - 0.5, "the car has to start just short of the line"

    before = np.array([e.state.x, e.state.y])
    _, reward, terminated, truncated, info = e.step(np.zeros(2, dtype=np.float32))
    after = np.array([e.state.x, e.state.y])
    e.close()

    assert not terminated and not truncated
    assert info["track_position"] < 0.01, "the step has to actually cross the line"
    # Independent expectation: the chord covered over the distance one flat-out
    # step covers, which is what progress is a fraction of. Chord and arc differ
    # only by the curvature of a ~4 m straight, so 2% is generous.
    expected = float(np.linalg.norm(after - before)) / (e.params.top_speed * e.params.dt)
    assert 0.4 < expected < 0.7, "40 m/s is a bit over half of a flat-out step"
    assert info["r_progress"] == pytest.approx(expected, rel=0.02)
    # And the whole reward, on the per-step scale it actually lives on.
    assert abs(reward) < 2.0, "reward spike consistent with an unwrapped lap rollover"
    assert info["lap_progress"] > 0.0, "crossing the line is forward progress, not backward"


def test_fixed_seed_pins_every_episode_to_one_layout():
    e = RacingEnv(config=RacingEnvConfig(n_tracks=8, fixed_track_seed=99), seed=0)
    seeds = set()
    for _ in range(4):
        e.reset()
        seeds.add(e.track.seed)
    assert seeds == {99}
    e.close()


def test_pool_is_reused_across_resets(env):
    env.reset(seed=0)
    pool = env.pool
    tracks = {id(env.track)}
    for _ in range(6):
        env.reset()
        tracks.add(id(env.track))
    assert env.pool is pool
    assert len(tracks) <= CFG.n_tracks, "resets must draw from the pool, not regenerate"


def test_rgb_array_render_mode_draws_a_scene_that_follows_the_car(monkeypatch):
    """The only test here that touches `render/`, so it has to be worth something.
    Shape and dtype are plumbing: a `draw()` that filled the backdrop satisfies
    them. Worth pinning is that a scene is in the frame at all and that it is a
    function of the car's state, the two ways a viewer breaks silently while the
    recorder keeps writing well-formed video.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    e = RacingEnv(config=pinned(), render_mode="rgb_array")
    e.reset(seed=0)
    e.step(np.zeros(2, dtype=np.float32))
    parked = e.render()

    assert parked.ndim == 3 and parked.shape[2] == 3 and parked.dtype == np.uint8
    # Walls, racing line, car and HUD: a drawn frame carries hundreds of distinct
    # colours, a flat fill exactly one.
    assert len(np.unique(parked[::3, ::3].reshape(-1, 3), axis=0)) > 32
    assert parked.std() > 1.0

    start = np.array([e.state.x, e.state.y])
    # 120 flat-out steps is ~24 m on this layout, short of the ~142 steps it takes
    # to run out of corridor with the steering straight.
    for _ in range(120):
        _, _, terminated, truncated, _ = e.step(np.array([0.0, 1.0], dtype=np.float32))
        if terminated or truncated:
            break
    moved = float(np.linalg.norm(np.array([e.state.x, e.state.y]) - start))
    driven = e.render()
    e.close()

    assert moved > 15.0, "the two frames have to be drawn from materially different states"
    changed = float(np.any(parked != driven, axis=-1).mean())
    assert changed > 0.005, f"only {changed:.2%} of the frame moved with the car"


def test_unknown_render_mode_is_rejected():
    # "human" is a plausible Gymnasium mode this env does not implement;
    # failing at construction beats silently rendering nothing later.
    with pytest.raises(ValueError, match="render_mode"):
        RacingEnv(config=CFG, render_mode="human")
