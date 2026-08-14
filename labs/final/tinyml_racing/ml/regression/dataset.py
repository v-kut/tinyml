"""Teacher rollouts: the (observation, action) pairs both supervised stages fit.

Noise rides on the *executed* action while the label stays the teacher's own, so
the clone sees the states its own errors produce (Laskey et al., DART, 2017).
Rewards ride along too, for the `VecNormalize` statistics and critic PPO inherits.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tinyml_racing import progress
from tinyml_racing.ml.config import RacingEnvConfig
from tinyml_racing.ml.env import RacingEnv
from tinyml_racing.sim.expert import PurePursuit
from tinyml_racing.sim.geometry import ArcLengthLUT

logger = logging.getLogger(__name__)

# `(env, obs) -> action in [-1, 1]^2`. Both arguments, because the two teachers
# want different halves of the world: the expert reads privileged simulator
# state off `env`, a policy reads the observation the deployed car would have.
Teacher = Callable[[RacingEnv, np.ndarray], np.ndarray]


def pure_pursuit_teacher() -> Teacher:
    """The analytic expert, in the action units the environment consumes.

    `PurePursuit.control` returns a front-wheel angle; `RacingEnv` takes a
    fraction of full lock, so the conversion uses the env's `params.max_steer`.
    One controller per layout, cached on the pool's stable `ArcLengthLUT` identity.
    """
    controllers: dict[ArcLengthLUT, PurePursuit] = {}

    def act(env: RacingEnv, _obs: np.ndarray) -> np.ndarray:
        controller = controllers.get(env.progress)
        if controller is None:
            controller = PurePursuit(env.progress.path, env.params)
            controllers[env.progress] = controller
        steer, throttle = controller.control(env.state)
        return np.array([steer / env.params.max_steer, throttle], dtype=np.float32)

    return act


def snapshot_teacher(snapshot) -> Teacher:
    """A trained policy as a teacher, for distillation.

    Deterministic: the student is fit to the mean the deployed car evaluates,
    not to samples from the exploration distribution around it.
    """

    def act(_env: RacingEnv, obs: np.ndarray) -> np.ndarray:
        return np.asarray(snapshot.act(obs), dtype=np.float32)

    return act


@dataclass(frozen=True)
class Dataset:
    """One teacher's rollouts, flattened into arrays.

    `obs` is *raw*, as the environment produced it: normalization is fit from
    this and applied afterwards, so the statistics and the samples cannot
    disagree about which observations they describe.
    """

    obs: np.ndarray  # (N, obs_dim) float32, unnormalized
    actions: np.ndarray  # (N, 2) float32, the teacher's own, pre-noise
    # (N,) float32, raw env reward for the action that was *executed*, under
    # DART that is `actions[i]` plus noise, so anything reading `rewards` as the
    # labelled policy's value needs a `noise_std=0.0` dataset (`returns_to_go`).
    rewards: np.ndarray
    dones: np.ndarray  # (N,) bool, True on the last step of an episode
    episodes: int
    crashes: int

    def __len__(self) -> int:
        return len(self.obs)

    @property
    def crash_rate(self) -> float:
        return self.crashes / max(self.episodes, 1)

    def reward_accumulator(self, gamma: float) -> np.ndarray:
        """`VecNormalize`'s `returns` buffer, replayed over this dataset.

        The wrapper divides every reward by the std of this quantity, a
        *forward* discounted accumulator that resets per episode, not the
        reward-to-go, so seeding `ret_rms` from anything else is not seeding it.
        """
        out = np.empty(len(self.rewards), dtype=np.float64)
        running = 0.0
        for i, (reward, done) in enumerate(zip(self.rewards, self.dones, strict=True)):
            running = gamma * running + float(reward)
            out[i] = running
            if done:
                running = 0.0
        return out

    def returns_to_go(self, gamma: float, reward_scale: float, clip: float) -> np.ndarray:
        """Value targets, in the units PPO's critic will be trained in.

        `reward_scale` and `clip` are what `VecNormalize` does to every reward
        before PPO sees it, and truncated episodes are not bootstrapped. Belongs on
        a `noise_std=0.0` dataset, since the value is the *executed* driver's.
        """
        scaled = np.clip(self.rewards.astype(np.float64) / reward_scale, -clip, clip)
        out = np.empty(len(scaled), dtype=np.float64)
        running = 0.0
        for i in range(len(scaled) - 1, -1, -1):
            running = scaled[i] + (0.0 if self.dones[i] else gamma * running)
            out[i] = running
        return out


def collect(
    teacher: Teacher,
    env_cfg: RacingEnvConfig,
    n_samples: int,
    *,
    noise_std: float,
    seed: int,
    label: str = "teacher",
) -> Dataset:
    """Drive `env_cfg` with `teacher` until `n_samples` steps are recorded.

    Single-process on purpose: a teacher step costs ~0.3 ms, so 200k samples is
    about a minute, less than the startup and pickling a vectorized collector
    would add.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    rng = np.random.default_rng(seed)
    env = RacingEnv(config=env_cfg, seed=seed)

    obs_buf = np.empty((n_samples, env_cfg.obs_dim), dtype=np.float32)
    act_buf = np.empty((n_samples, 2), dtype=np.float32)
    rew_buf = np.empty(n_samples, dtype=np.float32)
    done_buf = np.zeros(n_samples, dtype=bool)
    episodes = 0
    crashes = 0

    obs, _ = env.reset(seed=seed)
    collected = n_samples
    with progress.stage(label, n_samples) as bar:
        for i in range(n_samples):
            action = np.asarray(teacher(env, obs), dtype=np.float32)
            executed = action
            if noise_std > 0.0:
                executed = action + rng.normal(0.0, noise_std, size=action.shape)
            # Clipped whether or not noise was added: [-1, 1]^2 is this loop's
            # invariant to hold, not something to inherit from `RacingEnv.step`
            # happening to clamp too.
            executed = np.clip(executed, -1.0, 1.0)

            obs_buf[i] = obs
            act_buf[i] = action
            obs, reward, terminated, truncated, _ = env.step(executed)
            # The reward of `executed`, not of `act_buf[i]`: targets built from
            # these estimate the *perturbed* driver's return, which is why the
            # critic's rows come from a separate noise-free pass.
            rew_buf[i] = reward

            if terminated or truncated:
                done_buf[i] = True
                episodes += 1
                crashes += int(terminated)
                obs, _ = env.reset()
            bar.advance(note=f"{episodes} episodes")
            # Checked after the sample is stored, so a skip keeps it: the dataset
            # is simply shorter than asked for.
            if bar.skipped:
                collected = i + 1
                break

    env.close()
    obs_buf, act_buf = obs_buf[:collected], act_buf[:collected]
    rew_buf, done_buf = rew_buf[:collected], done_buf[:collected]

    # The trailing partial episode still ends here as far as the discounting
    # is concerned: there is no further reward recorded to credit it with.
    if not done_buf[-1]:
        done_buf[-1] = True
        episodes += 1

    dataset = Dataset(
        obs=obs_buf,
        actions=act_buf,
        rewards=rew_buf,
        dones=done_buf,
        episodes=episodes,
        crashes=crashes,
    )
    logger.info(
        "%s: %d samples over %d episodes, %.0f%% ended in a crash, mean reward %.3f/step",
        label,
        len(dataset),
        dataset.episodes,
        100.0 * dataset.crash_rate,
        float(dataset.rewards.mean()),
    )
    return dataset
