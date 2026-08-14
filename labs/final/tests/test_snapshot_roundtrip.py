"""The snapshot must reproduce the trainer's inference path exactly, and land atomically.

Feeding raw observations to a policy trained on `VecNormalize`-normalized ones
produces a policy that never existed while looking entirely plausible, and
`deploy/quantize.py` folds these same statistics into the first layer, so the
drift would reach the board.

The second half of the file is about *when* a reader may see those bytes.
`deploy/export.py`, `tinyml-watch` and `PolicySnapshotCallback` all poll the run's
deliverable while a stage is writing it, so the publish has to be a rename and
not a rewrite, the module docstring promises exactly that.
"""

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tinyml_racing.ml import snapshot as snapshot_module
from tinyml_racing.ml.config import PolicyKwargs, RacingEnvConfig
from tinyml_racing.ml.env import RacingEnv
from tinyml_racing.ml.snapshot import (
    SNAPSHOT_VERSION,
    ObsNorm,
    load_snapshot,
    publish_snapshot,
    save_snapshot,
)
from tinyml_racing.utils import Run

ENV_CFG = RacingEnvConfig(n_tracks=1, fixed_track_seed=42, max_episode_steps=64)
# Deliberately different shapes: the payload stores one `net_arch` mapping, so
# two keys crossed or collapsed onto one another is exactly the round-trip bug
# this file exists to catch, and identical arches could not see it.
POLICY_KWARGS = PolicyKwargs(pi_arch=(32, 16), vf_arch=(24, 12))


@pytest.fixture(scope="module")
def trained():
    env = DummyVecEnv([lambda: Monitor(RacingEnv(config=ENV_CFG, seed=0))])
    venv = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # Statistics are injected, not learned: these tests need `obs_rms` to be a
    # non-trivial affine (the identity it starts as would let the normalization
    # bug through unnoticed), not a converged policy, and `model.learn` is by
    # a wide margin the slowest thing in the suite.
    rng = np.random.default_rng(0)
    # `VecNormalize.obs_rms` is typed as either one `RunningMeanStd` or a dict
    # of them (the Dict-observation case); this env has a Box space, so it is
    # the single one, and narrowing here is what lets the attributes be set.
    obs_rms = venv.obs_rms
    assert isinstance(obs_rms, RunningMeanStd)
    obs_rms.mean = rng.normal(scale=2.0, size=ENV_CFG.obs_dim)
    obs_rms.var = rng.uniform(0.25, 4.0, size=ENV_CFG.obs_dim)
    obs_rms.count = 4096.0
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=POLICY_KWARGS.as_kwargs(),
        seed=0,
        device="cpu",
        verbose=0,
    )
    yield model, venv
    venv.close()


@pytest.fixture(scope="module")
def snapshot(trained, tmp_path_factory):
    model, venv = trained
    path = Run(tmp_path_factory.mktemp("run")).snapshot
    save_snapshot(path, model.policy, ENV_CFG, POLICY_KWARGS, 0, obs_norm=ObsNorm.from_venv(venv))
    return path


def test_normalize_obs_matches_vecnormalize(trained, snapshot):
    _, venv = trained
    loaded = load_snapshot(snapshot)
    raw = np.random.default_rng(1).normal(size=ENV_CFG.obs_dim).astype(np.float32)
    np.testing.assert_allclose(
        loaded.normalize_obs(raw), venv.normalize_obs(raw), rtol=1e-5, atol=1e-6
    )


def test_loaded_policy_reproduces_the_trainers_actions(trained, snapshot):
    model, venv = trained
    loaded = load_snapshot(snapshot)
    env = RacingEnv(config=ENV_CFG, seed=1)
    obs, _ = env.reset(seed=1)
    for _ in range(20):
        expected, _ = model.predict(venv.normalize_obs(obs), deterministic=True)
        np.testing.assert_allclose(loaded.act(obs), expected, rtol=1e-5, atol=1e-6)
        obs, _, terminated, truncated, _ = env.step(expected)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


def test_version_mismatch_is_rejected(snapshot, tmp_path):
    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    payload["version"] = SNAPSHOT_VERSION + 1
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="version"):
        load_snapshot(bad)


@pytest.fixture
def deliverable(trained, tmp_path):
    """A run directory whose deliverable already holds a published snapshot.

    The pre-existing file is the whole point: an atomicity claim about a
    destination that does not exist yet is untestable, and the truncate-in-place
    bug only shows up on the second publish.
    """
    model, venv = trained
    obs_norm = ObsNorm.from_venv(venv)
    run = Run(tmp_path / "run")
    stage = run.training / "first.pt"
    publish_snapshot(stage, run.snapshot, model.policy, ENV_CFG, POLICY_KWARGS, 1, obs_norm)
    return run, model.policy, obs_norm


def digest(data: bytes) -> tuple[int, str]:
    """Length and hash, so a failed byte comparison prints two lines rather than
    a 100 kB diff of a torch archive.
    """
    return len(data), hashlib.sha256(data).hexdigest()


def test_publishing_swaps_the_deliverable_instead_of_rewriting_it(deliverable):
    """The regression: `shutil.copyfile(stage_path, deliverable)`.

    `copyfile` opens the destination `wb`, which truncates it in place, so for
    the few milliseconds a ~100 kB snapshot takes to write there is a valid path
    holding a prefix of a torch archive. A polling reader gets a `torch.load`
    failure at best. The observable difference between a rewrite and a rename is
    identity: `os.replace` gives the name to a different inode, `copyfile` keeps
    the old one. That is what this asserts, plus that the bytes that land are
    complete and loadable and that no `.tmp` is left in the directory.
    """
    run, policy, obs_norm = deliverable
    before = run.snapshot.stat().st_ino

    stage = run.training / "second.pt"
    publish_snapshot(stage, run.snapshot, policy, ENV_CFG, POLICY_KWARGS, 4242, obs_norm)

    assert run.snapshot.stat().st_ino != before, (
        "the deliverable was written in place; a reader could see a partial file"
    )
    # Complete, and the same bytes the stage file got, the two are meant to be
    # bit-identical by construction rather than by two serializations agreeing.
    assert digest(run.snapshot.read_bytes()) == digest(stage.read_bytes())
    assert load_snapshot(run.snapshot).num_timesteps == 4242
    assert not list(run.training.glob("*.tmp")), "temporary files left behind"


def test_publishing_never_opens_the_deliverable_path_for_writing(deliverable, monkeypatch):
    """The same guarantee, taken at the copy itself.

    The inode check above would also pass if the deliverable were unlinked and
    rewritten, which is not atomic either. This one pins the direction: every
    copy goes to some other path, and the deliverable's name is only ever
    acquired by a rename.
    """
    run, policy, obs_norm = deliverable
    real_copyfile = shutil.copyfile
    replaced: list[tuple[str, str]] = []

    def guarded_copyfile(src, dst, **kw):
        assert Path(dst) != run.snapshot, "copied straight onto the deliverable"
        return real_copyfile(src, dst, **kw)

    real_replace = Path.replace

    def recording_replace(self, target, **kw):
        replaced.append((str(self), str(target)))
        return real_replace(self, target, **kw)

    monkeypatch.setattr(snapshot_module.shutil, "copyfile", guarded_copyfile)
    monkeypatch.setattr(Path, "replace", recording_replace)

    stage = run.training / "third.pt"
    publish_snapshot(stage, run.snapshot, policy, ENV_CFG, POLICY_KWARGS, 7, obs_norm)

    assert str(run.snapshot) in [dst for _, dst in replaced], (
        "the deliverable did not arrive by rename"
    )


def test_a_reader_during_the_publish_sees_the_whole_previous_deliverable(deliverable, monkeypatch):
    """The guarantee stated as the race it prevents.

    The copy is made to stall halfway through, and the deliverable is read at
    that moment, which is what `deploy/export.py`'s poll does. With a sibling
    temp the reader sees the complete previous snapshot; with a copy straight
    onto the deliverable it sees half a torch archive.
    """
    run, policy, obs_norm = deliverable
    previous = run.snapshot.read_bytes()
    observed: list[bytes] = []

    def halting_copyfile(src, dst, **kw):
        data = Path(src).read_bytes()
        half = len(data) // 2
        with Path(dst).open("wb") as fh:
            fh.write(data[:half])
            fh.flush()
            # A concurrent reader, at the worst possible instant.
            observed.append(run.snapshot.read_bytes())
            fh.write(data[half:])
        return dst

    monkeypatch.setattr(snapshot_module.shutil, "copyfile", halting_copyfile)

    stage = run.training / "fourth.pt"
    publish_snapshot(stage, run.snapshot, policy, ENV_CFG, POLICY_KWARGS, 99, obs_norm)

    assert len(observed) == 1, "the copy hook did not run; this test's premise is gone"
    assert digest(observed[0]) == digest(previous), (
        f"a mid-publish reader saw {len(observed[0])} bytes, not the previous "
        f"deliverable's {len(previous)}"
    )
    assert digest(run.snapshot.read_bytes()) == digest(stage.read_bytes())
    assert load_snapshot(run.snapshot).num_timesteps == 99
