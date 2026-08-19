# `tinyml_racing/ml/`

Everything between the simulator and the deployment bundle: the Gymnasium
environment, the config tree that derives the observation layout, the three
training stages, and the snapshot file that is the only handoff to `../deploy/`.
Nothing here imports `deploy/`, and the viewer is imported lazily and only for
`rgb_array`.

## Files

| file                    | what it does                                                               |
| ----------------------- | -------------------------------------------------------------------------- |
| `config.py`             | the dataclass tree, and `obs_dim` as the one statement of the layout       |
| `config_io.py`          | JSON round-trip and git provenance                                         |
| `config_cli.py`         | argparse flags reflected off the dataclass fields                          |
| `env.py`                | `RacingEnv`: observations, reward, termination, and the `rgb_array` render |
| `rollout.py`            | the one deterministic rollout, shared by every scoring path                |
| `snapshot.py`           | the train -> deploy file format, statistics included                       |
| `train.py`              | the `tinyml-train` CLI that runs the three stages in order                 |
| `rl/ppo.py`             | the PPO stage: vec-env construction, seeding, and the warm start           |
| `rl/callbacks.py`       | metrics, snapshots, evaluation, and checkpoint rotation                    |
| `regression/dataset.py` | teacher rollouts, with DART noise on the executed action                   |
| `regression/fit.py`     | the supervised fit and its episode-level train/val split                   |
| `../progress.py`        | one level up: the live per-stage display, shared with the deploy CLIs      |

## Contracts

`obs_dim` is the only place the observation layout is written down and `_observe` builds
exactly what it counts, with blocks switched on by width so 0 drops one and narrows the
network. The history lives in the environment rather than the policy, which is what lets
the device evaluate a pure function, and actions are `[steer, throttle]` in `[-1, 1]`
while the `steer` reported back is the achieved, slew-limited angle. The handoff is one
file: every stage overwrites `training/snapshot.pt` by rename, and `SNAPSHOT_VERSION`
gates it, because `env_config_from_dict` tolerates unknown keys and an unstamped snapshot
would silently take a new block's default at a different `obs_dim` than its weights.
`TRAIN_SEED_RANGE` and `EVAL_SEED_RANGE` are disjoint, so no reported number comes from a
layout the policy trained on.

`snapshot.pt` is always the newest policy, never the best one: PPO republishes it as it
goes and `_finish` writes it again at exit, so an interrupted run still deploys. The one
place that is the wrong policy is distillation, whose student overwrites it and ships, so
that stage teaches from `training/best.pt` when the run has one: `EvalCallback` picks the
best mean return over the held-out layouts and `BestSnapshotCallback` records that policy
with the `VecNormalize` statistics it needs. SB3's own `best_model.zip` is not written,
having neither.

## Run directory

```
data/runs/<run>/
  config.json      env + training settings for every stage, and the git SHA
  train.log
  tb/              TensorBoard events
  training/        checkpoints, best.pt / final policy, VecNormalize stats, snapshot.pt
  artifacts/       written by ../deploy/
```

## Tests

| test                         | what it pins here                                                       |
| ---------------------------- | ----------------------------------------------------------------------- |
| `test_env_contract.py`       | the observation layout, seeding, reward terms, and when an episode ends |
| `test_ml_pipeline.py`        | the train/val split, rollout frames, and reward bookkeeping             |
| `test_snapshot_roundtrip.py` | a loaded snapshot acting like the policy it came from                   |

Not covered: `train.py` end to end, `fit_policy` beyond one epoch, and the DART noise
path statistically.

Decisions and measurements:
[docs/findings/observation-design.md](../../docs/findings/observation-design.md),
[docs/findings/training-stages.md](../../docs/findings/training-stages.md).
