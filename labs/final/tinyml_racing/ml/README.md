# `tinyml_racing/ml/`

Everything between the simulator and the deployment bundle: the Gymnasium
environment, the config tree that derives the observation layout, the three
training stages, and the snapshot file that is the only handoff to `../deploy/`.
Nothing here imports `deploy/`, and the viewer is imported lazily and only for
`rgb_array`.

## Files

| file                    | owns                                                                                                                                                                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `config.py`             | the dataclass tree: `RacingEnvConfig` (and `obs_dim`, the single definition of the observation layout), `PPOConfig`, `RegressionConfig`, `PolicyKwargs`, `TrainConfig`, the train/eval seed ranges, `CLIP_OBS`, `WARM_START_LOG_STD` |
| `config_io.py`          | JSON round-trip and provenance: `git_revision`, `config_to_dict`, `env_config_from_dict`                                                                                                                                             |
| `config_cli.py`         | argparse reflection over those dataclasses: `add_config_arguments`, `configs_from_args`. Types, arity and help come off the field annotations, so a new config field is a flag with no edit here                                     |
| `env.py`                | `RacingEnv`: observation assembly, reward, termination and truncation, the metrics in `info`, `rgb_array` render                                                                                                                     |
| `rollout.py`            | one deterministic rollout implementation: `Frame`, `iter_rollout`, `closed_loop`, `eval_seeds`                                                                                                                                       |
| `snapshot.py`           | the train -> deploy file format: `ObsNorm`, `SnapshotPayload`, `save_snapshot`, `publish_snapshot`, `load_snapshot`, `Snapshot.act`                                                                                                  |
| `train.py`              | the `tinyml-train` CLI: stage orchestration, `publish`, `score`, `link_latest`                                                                                                                                                       |
| `rl/ppo.py`             | the PPO stage: vec-env construction, worker seeding, `VecNormalize` seeding, reward clipping, pinned evaluation episodes, the warm start                                                                                             |
| `rl/callbacks.py`       | `TrainingMetricsCallback`, `PolicySnapshotCallback`, `ProgressCallback`, `RotatingCheckpointCallback`, `QuietEvalCallback`                                                                                                           |
| `regression/dataset.py` | teacher rollouts: `pure_pursuit_teacher`, `snapshot_teacher`, `Dataset`, `collect`                                                                                                                                                   |
| `regression/fit.py`     | the supervised fit: `build_policy`, the episode-level split, `fit_policy`, `FitResult`                                                                                                                                               |
| `../progress.py`        | one level up: the live one-line-per-stage display and the `s` skip key, shared with `tinyml-build` and `tinyml-board`. A no-op outside a session                                                                                     |

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

## Run directory

```
data/runs/<run>/
  config.json      env + training settings for every stage, and the git SHA
  train.log
  tb/              TensorBoard events
  training/        checkpoints, best/final policy, VecNormalize stats, snapshot.pt
  artifacts/       written by ../deploy/
```

## Tests

| file                                     | what it pins here                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../../tests/test_env_contract.py`       | `gamma`/`lambda` as durations, the observation space and its finiteness, the scan carried verbatim and its reset boundaries, the throttle channel, the composition sweep proving `obs_dim` equals the width `_observe` writes, the cross-track block's sign and range, seeding reproducibility including the pool/episode stream split, clipping, the `info` keys against the metric list the callbacks read, lap-progress semantics across the start/finish line, the crash penalty, the shaping telescoping bound, termination/truncation exclusivity, stall truncation, and the `rgb_array` render |
| `../../tests/test_ml_pipeline.py`        | the episode-level train/val split and the `ObsNorm` leakage it prevents, `Frame` straddle through `iter_rollout`, `reward_clip`'s units, hand-computed reward accumulation and returns-to-go, and action clipping in `collect`                                                                                                                                                                                                                                                                                                                                                                        |
| `../../tests/test_snapshot_roundtrip.py` | `ObsNorm.normalize_obs` equals `VecNormalize.normalize_obs`, a loaded snapshot's `act` equals `model.predict` over 20 real steps, the version gate, and the publish atomicity trio                                                                                                                                                                                                                                                                                                                                                                                                                    |

Not covered: `train.py` end to end, `fit_policy` beyond one epoch, and the DART
noise path statistically.

Decisions and measurements:
[docs/findings/observation-design.md](../../docs/findings/observation-design.md),
[docs/findings/training-stages.md](../../docs/findings/training-stages.md).
