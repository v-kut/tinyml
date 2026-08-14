# `tests/`

No hardware, no display, no network, no trained run: everything is generated in
process or in `tmp_path`.

```bash
uv run pytest
uv run pytest tests/test_quantized_model.py   # the C kernel diff alone
```

## Files

| file                         | pins                                                                                                                                                                                                                                                                          |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `conftest.py`                | the one `ActorExport` builder every deploy test uses (`make_export`, plus an `export` fixture)                                                                                                                                                                                |
| `test_track_geometry.py`     | track generation and `sim/geometry.py`: arc-length sampling, wall nesting, tangent/normal/curvature agreement, the derived corner floor, the exact caster against the polyline oracle, `ArcLengthLUT` closed forms                                                            |
| `test_car_dynamics.py`       | `sim/car.py` and `sim/expert.py`: top speed as a force balance, coast-down, brake-to-zero, the friction ellipse, power oversteer, trail braking, the low-speed kinematic manifold, servo slew, a 20k-step boundedness run, pure pursuit lapping four layouts, spawn contracts |
| `test_track_pool.py`         | `sim/track_pool.py`: seed determinism, cache identity, `from_seed_range` purity, train/eval seed disjointness, the prebuilt LUT                                                                                                                                               |
| `test_env_contract.py`       | `ml/env.py` and `ml/config.py`: `obs_dim` against the width `_observe` builds, block boundaries at reset, seeding streams, clipping, the info-dict keys, reward terms and shaping bound, termination vs truncation, stall, `rgb_array` render                                 |
| `test_ml_pipeline.py`        | `ml/regression/`: the episode-level train/val split and `ObsNorm` leakage, `Frame` straddle, `reward_clip` units, hand-computed returns-to-go, and the policy distillation teaches from                                                                                       |
| `test_snapshot_roundtrip.py` | `ml/snapshot.py`: normalization equals `VecNormalize`'s, a loaded snapshot's `act` equals `model.predict`, the version gate, publish atomicity, `BestSnapshotCallback` recording a best in snapshot form                                                                      |
| `test_artifact.py`           | `deploy/artifact.py`: the `actor.npz` round trip enumerated from `dataclasses.fields`, and the version-gate cases                                                                                                                                                             |
| `test_quantized_model.py`    | `deploy/quantize.py`, `deploy/codegen.py` and `arduino/deploy/tinyml.h`: the int8 arithmetic, the tanh table, flash accounting, the weight digest, and an exact-equality diff of the compiled kernel                                                                          |
| `test_onnx_export.py`        | `deploy/onnx_export.py`: the graph against the float baseline, the IO contract, metadata, and the refusal of an activation the kernel cannot run                                                                                                                              |
| `test_run_layout.py`         | `utils.Run`, `ml/rl/callbacks.py`, `deploy/manifest.py`: path contract and collisions, `Run.resolve` precedence, checkpoint rotation, manifest content and hashes                                                                                                             |
| `ckernel/main.c`             | host harness that compiles `tinyml.h` against a generated `model.h` and prints actions (and, in `tanh` mode, the table alone)                                                                                                                                                 |

## Contracts

- `test_quantized_model.py` shells out to the host `cc` (via `shutil.which`) to
  compile `ckernel/main.c` against a freshly generated header and diffs its stdout
  against `QuantModel` for exact equality. It is the only check that can catch a
  divergence between the C kernel and the Python definition of the deployed model.
- Private names are reached into deliberately in three places, because each is the
  thing under test: `sim.track._signed_area`, `sim.car._tire_forces`,
  `ml.regression.fit._episode_split`.
- `test_track_pool.py` asserts on `ml/config.py`'s seed ranges: the sim tests depend
  on the ml package by design, since the train/eval split is a property of the pair.

Decisions, gaps and the reasoning behind the tests:
[docs/findings/testing-strategy.md](../docs/findings/testing-strategy.md).
