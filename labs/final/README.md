# Spacium

A LiDAR to (steering, throttle) driving policy trained by PPO in simulation, quantized to int8, compiled into a C header, and run on an Arduino Nano 33 BLE. The host is the car (track, physics, LiDAR, rendering), the board is the driver, one USB round trip per control step.

## Setup

```bash
uv sync
arduino-cli core install arduino:mbed_nano
```

`tinyml-build` compiles the sketch with the system `arm-none-eabi-g++` and refuses to
build without one, because the kernel's DSP path uses ACLE intrinsics the mbed core's
gcc 7.2 lacks. Install your distribution's toolchain. The build probes it by compiling those intrinsics, so a toolchain that is too old fails.

Hardware is optional: everything except `--flash`, `tinyml-board` and
`--policy board` runs on the host alone.

## Workflow

```bash
uv run tinyml-train --run-name myrun
uv run tinyml-build --flash                    # newest run to model.h to upload
uv run tinyml-board                            # board vs host, frame by frame
uv run tinyml-watch --policy expert ppo int8 board
```

| command        | does                                                               |
| -------------- | ------------------------------------------------------------------ |
| `tinyml-train` | clone, PPO, distill into one run directory                         |
| `tinyml-build` | export, quantize, codegen, ONNX, evaluate, report, sketch, compile |
| `tinyml-board` | replay the exported reference frames through hardware and diff     |
| `tinyml-watch` | several policies driving in one window                             |

Each command takes the run implicitly: `data/runs/latest`, the symlink `tinyml-train`
repoints when a run starts. Training and building at once resolves the unfinished
run, so pass `--run-dir` (or `run_dir`) when both are live.

## Run directory

```
data/runs/<run>/
  config.json      env + training settings for every stage, and the git SHA
  train.log
  tb/              TensorBoard events; tensorboard --logdir data/runs/<run>/tb
  training/        checkpoints, best.pt / final policy, VecNormalize stats, snapshot.pt
  artifacts/       actor.npz, actor.onnx, model.h, report.json, manifest.json
```

## Layout

Each package has a README: what every file owns, the
invariants it holds, what its tests pin. Why the code is what it is, with the
measurements, lives in [`docs/findings/`](docs/README.md#findings).

| path                                                      | what it is                                                                 |
| --------------------------------------------------------- | -------------------------------------------------------------------------- |
| [`tinyml_racing/sim/`](tinyml_racing/sim/README.md)       | track generation, wall geometry, vehicle model, LiDAR, pure-pursuit expert |
| [`tinyml_racing/ml/`](tinyml_racing/ml/README.md)         | Gymnasium env, config tree, the three training stages, snapshot handoff    |
| [`tinyml_racing/deploy/`](tinyml_racing/deploy/README.md) | export, quantize, codegen/ONNX, evaluate, manifest, board                  |
| [`tinyml_racing/render/`](tinyml_racing/render/README.md) | pygame viewer and the `tinyml-watch` front end                             |
| [`tinyml_racing/utils.py`](tinyml_racing/utils.py)        | run-directory layout (`Run`) and logging                                   |
| [`tinyml_racing/progress.py`](tinyml_racing/progress.py)  | the live stage display every CLI reports through, and the `s` skip key     |
| [`arduino/deploy/`](arduino/deploy/README.md)             | the sketch: dispatch loop, wire protocol, int8 kernel, generated weights   |
| [`tests/`](tests/README.md)                               | including a C harness that compiles the generated kernel                   |
| [`docs/`](docs/README.md)                                 | findings, the proposal, the report, background reading                     |

## Checks

```bash
uv run pytest        # no hardware, no display
uv run ruff check .  # every rule, minus a documented ignore list in pyproject
uv run ty check      # every ty rule at error
```

`tests/test_quantized_model.py` compiles `tests/ckernel/main.c` against the real
`arduino/deploy/tinyml.h` with a host C compiler and diffs it against the NumPy
emulator, so the deployed kernel is checked without hardware. Any change to
`deploy/quantize.py` or `tinyml.h` must keep those two bit-identical.

## License

MIT, declared per [REUSE](https://reuse.software) in [`REUSE.toml`](REUSE.toml). The
papers under `docs/materials/` belong to their authors and are excluded.
