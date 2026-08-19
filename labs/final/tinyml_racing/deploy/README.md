# `tinyml_racing/deploy/`

The host half of deployment. It takes the trained actor out of the run's snapshot,
quantizes it to int8, writes it out as a C header, mirrors it as ONNX, scores every
variant in closed loop, and records what it found. `quantize.QuantModel` is the
reference implementation of the deployed model: the C kernel on the board has to
reproduce it exactly, not approximately.

`tinyml-build` runs the whole chain:

```
export -> quantize -> C header -> ONNX gate -> evaluate -> report -> manifest -> compile -> upload
```

## Files

| file             | what it does                                                           |
| ---------------- | ---------------------------------------------------------------------- |
| `artifact.py`    | the `actor.npz` format, and the torch-free boundary below `export.py`  |
| `export.py`      | the only torch consumer: snapshot in, folded actor and calibration out |
| `quantize.py`    | the int8 scheme, the tanh table, and the NumPy emulator of the kernel  |
| `codegen.py`     | `QuantModel` as `const` arrays plus `MODEL_*` macros                   |
| `onnx_export.py` | an opset-17 float32 graph of the folded actor, and its error           |
| `evaluate.py`    | float32 / int8 / board over shared held-out seeds, into `report.json`  |
| `manifest.py`    | shapes, digests, SHA-256s, and the headline error                      |
| `board.py`       | the host side of the serial protocol, and the `tinyml-board` CLI       |
| `build.py`       | the `tinyml-build` CLI that drives the chain above                     |

Both CLIs report through `../progress.py`, one bar per stage, which is a no-op
outside a terminal session so `build()` and `evaluate_run()` stay callable from a
script or a test.

## Contracts

`QuantModel` is the single statement of the deployed model. `codegen.py` serializes it,
`tinyml.h` has to match it bit for bit, and the two are diffed for equality rather than
for closeness. The same holds for the two strings that describe it: `quantize.ACTIVATION`
is checked by the exporter, the quantizer, the ONNX builder and the kernel's `#if`, and
`QuantModel.arch` is what codegen writes, what the report prints, and what `Board.verify`
compares against the board's handshake. The board also reports a digest of its compiled
weights, so a stale upload is caught before it drives instead of showing up as a bad lap.

Two version gates: `EXPORT_VERSION` on `actor.npz` and `SNAPSHOT_VERSION` on
the training, because both files went through several arch changes and we would want to avoid the reader silently
taking a default the weights were never trained with. Flash is always the deployed figure:
`flash_bytes` counts what codegen emits, `deployed_flash_bytes` adds the kernel's own tanh
table, and every compression number divides by the latter. Every variant drives the same
tracks, drawn from `EVAL_SEED_RANGE` and disjoint from the training seeds, so the float32,
int8 and board rows differ only in the model.

Order matters at the end of a build. `arduino/deploy/model.h` is overwritten after
`report.json` and `manifest.json`, so the sketch in the tree always corresponds to a model
whose evaluation was recorded, and the compile itself is gated on `toolchain_prefix`
finding a system `arm-none-eabi-g++` that can build `ACLE_PROBE`.

## Bundle

```
artifacts/
  actor.npz      exported actor: weights, normalization, calibration, references
  actor.onnx     portable float32 graph: raw observations in, action out
  model.h        int8 weights as C, what the board runs
  report.json    float32 vs int8 vs board over held-out tracks
  manifest.json  shapes, digests, SHA-256s, headline error
```

## Tests

| test                      | what it pins here                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| `test_quantized_model.py` | the int8 math, the tanh table, flash accounting, and the C kernel diffed against `QuantModel` |
| `test_artifact.py`        | the `actor.npz` round trip and its version gate                                               |
| `test_onnx_export.py`     | the ONNX graph against the float baseline, and its IO contract                                |
| `test_run_layout.py`      | run paths, checkpoint rotation, and manifest hashing                                          |

Not covered: `export.py`, `board.py`, `evaluate.py` and `build.py`, which need torch, a
serial port or `arduino-cli`. `tinyml-board` is the manual check for the device half.

Decisions and measurements: [docs/findings/quantization.md](../../docs/findings/quantization.md),
[docs/findings/kernel-speed.md](../../docs/findings/kernel-speed.md).
