# `tinyml_racing/deploy/`

The host half of deployment: pull the trained actor out of the snapshot, quantize
it to int8, emit it as a C header, mirror it as ONNX, score all of it in closed
loop over held-out layouts, and hash the result. `quantize.QuantModel` is the
definition of the deployed model; `arduino/deploy/tinyml.h` must match it bit for
bit.

## Files

| file             | owns                                                                                                                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `artifact.py`    | the `actor.npz` schema: `DenseLayer`, `ActorExport`, `EXPORT_VERSION`, save/load with a hard version gate. The torch-free boundary: nothing downstream of `export.py` imports torch                           |
| `export.py`      | `extract_actor`, `collect_calibration`, `export_actor`: the only torch consumer. Reads `training/snapshot.pt`, folds in the `VecNormalize` statistics, collects on-policy calibration states and reference IO |
| `quantize.py`    | the int8 scheme and the emulator: `ACTIVATION`, `quantize_model`, `QuantModel` (`arch`, `flash_bytes`, `deployed_flash_bytes`, `digest`), `TANH_LUT`, `float_actor`, `action_error`, `clipping_rate`          |
| `codegen.py`     | `emit_header` / `write_header`: `QuantModel` as `const` int8 and float32 blobs plus `MODEL_*` macros                                                                                                          |
| `onnx_export.py` | `build_model`, `write_onnx`, `onnx_error`: an opset-17 float32 graph of the folded actor, plus its error against the export                                                                                   |
| `evaluate.py`    | `evaluate_run`, `format_table`, `write_report`: float32 / int8 / board over shared held-out seeds, into `report.json`                                                                                         |
| `manifest.py`    | `bundle`, `sha256`, `write_manifest`: shapes, digests, SHA-256s, the headline error, `MANIFEST_VERSION`                                                                                                       |
| `board.py`       | the host side of the wire protocol: `find_port`, `BoardIdentity`, `Board` (`verify`, `infer`, `last_infer_us`, `last_round_trip_ms`), `self_test`, and the `tinyml-board` CLI                                 |
| `build.py`       | the `tinyml-build` CLI: export -> quantize -> codegen -> ONNX gate -> evaluate -> report -> manifest -> `update_sketch` -> `toolchain_prefix` -> compile -> optional upload                                   |

Both CLIs report through `../progress.py`, the trainer's live display: one bar per
stage (`calibrate`, `quantize`, `artifacts`, `evaluate`, `record`, `compile`,
`upload`), log lines above it, and the summary printed as plain text once the
display closes. The bars are a no-op outside a session, so `build()` and
`evaluate_run()` stay usable from a script or a test.

## Contracts

- **Two version gates, both fatal.** `EXPORT_VERSION` on `actor.npz` and
  `SNAPSHOT_VERSION` on the training handoff. Nothing reads a file whose meaning
  may have changed.
- **`QuantModel` is the reference implementation.** `codegen.py` serializes it,
  `tinyml.h` must reproduce it exactly, and `tests/test_quantized_model.py`
  compiles the C and diffs the two for equality, not for closeness.
- **One activation, `quantize.ACTIVATION`.** `extract_actor`, `quantize_model`,
  `onnx_export.build_model` and `tinyml.h`'s `#if` each refuse anything else.
- **One architecture string.** `QuantModel.arch` is what codegen writes, what the
  report prints and what `Board.verify` compares against the handshake.
- **Model identity.** The board reports a digest of its compiled weights;
  `Board.verify` refuses to run unless it matches the model on disk.
- **Flash figures are the deployed ones.** `flash_bytes` counts only what codegen
  emits; `deployed_flash_bytes` adds `tinyml.h`'s tanh table, and both the
  manifest's `compression` and the report divide by the deployed figure.
- **Ordering.** The generated header is copied over the tracked
  `arduino/deploy/model.h` as the last write of a build, after `report.json` and
  `manifest.json`, so the sketch in the tree always corresponds to a model whose
  evaluation was recorded.
- **The sketch build requires a current arm-none-eabi.** `toolchain_prefix`
  points `compiler.path` at the system `arm-none-eabi-g++` and refuses the build
  if it cannot compile `ACLE_PROBE`.
- **Every variant drives the same seeds**, drawn from `EVAL_SEED_RANGE`.

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

| file                                  | what it pins here                                                                                                                                    |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../../tests/test_quantized_model.py` | the int8 arithmetic, the tanh table including the `                                                                                                  | x   | == 8.0`edge, flash accounting, the weight digest, the header's float literals, and, via`tests/ckernel/main.c`compiled with the host`cc`, an exact-equality diff of the compiled kernel against `QuantModel`, plus an ASAN run |
| `../../tests/test_artifact.py`        | the `actor.npz` round trip enumerated from `dataclasses.fields`, so a new field cannot skip disk, and the version-gate cases                         |
| `../../tests/test_onnx_export.py`     | the graph against the float baseline, the named and batched IO contract, metadata provenance, and the refusal of an activation the kernel cannot run |
| `../../tests/test_run_layout.py`      | the `Run` path contract, `Run.resolve` precedence, checkpoint rotation, manifest content and hashing                                                 |
| `../../tests/conftest.py`             | the shared `ActorExport` builder all four use                                                                                                        |

Not covered: `export.py`, `board.py`, `evaluate.py`, `build.py`. They need torch, a
serial port or `arduino-cli`. `tinyml-board` is the manual check for the device
half.

Decisions and measurements: [docs/findings/quantization.md](../../docs/findings/quantization.md),
[docs/findings/kernel-speed.md](../../docs/findings/kernel-speed.md).
