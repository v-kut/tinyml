# `arduino/deploy/`

The device half: the laptop simulates the car, this sketch is the driver. One
inference request per control step over USB CDC, 50 per second at
`CarParams.dt` = 0.02 s. No policy state, no track, no simulation on the board.

## Files

| file         | owns                                                                             |
| ------------ | -------------------------------------------------------------------------------- |
| `deploy.ino` | the dispatch loop: `link_begin()` in `setup`, then identify / serve / yield      |
| `link.h`     | the wire protocol: IDENTITY, the framed request, `Reply`, `reject()`, timing     |
| `tinyml.h`   | the kernel: `tinyml_quantize`, `tinyml_dot`, the tanh table, `tinyml_infer`      |
| `model.h`    | generated weights and shape macros, emitted by `tinyml-build`, never hand-edited |

## Building

`tinyml-build` compiles and flashes this directory. It requires a system
`arm-none-eabi-g++` and refuses to build without one, because `tinyml_dot` uses
ACLE DSP intrinsics the mbed core's gcc 7.2 does not provide; the check is
`build.ACLE_PROBE`, a compile rather than a version test. Flags are
`-O3 -funroll-loops` at the core's `gnu++14`. To build by hand:

```bash
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble \
  --build-property compiler.path=/usr/bin/ \
  --build-property "compiler.cpp.extra_flags=-O3 -funroll-loops" \
  --upload -p /dev/ttyACM0 arduino/deploy
```

Then `uv run tinyml-board` replays the exported reference frames and diffs them
against the host emulator. See [docs/findings/kernel-speed.md](../../docs/findings/kernel-speed.md) for why the
flags, the intrinsics and the toolchain floor are what they are.

## Contracts

- `tinyml_infer(in, out)`: `in` is `MODEL_N_IN` raw observations, `out` receives
  `MODEL_N_OUT` actions already clipped to `MODEL_CLIP`. Both buffers are
  caller-owned, the kernel allocates nothing, and its layer buffers are function
  statics, so it is not reentrant.
- `tinyml.h` must reproduce `quantize.QuantModel` bit for bit. It is included
  from C by `tests/ckernel` as well as from C++ here, so nothing in it may be
  C++-only.
- Two paths are ARM-only and unreachable by the host harness: the `SMLAD` dot
  product (`TINYML_DOT_DSP`) and the `VCVTR` quantizer (`TINYML_QUANT_VFP`). Each
  has a portable spelling the harness diffs exactly; the ARM spelling is verified
  against a board.
- One activation. The kernel evaluates tanh only, so `MODEL_ACTIVATION` is a
  build-time cross-check, not a dispatch: a `model.h` that does not say
  `TINYML_TANH` fails the `#if`. `TINYML_ACT_NAME` is what IDENTITY reports and
  what `Board.verify` compares against `QuantModel.activation`.
- `MODEL_N_IN` must equal the training-time observation width. IDENTITY exports
  arch, activation, `n_in`, `n_out` and `MODEL_DIGEST` so `Board.verify` can
  refuse a mismatched pair before a step is driven.
- A request is `'R'`, then `MODEL_N_IN` float32 observations (little-endian, host
  order), then one xor8 byte over the payload. A reply is `'A'`, the `Reply`, one
  xor8 byte over the `Reply`. Rejections are `'E'` plus a `uint16_t`: the
  short-read count, or `REJECT_BAD_CHECKSUM` / `REJECT_NO_CHECKSUM`.
- `struct Reply` is `packed` and `aligned(4)`. The host unpacks the literal
  `<{n_out}fHH`, so a field added here surfaces as noise rather than an error. The
  `static_assert` on `sizeof(Reply)` is the only guard. `aligned(4)` is
  load-bearing: `&reply.action[0]` is handed to `tinyml_infer` as a `float *`, and
  an underaligned VSTR on Cortex-M4's VFP is a UsageFault.
- `reject()` drains to a 2 ms idle gap _before_ writing the error frame, so the
  host's next command byte survives and no payload byte reaches `loop()`, which
  dispatches on single bytes.
- `us_infer` brackets `tinyml_infer` alone; `us_read` covers the read plus the
  checksum verify. Both saturate at 0xFFFF via `clamp_us`.
- `setup()` does not wait for the USB host; the handshake is `Board.__init__`'s
  1.5 s sleep after opening the port (which resets the Nano).

## Tests

`tests/test_quantized_model.py` compiles `tests/ckernel/main.c` against this
header with the host `cc` and diffs it against the NumPy emulator for exact
equality, including saturating frames and the `|x| == 8.0` table edge, and runs it
once under ASAN. `link.h` needs `Serial`/`millis`/`micros`, so the framing, the
reject sentinels and the drain are exercised only against hardware, by
`tinyml-board`.
