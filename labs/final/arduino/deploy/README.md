# `arduino/deploy/`

The device half of deployment: the laptop simulates the car, this sketch is the
driver. One inference request per control step over USB CDC, 50 per second at
`CarParams.dt` = 0.02 s. No policy state, no track and no simulation live on the
board, so what it runs is a pure function of the observation the host sends.
`tinyml_racing/deploy/` is the other half.

## Files

| file         | what it does                                                                     |
| ------------ | -------------------------------------------------------------------------------- |
| `deploy.ino` | the dispatch loop: `link_begin()` in `setup`, then identify / serve / yield      |
| `link.h`     | the wire protocol: IDENTITY, the framed request, `Reply`, `reject()`, timing     |
| `tinyml.h`   | the kernel: `tinyml_quantize`, `tinyml_dot`, the tanh table, `tinyml_infer`      |
| `model.h`    | generated weights and shape macros, emitted by `tinyml-build`, never hand-edited |

## Building

`tinyml-build` compiles and flashes this directory. It requires a system
`arm-none-eabi-g++` rather than the gcc 7.2 the mbed core ships, checked by compiling
`build.ACLE_PROBE` and refused without one, and builds at `-O3 -funroll-loops` on the
core's `gnu++14`. To build by hand:

```bash
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble \
  --build-property compiler.path=/usr/bin/ \
  --build-property "compiler.cpp.extra_flags=-O3 -funroll-loops" \
  --upload -p /dev/ttyACM0 arduino/deploy
```

Then `uv run tinyml-board` replays the exported reference frames and diffs them
against the host emulator.

`Arduino.h`, the core and the CMSIS headers are none of them on a default include
path, so an editor needs the build's own flags to resolve `Serial`. `arduino-cli`
will emit them, and one `jq` pass points them at the tracked sources rather than at
the generated sketch copy:

```bash
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble \
  --only-compilation-database --build-path "$PWD/.arduino-build" arduino/deploy
jq --arg d "$PWD/arduino/deploy" '
  [.[] | select(.file | endswith("deploy.ino.cpp"))][0] as $e
  | ["deploy.ino", "link.h", "tinyml.h"]
  | map({directory: $e.directory, file: ($d + "/" + .),
         arguments: ($e.arguments[:-3] + ["-x", "c++", $d + "/" + .])})
' .arduino-build/compile_commands.json > arduino/deploy/compile_commands.json
```

Both outputs are gitignored, being absolute paths. clangd also needs
`--query-driver=/usr/bin/arm-none-eabi-*` to pick up the cross toolchain's own system
headers, and the editor has to read `.ino` as C++.

## Contracts

`tinyml.h` must reproduce `quantize.QuantModel` bit for bit, not approximately, and it
is included from C by `tests/ckernel` as well as from C++ here, so nothing in it may be
C++-only. `tinyml_infer(in, out)` takes `MODEL_N_IN` raw observations and fills
`MODEL_N_OUT` actions already clipped to `MODEL_CLIP`; both buffers are caller-owned,
the kernel allocates nothing, and its layer buffers are function statics, so it is not
reentrant. Two paths in it are ARM-only and unreachable by the host harness: the `SMLAD`
dot product (`TINYML_DOT_DSP`) and the `VCVTR` quantizer (`TINYML_QUANT_VFP`). Each has a
portable spelling the harness diffs exactly, and the ARM spelling is verified against a
board.

The kernel evaluates tanh only, so `MODEL_ACTIVATION` is a build-time cross-check rather
than a dispatch: a `model.h` that does not say `TINYML_TANH` fails the `#if`. What the
board says about itself is the other half of that guard. IDENTITY reports arch,
`TINYML_ACT_NAME`, `n_in`, `n_out` and `MODEL_DIGEST`, so `Board.verify` can refuse a
model that disagrees with the host's `QuantModel`, or a stale upload, before a single
step is driven. `MODEL_N_IN` must equal the training-time observation width.

A request is `'R'`, then `MODEL_N_IN` float32 observations (little-endian, host order),
then one xor8 byte over the payload. A reply is `'A'`, the `Reply`, one xor8 byte over
the `Reply`. Rejections are `'E'` plus a `uint16_t`: the short-read count, or
`REJECT_BAD_CHECKSUM` / `REJECT_NO_CHECKSUM`. `reject()` drains to a 2 ms idle gap
_before_ writing the error frame, so the host's next command byte survives and no
payload byte reaches `loop()`, which dispatches on single bytes.

**Nothing in this sketch may read from `Serial`.** The receive path is `link_on_rx`,
`rx_ring` and `read_exact`, and the reason is a latency contract: `USBSerial::read()`
costs ~8 us a byte in USB locks and `Stream::readBytes` adds a `millis()` per byte on
top, which at 246 bytes a request was 5.2 ms of a 5.9 ms control step. Taking the whole
packet with `USBCDC::receive_nb` instead costs 0.48 us, but only works while the core's
own 256-byte ring stays full, and a single `Serial.read`, `available` or `peek` anywhere
would free space in it and hand the next packet back to the slow path. `Board._identify`
fills that ring during the handshake, and an answered handshake is the proof it is full.
See [docs/findings/link-latency.md](../../docs/findings/link-latency.md).

`struct Reply` is `packed` and `aligned(4)`. The host unpacks the literal `<{n_out}fHH`,
so a field added here would surface as noise rather than an error and the `static_assert`
on `sizeof(Reply)` is the only guard. `aligned(4)` is load-bearing: `&reply.action[0]` is
handed to `tinyml_infer` as a `float *`, and an underaligned VSTR on Cortex-M4's VFP is a
UsageFault. Of the two timings it carries, `us_infer` brackets `tinyml_infer` alone and
`us_read` covers the read plus the checksum verify; both saturate at 0xFFFF via
`clamp_us`. `setup()` does not wait for the USB host at all: the handshake is
`Board.__init__`'s, after a 1.5 s sleep that covers the reboot an upload leaves behind.
Opening the port does not reset this board -- unlike an AVR, the Nano resets only on the
1200-baud touch -- so a sketch that has already been primed stays primed across a
reconnect.

## Tests

| test                      | what it pins here                                                             |
| ------------------------- | ------------------------------------------------------------------------------- |
| `test_quantized_model.py` | `tests/ckernel/main.c` compiled against this header, diffed against the emulator |
| `tinyml-board`            | the framing, the reject sentinels and the drain, against real hardware           |

The compiled diff is for exact equality, including saturating frames and the
`|x| == 8.0` table edge, and it runs once under ASAN. Not covered by it: `link.h` needs
`Serial`/`millis`/`micros`, so nothing in this directory's wire half runs off the board.

Decisions and measurements:
[docs/findings/kernel-speed.md](../../docs/findings/kernel-speed.md),
[docs/findings/link-latency.md](../../docs/findings/link-latency.md),
[docs/findings/quantization.md](../../docs/findings/quantization.md).
