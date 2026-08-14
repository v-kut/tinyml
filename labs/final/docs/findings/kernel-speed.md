# Making the int8 kernel fast on a Cortex-M4

Measured on a Nano 33 BLE by `tinyml-board`, which replays the exported reference
frames and diffs the actions against the host emulator. The kernel is three dense int8
layers: 1,120 MACs at 61-16-8-2, one activation per hidden unit, one requantization per
input element per layer.

## The chain

| step | result | cost to exactness |
| --- | --- | --- |
| mbed core's `-Os`, scalar dot product | 1,150 us | none |
| `-O3 -funroll-loops` on the sketch TU | 693 us | none |
| `SMLAD` + `SXTB16` dot product | 523 us | none |
| `tanhf` to a 257-knot table | below | one int8 level, bounded |
| `lrintf` to `VCVTR` | 234 to 138 us | none |

The first three rows were measured on the earlier 64-32-16-2 net, so they compare with
each other and not with the last. The ordering is the finding: two compile decisions
each beat the hand-written SIMD that followed.

## Dot product

`tinyml_dot` widens four int8 bytes with `SXTB16` and accumulates them with two
`SMLAD`s. GCC does not find this: with `TINYML_DOT_DSP=0` the same `-O3` compile emits
zero `SMLAD`, its vectorizer targeting NEON/MVE rather than the M4's packed-halfword
GPR ops.

Written as `__asm volatile` first, now ACLE intrinsics. Clean sketch builds are byte
for byte the same size with the same `smlad` count, and the intrinsics issue ~19 fewer
loads, since `volatile` forces spills around every statement.

## Activation

One `tanhf` per hidden unit was over a third of the budget. The table is 257 knots,
1/32 apart over [0, 8]: a couple of dozen instructions against roughly 250 cycles for
the call, and the sketch barely grows, the 1,028-byte table costing about what the libm
code did.

The only step that changes output bits. Interpolation error peaks at 9.4e-5, 84x below
the 1/127 LSB the value is requantized to one line later, which bounds any activation to
a one-level shift without forbidding one. So the tests assert the consequence: the action
changes by orders of magnitude less than int8 already costs, and closed-loop reward over
held-out tracks is indistinguishable from the `tanhf` build.

## `lrintf` was a library call

The quantizer runs 85 times per inference, and `lrintf` is not an instruction here: GCC
emits `bl lrintf` at every optimization level. Neither `-fno-math-errno` nor
`-ffast-math` removes it, and both make the code larger (559 static instructions against
725 and 653).

`VCVTR` is the instruction libm is called for. It rounds by FPSCR, round-half-even,
exactly what `lrintf` and NumPy's `rint` do, and ARMv7-M conversion already gives NaN as
0 and saturates out of range, so the float comparisons that guarded them move after the
conversion and become integer comparisons. Every float compare also cost a `VMRS`.

| | before | after |
| --- | --- | --- |
| `bl lrintf` sites | 8 | 0 |
| `vcmp`/`vmrs` pairs | 30 | 6 |
| inference, 128 frames | 234 us mean, 254 max | **138 us mean, 161 max** |
| board vs emulator (max) | 5.960e-08 | 5.960e-08 |
| int8 vs PyTorch | 0.04300 max, 0.00525 mae | unchanged |

1.7x, same value for every input, so the parity contract did not move. The portable
`lrintf` spelling stays for the host harness; the board verifies the `VCVTR` path.

## Breaking bit-exactness: refused

Each variant flashed and replayed:

| variant | inference | isolates |
| --- | --- | --- |
| current, bit-exact | 138 us | |
| tanh to identity | 116 us | tanh table: 22 us |
| plus no float dequantization | 109 us | `b + m * acc`: 7 us |
| plus truncating quantizer | 103 us | rounding and rails: 6 us |

So ~103 us is MACs and loads, which no relaxation touches. Integer requantization with a
fused int8-to-int8 activation table could recover most of the 35 us above that, a sixth
of the kernel. Refused because `tests/ckernel`'s exact-equality diff would become a
tolerance, `quantize.py` would carry a second definition of the model, and the loop waits
on the link anyway: a step costs ~5.8 ms of USB round trip, so 138 us is 2.4% of it and
0.7% of the 20 ms control interval.

Next exactness-preserving lever, unimplemented: layer 0 reloads the quantized activation
vector once per output channel (61 bytes, 16 times). Blocking two channels per pass halves
that traffic and only reorders adds within separate accumulators.

## Toolchain

`tinyml-build` compiles with the system `arm-none-eabi-g++`, not the gcc 7.2.1 (2017) the
mbed core ships, and refuses without one. The check is a compile (`build.ACLE_PROBE`), not
a version number: gcc 7.2's `arm_acle.h` neither declares the DSP intrinsics nor compiles
as C++.

A current compiler buys no newer `-std`. The core's `Arduino.h` defines `abs(x)` as a
macro, which collides with libstdc++ from C++17 on:

```
Arduino.h:63:16: error: expected unqualified-id before '(' token          # gnu++20
chrono.h:463:38: error: macro 'abs' passed 2 arguments, but takes just 1  # gnu++17
```

So the sketch stays at `gnu++14`. No `-ffast-math` either: float rules stay IEEE-exact
for the parity test, and it measured worthless. Where the current host GCC does pay off is
the harness, which adds `-Wdouble-promotion -Wconversion -Wsign-conversion
-Wfloat-conversion -Werror`. The first matters most on a single-precision FPU, where a
`double` is a soft-float call.
