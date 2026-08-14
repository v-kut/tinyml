# How this suite decides what is true

## The oracle pattern

`sim/geometry.py` ships two ray casters and two containment tests. The exact one runs; the
sampled-polyline one has no production caller and exists so
`test_the_drawn_wall_converges_on_the_exact_one` can show the polyline error falling
quadratically with spacing, which makes "exact" a measurement rather than a claim.

The C kernel is the same shape: `QuantModel` is the definition and `tests/ckernel/main.c`
compiles the shipped header against a generated `model.h`, so the diff is between two
implementations of one specification. Checked for exact equality, with one case under ASAN,
which caught a `lut[257]` load whose value happened to be right because the interpolation
fraction was zero there.

## Composition sweep instead of a golden number

`obs_dim` is derived, so the environment test enumerates block-width combinations and
asserts the advertised space equals the width `_observe` writes. A new block cannot be added
without updating `obs_dim` or failing here.

## Atomicity is tested, not asserted

The publish trio checks that the inode changes, that the deliverable is never opened for
writing in place, that a reader interrupting a half-written copy sees the whole previous
file, and that no `.tmp` is left behind.

## Tolerances are bounded by what governs them

`ds/R` for the tangent, `1/(2R)` for the curvature of a square wave under a central
difference, rather than a literal that happened to hold at the radii the generator drew.

## Two paths need hardware

`tinyml.h`'s `SMLAD` dot product and `VCVTR` quantizer are ARM-only, so the harness diffs
the portable spelling of each and `tinyml-board` diffs the ARM spelling. The harness compile
is strict for the same reason a single-precision FPU is: `-Wdouble-promotion` catches a
`double` that would be a soft-float call on the board.

## Known gaps

- `ml/train.py`, `deploy/build.py`, `deploy/export.py`, `deploy/board.py` and
  `deploy/evaluate.py` have no automated coverage: they need torch and SB3, a serial port,
  or `arduino-cli`. `tinyml-board` is the manual check for the last two.
- No test draws pixels; the render package is covered as arithmetic.
- The DART noise path and `fit_policy` end to end are exercised, not measured.
