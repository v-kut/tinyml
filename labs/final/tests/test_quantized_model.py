"""The deployed model is defined twice, in NumPy (`deploy/quantize.py`) and in C
(`arduino/deploy/tinyml.h`), and every number quoted downstream comes from the NumPy
side. If the C side disagrees those numbers describe a model never flashed, and
nothing else catches it: emulator against PyTorch measures quantization error rather
than transcription, and board against emulator needs hardware. So this compiles the
real kernel with the host compiler and diffs it.

It also pins the boundaries neither implementation reaches alone. Every scale here is
calibrated, so no reference frame quantizes past `+-127` and no activation reaches the
table's last knot, which is why `_quantize` and `tanh_lut` are tested directly and the
frames fed to the harness are extended onto both.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import make_export

from tinyml_racing.deploy import quantize
from tinyml_racing.deploy.artifact import ActorExport
from tinyml_racing.deploy.codegen import emit_header, write_header
from tinyml_racing.deploy.quantize import (
    QMAX,
    TANH_LUT,
    TANH_LUT_N,
    TANH_LUT_SCALE,
    _quantize,
    activate,
    float_actor,
    float_forward,
    float_layers,
    quantize_model,
    tanh_lut,
)

ROOT = Path(__file__).resolve().parents[1]
SKETCH = ROOT / "arduino" / "deploy"
HARNESS = Path(__file__).resolve().parent / "ckernel" / "main.c"
ARCH = (64, 32, 16, 2)
N_FRAMES = 96

CFLAGS = ("-O2",)
ASAN_CFLAGS = ("-fsanitize=address", "-fno-omit-frame-pointer", "-O1", "-g")
# The host compiler is a current GCC, so it is worth more than `-Wall`: the kernel also
# compiles for a single-precision FPU, where a `double` is a soft-float call rather than
# a lost ulp, and every narrowing is deliberate. `-std=c11` is the header's floor, since
# `tinyml.h` is included from C here and from C++ on the board.
WARNINGS = (
    "-std=c11",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wdouble-promotion",
    "-Wconversion",
    "-Wsign-conversion",
    "-Wfloat-conversion",
)

# What `flash_bytes` claims the header costs, read back off the header.
DECLARATION = re.compile(r"static const (\w+) (\*const )?(\w+)(?:\[(\d+)\])? =")
C_SIZEOF = {"int8_t": 1, "uint16_t": 2, "float": 4}
POINTER_BYTES = 4  # Cortex-M


def with_layer(export: ActorExport, i: int, **changes) -> ActorExport:
    """A copy of `export` with one layer's kernel or bias replaced."""
    layers = list(export.layers)
    layers[i] = replace(layers[i], **changes)
    return replace(export, layers=tuple(layers))


def saturating_frames(model: quantize.QuantModel) -> np.ndarray:
    """Frames that drive layer 0 onto both int8 rails and past the tanh domain.

    An observation aligned with the signs of one channel's weights maximizes that
    accumulator, the only way to reach a pre-activation the calibration set never
    produced. Eight channels, both signs, so both rails and both table ends.
    """
    layer = model.layers[0]
    signs = np.sign(layer.qw[:8].astype(np.float32))
    reach = np.float32(200.0) / np.float32(layer.inv_input_scale)  # well past the +-127 rail
    return (np.concatenate([signs, -signs]) * reach).astype(np.float32)


def edge_model(export: ActorExport) -> quantize.QuantModel:
    """A model whose layer-0 pre-activations are its biases.

    With a zero kernel every accumulator is zero, so `0 * mult + bias` is exactly
    `bias` on both sides, the only way to hand the activation an exact `+-8.0f`: the
    int8 rails cap what an observation can reach, and this export never lands on 8.

    `clip_action` is opened up because the `+-1` clip would hide the activations
    behind a constant.
    """
    edge = np.float32(8.0)
    grid = np.concatenate(
        [
            [edge, -edge],
            [np.nextafter(edge, np.float32(0.0)), -np.nextafter(edge, np.float32(0.0))],
            [np.nextafter(edge, np.float32(np.inf)), -np.nextafter(edge, np.float32(np.inf))],
            [8.5, -8.5, 30.0, -30.0, 1e6, -1e6],
            np.linspace(-7.9, 7.9, ARCH[1] - 12),
        ]
    ).astype(np.float32)
    assert grid.size == ARCH[1]

    zeroed = with_layer(export, 0, w=np.zeros_like(export.layers[0].w), b=grid)
    return replace(quantize_model(zeroed), clip_action=100.0)


def tanh_probe() -> np.ndarray:
    """Every knot, both signs, plus the edge and everything beyond it."""
    knots = (np.arange(TANH_LUT_N + 1, dtype=np.float32) / TANH_LUT_SCALE).astype(np.float32)
    edge = np.float32(8.0)
    special = np.array(
        [
            edge,
            np.nextafter(edge, np.float32(0.0)),
            np.nextafter(edge, np.float32(np.inf)),
            8.5,
            30.0,
            1e30,
            np.inf,
            0.015625,  # mid-cell, exactly representable
            7.984375,  # mid-cell in the last interval
        ],
        dtype=np.float32,
    )
    return np.concatenate([knots, -knots, special, -special]).astype(np.float32)


@pytest.fixture(scope="module")
def export() -> ActorExport:
    return make_export(ARCH, calib_rows=2048, reference_rows=N_FRAMES)


@pytest.fixture(scope="module")
def model(export):
    return quantize_model(export)


def test_quantize_rounds_halves_to_even():
    """`_quantize` claims `np.rint` and C's `lrintf` agree on ties, and the
    exact-equality diff rests on it. `floor(x + 0.5)`, the obvious spelling, rounds
    ties away from even and moves half the grid one level against the board.
    """
    halves = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    even = np.array([-2, -2, 0, 0, 2, 2], dtype=np.int8)
    np.testing.assert_array_equal(_quantize(halves, 1.0), even)

    # `inv_scale` multiplies, so halving the input at scale 2 is the same grid.
    np.testing.assert_array_equal(_quantize(halves / np.float32(2.0), 2.0), even)


def test_quantize_saturates_on_both_rails():
    """The clamp is unreachable through `quantize_model` and load-bearing: an
    off-distribution activation does quantize past `+-127`, and wrapping instead of
    clamping turns hard-left into hard-right. `+-127.5` is the pair worth having, its
    tie rounding outward so the clamp must catch what rounding produced.
    """
    x = np.array([-200.0, -127.5, -127.0, 0.0, 127.0, 127.5, 200.0], dtype=np.float32)
    q = _quantize(x, 1.0)
    np.testing.assert_array_equal(q, np.array([-127, -127, -127, 0, 127, 127, 127], dtype=np.int8))
    assert q.dtype == np.int8


def test_quantize_never_emits_the_asymmetric_level():
    """-128 exists in int8 and is deliberately unused.

    Both scales are symmetric about zero, so the extra negative level would bias a
    signed output towards it at the rail, and the kernel returns -127 there, so an
    emulator returning -128 disagrees by a full level. The clamp is asserted as total
    anyway, a bound being easier to trust than a distribution.
    """
    probe = np.concatenate(
        [
            -np.geomspace(1e-3, 1e30, 128).astype(np.float32),
            np.array([-np.inf, np.finfo(np.float32).min, -200.0, -128.0], dtype=np.float32),
        ]
    ).astype(np.float32)

    with np.errstate(over="ignore"):
        for inv_scale in (1.0, 0.5, 82.47785186767578, 1e9):
            q = _quantize(probe, inv_scale)
            assert not np.any(q == -128), f"inv_scale={inv_scale} emitted the unused level"
            # And it is the rail, not an empty probe, that made that true.
            assert q.min() == -QMAX, f"inv_scale={inv_scale} never reaches the negative rail"


def test_folding_normalization_preserves_the_function(export):
    """The fold is an algebraic identity, not an approximation."""
    raw = export.calibration[:256]
    folded, _ = float_forward(float_layers(export, True), raw)

    scale = np.sqrt(export.obs_var + export.epsilon)
    normalized = np.clip((raw - export.obs_mean) / scale, -export.clip_obs, export.clip_obs).astype(
        np.float32
    )
    unfolded, _ = float_forward(float_layers(export, False), normalized)

    np.testing.assert_allclose(folded, unfolded, rtol=1e-4, atol=1e-5)


def test_weights_use_the_full_int8_range(model):
    """A per-channel scale that leaves headroom is a wasted scale: every output
    channel should push at least one weight to the rail, and a peak well under
    127 means the scales were computed against the wrong axis.
    """
    for i, layer in enumerate(model.layers):
        peak = np.abs(layer.qw).max(axis=1)
        assert peak.min() == QMAX, f"layer {i} has an under-ranged output channel"


def test_int8_tracks_float32(export, model):
    """Quantization is allowed to cost accuracy, not to change the policy."""
    reference = np.stack([float_actor(export)(o) for o in export.calibration[:512]])
    quantized = model(export.calibration[:512])
    assert np.abs(quantized - reference).mean() < 0.01
    assert np.abs(quantized - reference).max() < 0.05


@pytest.mark.parametrize("push", [50.0, -50.0])
def test_actions_are_clipped_to_the_action_space(export, push):
    """Both rails, because `np.clip(h, 0.0, clip_action)` satisfies the positive
    case on its own and leaves a car that cannot steer left or brake.
    """
    loud = with_layer(export, 2, b=np.full_like(export.layers[2].b, push))
    out = quantize_model(loud)(export.calibration[:64])
    assert out.min() >= -1.0 and out.max() <= 1.0
    assert np.all(out == np.sign(push))


def test_digest_tracks_the_weights(export):
    """The board's identity check is only worth anything if this holds."""
    baseline = quantize_model(export).digest()
    assert quantize_model(export).digest() == baseline

    w = export.layers[1].w.copy()
    w[0, 0] *= 1.5
    nudged = with_layer(export, 1, w=w)
    assert quantize_model(nudged).digest() != baseline


def test_header_has_no_bare_integer_float_literals(model):
    """`1f` is a syntax error; `%.9g` produces it for every round number."""
    header = emit_header(model)
    for token in header.replace(",", " ").replace("{", " ").replace("}", " ").split():
        if token.endswith("f") and token[0].isdigit():
            assert "." in token or "e" in token, f"invalid C float literal {token!r}"


def test_flash_bytes_counts_every_object_the_header_declares(model):
    """`deployed_flash_bytes`, what the manifest and report divide by, is this figure
    plus the tanh table, so an optimistic count understates the budget that overflows.
    Every `static const` object the generator declares, sized by its C type, has to add
    up: counting only the weight blobs shows up here.
    """
    header = emit_header(model)
    declared = {
        name: (POINTER_BYTES if pointer else C_SIZEOF[c_type]) * (int(count) if count else 1)
        for c_type, pointer, name, count in DECLARATION.findall(header)
    }

    n = len(model.layers)
    assert declared.keys() == {
        *(f"model_{kind}{i}" for i in range(n) for kind in ("w", "m", "b")),
        "model_dims",
        "model_w",
        "model_m",
        "model_b",
        "model_inv_sx",
    }
    assert sum(declared.values()) == model.flash_bytes


def test_tanh_table_is_the_one_the_kernel_compiles():
    """The table is written twice, so assert the copies are one table.

    Every other guarantee here is stated against `TANH_LUT`, so a literal in
    `tinyml.h` off by a digit makes all of them about a function the board never
    computes.
    """
    source = (SKETCH / "tinyml.h").read_text()
    body = re.search(r"tinyml_tanh_lut\[[^]]*\] = \{(.*?)\};", source, re.DOTALL)
    assert body, "tinyml.h has no tanh table"

    literals = np.array(
        [np.float32(v) for v in re.findall(r"-?[\d.eE+-]+(?=f)", body.group(1))],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(literals, TANH_LUT)


def test_tanh_table_stays_under_half_an_int8_level():
    """The design constraint: interpolation error cannot move an activation
    by more than one level of the grid it is about to be quantized onto.
    """
    x = np.linspace(-30.0, 30.0, 400_001, dtype=np.float32)
    assert np.abs(tanh_lut(x) - activate(x)).max() < 0.5 / QMAX


def test_tanh_table_is_exact_on_every_knot():
    """The knots are the only inputs where the table is not an approximation.

    `TANH_LUT_SCALE` is `N / 8`, so knot `k` sits at `k / 32`, a dyadic rational whose
    index arithmetic leaves a zero fraction. An off-by-one index, or a scale that is
    not `N / 8`, breaks equality here while leaving the error bound intact.
    """
    assert TANH_LUT.size == TANH_LUT_N + 1
    assert float(TANH_LUT_SCALE) * 8.0 == TANH_LUT_N, "the table no longer spans [0, 8]"

    knots = (np.arange(TANH_LUT_N + 1, dtype=np.float32) / TANH_LUT_SCALE).astype(np.float32)
    np.testing.assert_array_equal(tanh_lut(knots), TANH_LUT)
    np.testing.assert_array_equal(tanh_lut(-knots), -TANH_LUT)

    # Between the knots it is the interpolation, so it stays bracketed by them.
    interpolated = tanh_lut((knots[:-1] + np.float32(0.5) / TANH_LUT_SCALE).astype(np.float32))
    assert np.all(TANH_LUT[:-1] <= interpolated) and np.all(interpolated <= TANH_LUT[1:])
    # Above x ~ 6.8 consecutive knots are one ulp or less apart, so the
    # midpoint has nowhere to land; where float32 can resolve the cell, the
    # interpolation is strictly inside it.
    gap = (TANH_LUT[1:] - TANH_LUT[:-1]).astype(np.float32)
    resolvable = gap > 2 * np.spacing(TANH_LUT[1:])
    assert resolvable.sum() > 200, "the table lost most of its resolution"
    assert np.all(TANH_LUT[:-1][resolvable] < interpolated[resolvable])
    assert np.all(interpolated[resolvable] < TANH_LUT[1:][resolvable])


def test_tanh_table_clamps_its_index_at_the_last_knot():
    """The kernel read `tinyml_tanh_lut[257]` at exactly `|x| == 8.0f`.

    ASAN called it a global-buffer-overflow. It survived because the sweep above steps
    by 1.5e-4 and never lands on 8.0, and because it only tested the Python side. Both
    sides clamp the index now: this pins the Python one, where the missing clamp raises
    IndexError, and `test_c_tanh_matches_the_table_at_and_past_the_edge` pins the C one.

    At and past the edge the value is the last knot: the table saturates rather than
    extrapolating.
    """
    edge = np.float32(np.tanh(np.float32(8.0)))
    assert edge == TANH_LUT[-1], "the last knot is no longer tanh(8)"

    beyond = np.array(
        [8.0, np.nextafter(np.float32(8.0), np.float32(np.inf)), 8.5, 30.0, 1e30, np.inf],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(tanh_lut(beyond), np.full(beyond.shape, edge))
    np.testing.assert_array_equal(tanh_lut(-beyond), np.full(beyond.shape, -edge))

    # One ulp inside: the last cell is flat in float32, so interpolating over
    # it must land on the same knot rather than on anything below it.
    inside = np.array([np.nextafter(np.float32(8.0), np.float32(0.0))], dtype=np.float32)
    np.testing.assert_array_equal(tanh_lut(inside), np.array([edge]))
    np.testing.assert_array_equal(tanh_lut(-inside), np.array([-edge]))


def test_tabulated_tanh_is_lost_in_the_quantization_error(export, model):
    """The bound above permits single-level shifts, so the question is not whether the
    table changes the action (it does) but whether it matters next to the error int8
    already introduces. A fifth of that error is loose enough never to be luck.
    """
    reference = np.stack([float_actor(export)(obs) for obs in export.reference_in])
    tabulated = model(export.reference_in)

    quantization_cost = np.abs(tabulated - reference).mean()
    assert quantization_cost > 0, "a lossless quantizer makes this test vacuous"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(quantize, "tanh_lut", quantize.activate)
        exact = model(export.reference_in)

    assert np.abs(tabulated - exact).mean() < 0.2 * quantization_cost


def build_kernel(model, tmp_path: Path, cflags=CFLAGS) -> Path:
    """Compile the real kernel plus `tests/ckernel/main.c` against `model`."""
    shutil.copy(SKETCH / "tinyml.h", tmp_path / "tinyml.h")
    shutil.copy(HARNESS, tmp_path / "main.c")
    write_header(model, tmp_path / "model.h")

    cc = shutil.which("cc")
    assert cc, "no host C compiler on PATH"
    # S603: a fixed argv list invoking the host compiler, not user input.
    subprocess.run(  # noqa: S603
        [cc, *cflags, *WARNINGS, "main.c", "-o", "ckernel", "-lm"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return tmp_path / "ckernel"


def run_kernel(binary: Path, stdin: str, *args: str, check: bool = True):
    return subprocess.run(  # noqa: S603, argv list, the binary this test just built
        [str(binary), *args], input=stdin, capture_output=True, text=True, check=check, timeout=120
    )


def kernel_actions(binary: Path, frames: np.ndarray) -> np.ndarray:
    stdin = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in frames)
    lines = run_kernel(binary, stdin).stdout.strip().splitlines()
    return np.array([[float(v) for v in line.split()] for line in lines], dtype=np.float32)


def kernel_tanh(binary: Path, x: np.ndarray, check: bool = True):
    """`tinyml_tanh` alone, which the inference path hides behind int8."""
    proc = run_kernel(binary, "\n".join(f"{v:.9g}" for v in x), "tanh", check=check)
    return np.array([np.float32(v) for v in proc.stdout.split()], dtype=np.float32), proc


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")
def test_c_kernel_matches_the_emulator(model, export, tmp_path):
    """Compile `tinyml.h` for the host and diff it against NumPy.

    Exact equality, not a tolerance: integer accumulation is exact and every float step
    is one IEEE-754 float32 operation in the same order on both sides, with no libm call
    in the tanh path. Loosen it deliberately or not at all; `board.self_test` is the
    tolerant comparison, because the device may contract into a VFMA.

    The reference frames are on-policy and leave both clamps dead, so
    `saturating_frames` extends them onto both rails and past the table's domain.
    """
    binary = build_kernel(model, tmp_path)

    extra = saturating_frames(model)
    rails = _quantize(extra, model.layers[0].inv_input_scale)
    assert rails.max() == QMAX and rails.min() == -QMAX, "added frames miss the int8 rails"

    frames = np.concatenate([export.reference_in, extra]).astype(np.float32)
    from_c = kernel_actions(binary, frames)

    assert from_c.shape == (len(frames), ARCH[-1])
    np.testing.assert_array_equal(from_c, model(frames))


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")
def test_c_kernel_matches_the_emulator_at_the_tanh_table_edge(export, tmp_path):
    """The same diff with the activation pinned to exactly `+-8.0f`.

    No observation reaches the edge exactly, the rails capping the accumulator and the
    grid being coarser than a float, so the model puts the edge in layer 0's bias. That
    is the input where the kernel used to read past `tinyml_tanh_lut`.
    """
    edged = edge_model(export)
    layer = edged.layers[0]
    assert np.all(layer.qw == 0), "the accumulator must be zero for bias == pre-activation"
    assert np.any(layer.bias == np.float32(8.0)) and np.any(layer.bias == np.float32(-8.0))

    binary = build_kernel(edged, tmp_path)
    frames = export.reference_in[:8]
    from_c = kernel_actions(binary, frames)

    assert from_c.shape == (len(frames), ARCH[-1])
    assert np.abs(from_c).max() < edged.clip_action, "the action clip is hiding the comparison"
    np.testing.assert_array_equal(from_c, edged(frames))


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")
def test_c_tanh_matches_the_table_at_and_past_the_edge(model, tmp_path):
    """`tinyml_tanh` against `tanh_lut`, value by value, edge included.

    Inference requantizes every activation to int8, so a wrong read near the top of the
    table lands on the same `+-127` and vanishes. Comparing the activation makes the
    edge observable: without the magnitude clamp this segfaults or disagrees, where the
    full-model diff notices nothing.
    """
    binary = build_kernel(model, tmp_path)
    x = tanh_probe()
    from_c, _ = kernel_tanh(binary, x)

    assert from_c.shape == x.shape
    np.testing.assert_array_equal(from_c, tanh_lut(x))


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")
def test_c_tanh_reads_nothing_past_the_end_of_the_table(model, tmp_path):
    """The ASAN run that found the overflow, as a test.

    At exactly `|x| == 8.0f` the fraction is zero, so the out-of-range element is
    multiplied by zero and the value comes out right. The read is the only symptom, and
    only a sanitizer sees it.
    """
    try:
        binary = build_kernel(model, tmp_path, ASAN_CFLAGS)
    except subprocess.CalledProcessError as exc:
        if re.search(r"sanitiz|asan", exc.stderr, re.IGNORECASE):
            pytest.skip("host compiler has no AddressSanitizer runtime")
        raise

    x = tanh_probe()
    from_c, proc = kernel_tanh(binary, x, check=False)

    assert "AddressSanitizer" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, f"the sanitized kernel exited {proc.returncode}"
    np.testing.assert_array_equal(from_c, tanh_lut(x))
