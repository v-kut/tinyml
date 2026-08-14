"""Post-training int8 quantization of the exported actor, and a NumPy emulator of
the arithmetic the board performs.

The definition of the deployed model: `codegen.py` serializes it and
`arduino/deploy/tinyml.h` reimplements it, so the two are read side by side. Scheme
and measurements: docs/findings/quantization.md.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from tinyml_racing.deploy.artifact import ActorExport

# Symmetric range. Allowing -128 makes the grid asymmetric about zero, which
# biases a signed regression output towards the extra level.
QMAX = 127


# (W, b) per layer, `W` shaped (n_in, n_out). Named because `list` is invariant,
# so the producer and the four consumers need one spelling of the pair.
FloatLayers = list[tuple[np.ndarray, np.ndarray]]

# The one activation the pipeline produces and the kernel implements, since SB3's
# trunk is `nn.Tanh` and nothing exposes a choice. `extract_actor` refuses any other
# module; `QuantModel.activation` still carries the name for `Board.verify`.
ACTIVATION = "tanh"


def normalization_scale(export: ActorExport) -> np.ndarray:
    return np.sqrt(export.obs_var + export.epsilon).astype(np.float32)


def clipping_rate(export: ActorExport) -> float:
    """Fraction of calibration elements `VecNormalize` would clip.

    Folding normalization into layer 0 drops that clip, so this is the error budget
    the fold introduces. If it is not ~0 the fold is not free.
    """
    normalized = (export.calibration - export.obs_mean) / normalization_scale(export)
    return float(np.mean(np.abs(normalized) > export.clip_obs))


def float_layers(export: ActorExport, fold_normalization: bool = True) -> FloatLayers:
    """The actor as a list of `(W, b)` with `W` shaped (n_in, n_out).

    With `fold_normalization`, layer 0 absorbs the `VecNormalize` affine
    transform:  `(x - mu)/s @ W + b  ==  x @ (W/s) + (b - (mu/s) @ W)`.
    """
    layers = [(layer.w.astype(np.float32), layer.b.astype(np.float32)) for layer in export.layers]
    if fold_normalization:
        scale = normalization_scale(export)
        kernel, bias = layers[0]
        layers[0] = (
            (kernel / scale[:, None]).astype(np.float32),
            (bias - (export.obs_mean / scale) @ kernel).astype(np.float32),
        )
    return layers


def activate(x: np.ndarray) -> np.ndarray:
    """The reference activation: what PyTorch computed during training."""
    return np.tanh(x, dtype=np.float32)


# Affordable because every activation is requantized to int8 one layer later, where
# the worst interpolation error is far under an LSB. It bounds a shift to one level
# rather than forbidding it, so `test_tabulated_tanh_is_lost_in_the_quantization_error`
# asserts the consequence.
TANH_LUT_N = 256
TANH_LUT_SCALE = np.float32(32.0)  # N / 8.0, a power of two, so every knot is exact
TANH_LUT = np.tanh(np.arange(TANH_LUT_N + 1, dtype=np.float32) / TANH_LUT_SCALE).astype(np.float32)

# Flash the kernel carries for `tanh` and `codegen.py` does not emit: `tinyml.h`
# holds its own copy of the table. Counted by `deployed_flash_bytes`.
TANH_LUT_BYTES = int(TANH_LUT.nbytes)


def tanh_lut(x: np.ndarray) -> np.ndarray:
    """Mirrors `tinyml_tanh` in the C kernel, operation for operation.

    `t` is clamped to N and the index separately to N - 1; clamping only `t` reads one
    past the table at exactly |x| == 8.0.

    NaN is mapped to 0 first, as the kernel does: it survives `np.minimum`, and
    `astype` then yields INT32_MIN and indexes wildly out of bounds, while the board's
    `(int)t` is undefined. One rule on both sides, so a non-finite frame fails rather
    than disagreeing. +-Inf needs no guard: the magnitude clamp saturates it.
    """
    t = np.abs(np.asarray(x, dtype=np.float32)) * TANH_LUT_SCALE
    t = np.minimum(np.where(np.isnan(t), np.float32(0.0), t), TANH_LUT_N)
    i = np.minimum(t.astype(np.int32), TANH_LUT_N - 1)
    f = (t - i.astype(np.float32)).astype(np.float32)
    lo = TANH_LUT[i]
    return np.copysign((lo + f * (TANH_LUT[i + 1] - lo)).astype(np.float32), x)


def float_forward(layers: FloatLayers, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run the float model, returning the output and each layer's input.

    The per-layer inputs are what activation calibration needs, and computing them
    here keeps the ranges with the model the weights came from.
    """
    h = np.atleast_2d(np.asarray(x, dtype=np.float32))
    inputs = []
    for i, (kernel, bias) in enumerate(layers):
        inputs.append(h)
        h = h @ kernel + bias
        if i < len(layers) - 1:
            h = activate(h)
    return h, inputs


def _quantize(x: np.ndarray, inv_scale: float) -> np.ndarray:
    """Float to int8, matching `tinyml_quantize` in the C kernel.

    `np.rint` and C's `lrintf` both round half to even, so the two agree on ties
    rather than differing by a level exactly there.

    NaN is mapped to 0 as the kernel does. Nothing upstream rejects a non-finite
    observation, and a NaN loses both rail comparisons, so without the guard the board
    reaches undefined behaviour while this side casts a NaN, and the two disagree
    silently. +-Inf flows to the rails on both sides.
    """
    s = np.asarray(x, dtype=np.float32) * np.float32(inv_scale)
    s = np.where(np.isnan(s), np.float32(0.0), s)
    return np.clip(np.rint(s), -QMAX, QMAX).astype(np.int8)


@dataclass(frozen=True)
class QuantLayer:
    """One quantized Dense layer, in the layout the C header emits."""

    qw: np.ndarray  # int8, (n_out, n_in), output-major, so the C inner loop is contiguous
    mult: np.ndarray  # float32, (n_out,), weight_scale[j] * input_scale
    bias: np.ndarray  # float32, (n_out,)
    inv_input_scale: float  # float32 reciprocal, so the kernel multiplies rather than divides

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.qw.shape[1]), int(self.qw.shape[0])


@dataclass(frozen=True)
class QuantModel:
    """The deployed model: what `codegen.py` writes and the board runs."""

    layers: tuple[QuantLayer, ...]
    activation: str
    clip_action: float = 1.0

    @property
    def dims(self) -> tuple[int, ...]:
        return (self.layers[0].shape[0], *(layer.shape[1] for layer in self.layers))

    @property
    def arch(self) -> str:
        """`61-16-8-2`. Baked into the header as `MODEL_ARCH` and compared for
        equality by `Board.verify`, so there is exactly one definition.
        """
        return "-".join(str(d) for d in self.dims)

    @property
    def n_in(self) -> int:
        return self.dims[0]

    @property
    def n_out(self) -> int:
        return self.dims[-1]

    @property
    def n_weights(self) -> int:
        return sum(layer.qw.size for layer in self.layers)

    @property
    def flash_bytes(self) -> int:
        """Bytes of `.rodata` the generated header adds, and only the header.

        Every object `emit_header` declares, not just the weights: the file-scope index
        is another ~60 bytes. Padding belongs to the linker, and the kernel's own
        constants are `deployed_flash_bytes`.
        """
        n = len(self.layers)
        blobs = sum(
            layer.qw.size + 4 * layer.mult.size + 4 * layer.bias.size for layer in self.layers
        )
        dims = 2 * (n + 1)  # uint16_t model_dims[n + 1]
        tables = 3 * 4 * n  # model_w, model_m, model_b, pointers, 4 bytes on Cortex-M
        inv_sx = 4 * n  # float model_inv_sx[n]
        return blobs + dims + tables + inv_sx

    @property
    def deployed_flash_bytes(self) -> int:
        """What the deployed net costs in flash: the header plus `tinyml.h`'s tanh table.
        The manifest and the report quote this one, because a budget wrong in the
        optimistic direction is the one that overflows.
        """
        return self.flash_bytes + TANH_LUT_BYTES

    def digest(self) -> int:
        """CRC32 over every constant in the model. The board reports it at handshake, so a
        host talking to last week's weights fails immediately.
        """
        crc = 0
        for layer in self.layers:
            for blob in (
                layer.qw.astype(np.int8).tobytes(),
                layer.mult.astype("<f4").tobytes(),
                layer.bias.astype("<f4").tobytes(),
                np.float32(layer.inv_input_scale).astype("<f4").tobytes(),
            ):
                crc = zlib.crc32(blob, crc)
        return crc & 0xFFFFFFFF

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Emulate the device bit-for-bit, one batch of raw observations in.

        Not the float model: `tanh_lut` approximates `activate`, and `evaluate.py`
        scores both because the gap is the point. The device may still contract a
        multiply-add into a VFMA, which GCC takes at -O3 in both float expressions
        `tinyml.h` evaluates, worth a last ulp. Hence `board.self_test` compares with
        a tolerance while `tests/ckernel` compares for equality, its x86 build having
        no FMA to contract into.
        """
        h = np.atleast_2d(np.asarray(x, dtype=np.float32))
        for i, layer in enumerate(self.layers):
            q = _quantize(h, layer.inv_input_scale).astype(np.int32)
            acc = q @ layer.qw.astype(np.int32).T  # int32, exact
            h = (acc.astype(np.float32) * layer.mult + layer.bias).astype(np.float32)
            if i < len(self.layers) - 1:
                h = tanh_lut(h)
        return np.clip(h, -self.clip_action, self.clip_action)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Single-observation call, shaped like `Snapshot.act`."""
        return self(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0]


def quantize_model(export: ActorExport) -> QuantModel:
    """Post-training quantization of an exported actor.

    Activation ranges come from the on-policy observation set the exporter
    collected, so they cover the distribution the deployed car visits.
    """
    if export.activation != ACTIVATION:
        raise ValueError(
            f"unsupported activation {export.activation!r}; the deployed kernel "
            f"implements {ACTIVATION!r} only"
        )

    layers = float_layers(export)
    calib = np.asarray(export.calibration, dtype=np.float32)

    _, layer_inputs = float_forward(layers, calib)

    quantized = []
    for (kernel, bias), acts in zip(layers, layer_inputs, strict=True):
        # Per-output-channel weight scale; a dead channel (all-zero column) would
        # otherwise divide by zero and emit NaN weights.
        w_absmax = np.abs(kernel).max(axis=0)
        w_scale = np.where(w_absmax > 0, w_absmax / QMAX, np.float32(1.0)).astype(np.float32)
        qw = np.clip(np.rint(kernel / w_scale), -QMAX, QMAX).astype(np.int8)

        a_absmax = float(np.abs(acts).max())
        a_scale = np.float32(a_absmax / QMAX if a_absmax > 0 else 1.0)

        quantized.append(
            QuantLayer(
                qw=np.ascontiguousarray(qw.T),  # (n_out, n_in)
                mult=(w_scale * a_scale).astype(np.float32),
                bias=bias.astype(np.float32),
                inv_input_scale=float(np.float32(1.0) / a_scale),
            )
        )

    return QuantModel(layers=tuple(quantized), activation=ACTIVATION)


# The shipped head is (steer, throttle) and the report reads better for saying so.
# Everything else is generic in the output width, so a wider head gets numbered
# channels rather than an `IndexError` inside `evaluate_run`.
_ACTION_CHANNELS = ("steer", "throttle")


def action_error(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    err = np.abs(np.asarray(predicted, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    err = err.reshape(len(err), -1)
    names = (
        _ACTION_CHANNELS
        if err.shape[1] == len(_ACTION_CHANNELS)
        else tuple(f"ch{i}" for i in range(err.shape[1]))
    )
    return {
        "mae": float(err.mean()),
        "max": float(err.max()),
        **{f"{name}_mae": float(err[:, i].mean()) for i, name in enumerate(names)},
    }


def float_actor(export: ActorExport, clip_action: float = QuantModel.clip_action):
    """The unquantized model as a callable, for the float32 baseline row.

    `clip_action` is the deployed model's own bound, passed by `evaluate_run` off the
    model it compares against, because the two rows measure quantization error only if
    the baseline saturates where the deployed model does. The default is also the
    bound `onnx_export` bakes into `Clip`.
    """
    layers = float_layers(export)

    def act(obs: np.ndarray) -> np.ndarray:
        out, _ = float_forward(layers, np.asarray(obs, dtype=np.float32).reshape(1, -1))
        return np.clip(out[0], -clip_action, clip_action)

    return act
