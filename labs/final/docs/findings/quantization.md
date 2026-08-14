# The int8 scheme, and reading the deployment numbers

## Weights per channel, activations per tensor

Weights are int8 with one scale per output channel, `|W|.max(axis=0) / 127`. Per channel
because the first layer's columns differ by an order of magnitude in norm once observation
normalization is folded in, and one tensor-wide scale would quantize the small columns to
zero. Cost: one float per output channel.

Activations are int8 with one scale per tensor, calibrated on the policy's own rollout
states. Calibrating on noise or an earlier checkpoint sets ranges for a distribution the
deployed car never visits, and the clipping appears as biased steering rather than an
obvious failure. Dequantization is one `mult = w_scale * a_scale` on the int32 accumulator,
with the input scale stored as a reciprocal so the kernel multiplies.

The split follows Krishnamoorthi's whitepaper (`../README.md`).

## One activation

The pipeline produces tanh only, SB3's trunk being `nn.Tanh` with nothing exposing a
choice. Three places used to dispatch on a name, each carrying untested `relu`, `elu` and
`linear` branches no configuration could reach.

They name it once now, `quantize.ACTIVATION`, and every boundary refuses rather than falls
back: `extract_actor` rejects a trunk module that is not `Tanh`, `quantize_model` and
`build_model` reject an export claiming otherwise, `tinyml.h`'s `#if` rejects a `model.h`
that does.

## The evaluation is the gate

Everything before it can succeed while producing a model that drives into a wall:
quantization error is per-step and small, and closed-loop control is where small errors
either wash out or integrate. So the criterion is laps and reward over held-out layouts,
not action-space MAE, and every variant drives the same seeds from `EVAL_SEED_RANGE`.

## Reading the error figures

| figure | for |
| --- | --- |
| board vs host emulator | correctness: the only check that catches `tinyml.h` and `quantize.py` diverging |
| int8 vs float32 / PyTorch | the real cost of quantization, and the number to quote |
| ONNX vs the export | that the portable graph is the same function |
| VecNormalize clipping on calibration | that `CLIP_OBS` is not truncating the calibration states |

Every comparison but the first is the host checking itself. Device timings are bracketed
for the same reason: `us_infer` covers the kernel alone, `us_read` the read plus checksum
verify, so figures from before that split do not compare with anything since.

## Flash accounting is pessimistic on purpose

The reported comparison is `deployed_flash_bytes`, codegen's constants plus the tanh table,
against the float32 weights; the header alone is the figure that flatters int8. A float32
baseline would also need libm's `tanhf`, so the comparison is arguable either way, and the
point is that it is qualified where it is produced. The smaller the net, the worse int8
looks, the table being a fixed cost the weights no longer dwarf.

## ONNX is a record, not a path

The car runs `model.h`. `actor.onnx` is the one file another toolchain can open, and the
build refuses to finish unless it reproduces the export on the reference vectors the board
replays. Built by hand with `onnx.helper` so nothing downstream of `export.py` needs torch:
one `Gemm` per layer, `Tanh` between, `Clip` on the head, and a named batch dimension.
