"""Serialize an `ActorExport` into `actor.onnx`: the portable float32 record of the
deployed function, and the reference the int8 header is measured against.

Built by hand with `onnx.helper` so every reader of `actor.npz` stays torch-free.
It takes RAW observations, the `VecNormalize` affine folded into layer 0, its
`clip_obs` not (see `quantize.clipping_rate`), and clips the action to [-1, 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tinyml_racing.deploy.artifact import EXPORT_VERSION, ActorExport
from tinyml_racing.deploy.quantize import ACTIVATION, QuantModel, action_error, float_layers

__all__ = ["OPSET", "onnx_error", "write_onnx"]

# Pinned rather than "whatever this onnx build defaults to": the opset is part
# of the artifact's contract, and 17 is old enough that every current runtime
# reads it while covering every op below.
OPSET = 17


def build_model(export: ActorExport, provenance: str = "") -> onnx.ModelProto:
    """The graph `float_actor` computes: `Gemm` per layer, tanh between them,
    `Clip` to the action bounds.

    One `Gemm` rather than `MatMul` + `Add` because a dense layer is one node in
    the format's own vocabulary, and `W` already lies (n_in, n_out), the
    transpose happened once in `export.extract_actor`, so there is nothing to
    keep straight. `Tanh` is the only activation the pipeline produces; the
    guard is here because `build_model` takes an export, not a `QuantModel`,
    and a graph that silently disagreed with `model.h` is worse than a refusal.
    """
    if export.activation != ACTIVATION:
        raise ValueError(
            f"unsupported activation {export.activation!r} for ONNX export; "
            f"the deployed function evaluates {ACTIVATION!r}"
        )
    layers = float_layers(export)
    n_in, n_out = int(layers[0][0].shape[0]), int(layers[-1][0].shape[1])

    # 0-d arrays, not numpy scalars: opset 17 wants `Clip`'s bounds as rank-0
    # tensors, and `np.float32(x).reshape(())` returns a scalar that only
    # happens to serialize like one.
    tensors: dict[str, np.ndarray] = {
        "action_min": np.array(-QuantModel.clip_action, dtype=np.float32),
        "action_max": np.array(QuantModel.clip_action, dtype=np.float32),
    }
    nodes, h = [], "obs"
    for i, (kernel, bias) in enumerate(layers):
        tensors[f"w{i}"], tensors[f"b{i}"] = kernel, bias
        nodes.append(helper.make_node("Gemm", [h, f"w{i}", f"b{i}"], [f"z{i}"], f"layer{i}"))
        h = f"z{i}"
        # Between layers, never after the last one: the output head is linear.
        if i < len(layers) - 1:
            nodes.append(helper.make_node("Tanh", [h], [f"h{i}"], f"act{i}"))
            h = f"h{i}"
    nodes.append(helper.make_node("Clip", [h, "action_min", "action_max"], ["action"], "clip"))

    # A named batch dim, not a fixed 1: the artifact is for offline evaluation
    # over a reference set as much as for single-step inference.
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "actor",
            [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", n_in])],
            [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch", n_out])],
            initializer=[
                numpy_helper.from_array(np.asarray(v, dtype=np.float32), k)
                for k, v in tensors.items()
            ],
        ),
        producer_name="tinyml-racing",
        opset_imports=[helper.make_operatorsetid("", OPSET)],
    )
    model.doc_string = provenance
    # The same provenance `codegen.py` puts in the header comment, but in fields
    # a loader can read: an ONNX that outlives this checkout should still say
    # which schema and which run it came from.
    helper.set_model_props(
        model,
        {
            "export_version": str(EXPORT_VERSION),
            "num_timesteps": str(export.num_timesteps),
            "activation": export.activation,
            "shape": export.shape(),
            "normalization": "folded",
        },
    )
    # Fail here, where the graph was built and the mistake is legible, rather
    # than in whatever runtime loads it next month.
    onnx.checker.check_model(model, full_check=True)
    return model


def write_onnx(export: ActorExport, path: str | Path, provenance: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(build_model(export, provenance), path)
    return path


def onnx_error(path: str | Path, export: ActorExport) -> dict[str, float]:
    """Run the written graph over `reference_in` and score it like any other row.

    Imported here rather than at module scope: `onnxruntime` loads a native runtime
    and its thread pools, and the build only writes the file. Verifying it is
    optional.
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    obs = np.asarray(export.reference_in, dtype=np.float32)
    # `session.run` is typed as returning any of onnxruntime's output kinds; this
    # graph has one dense float output, so it is narrowed here rather than left
    # as a union the error function would have to accept.
    (predicted,) = session.run(["action"], {"obs": obs})
    return action_error(np.asarray(predicted, dtype=np.float32), export.reference_out)
