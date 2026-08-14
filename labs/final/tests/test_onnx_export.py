"""`actor.onnx` is the artifact this project hands to anyone who is not this
project: it is read by runtimes that never see `quantize.py` and cannot ask it
what the weights meant. A graph that loads, runs, and is wrong by a
normalization fold or a missing output clip looks perfectly healthy from the
outside, so these tests pin the graph to `quantize.float_actor` numerically, pin
the io contract other tools bind to by name, and pin the refusal that keeps an
export the deployed kernel cannot run out of the file.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import onnx
import pytest
from conftest import make_export

from tinyml_racing.deploy.artifact import EXPORT_VERSION, ActorExport
from tinyml_racing.deploy.onnx_export import build_model, onnx_error, write_onnx
from tinyml_racing.deploy.quantize import float_actor, float_forward, float_layers

ARCH = (6, 4, 2)
TOLERANCE = 1e-6


def onnx_export() -> ActorExport:
    """The shared actor with `reference_out` set to `float_actor`'s own output,
    which is what makes `onnx_error` a measurement of the exporter rather than of
    the policy. The over-scaled head keeps part of it clipped.
    """
    export = make_export(ARCH, head_scale=4.0)
    return replace(export, reference_out=float_reference(export))


def float_reference(export: ActorExport) -> np.ndarray:
    """`float_actor` is single-observation; the graph is batched."""
    act = float_actor(export)
    return np.stack([act(obs) for obs in export.reference_in]).astype(np.float32)


@pytest.fixture
def export() -> ActorExport:
    return onnx_export()


def test_graph_matches_the_float_baseline(export, tmp_path):
    path = write_onnx(export, tmp_path / "actor.onnx")

    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (predicted,) = session.run(["action"], {"obs": np.asarray(export.reference_in)})

    reference = float_reference(export)
    # Guards the guard: the comparison below only covers the output Clip while
    # the fixture keeps driving the head past the action bounds.
    unclipped, _ = float_forward(float_layers(export), export.reference_in)
    assert (np.abs(unclipped) > 1.0).any(), "fixture no longer exercises the clip"
    assert np.abs(reference).max() <= 1.0

    max_abs = float(np.abs(predicted - reference).max())
    assert max_abs < TOLERANCE, f"max abs deviation {max_abs:g}"


def test_onnx_error_is_near_zero_against_its_own_reference(export, tmp_path):
    """The same shape of check `build.py` runs on the real artifact."""
    error = onnx_error(write_onnx(export, tmp_path / "actor.onnx"), export)

    assert set(error) == {"mae", "max", "steer_mae", "throttle_mae"}
    assert error["max"] < TOLERANCE, error


def test_io_contract_is_named_and_batched(export, tmp_path):
    """Names, dtypes and the dynamic batch dim are what external tools bind to."""
    model = onnx.load(write_onnx(export, tmp_path / "actor.onnx"))
    (obs,), (action,) = model.graph.input, model.graph.output

    assert (obs.name, action.name) == ("obs", "action")
    for value, width in ((obs, ARCH[0]), (action, ARCH[-1])):
        assert value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, value.name
        batch, features = value.type.tensor_type.shape.dim
        assert batch.dim_param, f"{value.name}: batch dim is fixed at {batch.dim_value}"
        assert features.dim_value == width, value.name


def test_metadata_records_the_provenance(export, tmp_path):
    """An ONNX outliving this checkout should still say what produced it."""
    model = onnx.load(write_onnx(export, tmp_path / "actor.onnx", provenance="run test"))
    props = {entry.key: entry.value for entry in model.metadata_props}

    assert model.producer_name == "tinyml-racing"
    assert model.doc_string == "run test"
    assert props["export_version"] == str(EXPORT_VERSION)
    assert props["num_timesteps"] == str(export.num_timesteps)
    assert props["activation"] == export.activation


def test_activation_the_kernel_cannot_run_is_refused(export):
    """`relu` is a fine network and not the one that ships: the graph is the
    portable record of the *deployed* function, so it may not quietly say Tanh
    over an export that claims something else.
    """
    with pytest.raises(ValueError, match="relu"):
        build_model(replace(export, activation="relu"))
