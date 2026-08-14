"""`actor.npz` is the training -> deployment boundary: everything after it
(quantization, codegen, the board handshake) trusts the file's layout without
re-deriving it from the policy. A file that loads but means something else is the
one failure the pipeline cannot notice, it quantizes, compiles, flashes, and
only the car disagrees. Hence the version stamp, and hence these tests.
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from tinyml_racing.deploy.artifact import EXPORT_VERSION, ActorExport


def test_roundtrip_preserves_every_field(export, tmp_path):
    """Enumerated from the dataclass, so a new field cannot silently skip disk.

    `clip_obs` and `epsilon` cross as float32, the same width the device
    computes in, so the roundtrip is compared at float32, not at the width the
    Python literal happened to have.
    """
    loaded = ActorExport.load(export.save(tmp_path / "actor.npz"))

    for field in fields(ActorExport):
        original, restored = getattr(export, field.name), getattr(loaded, field.name)
        if field.name == "layers":
            assert len(restored) == len(original)
            for i, (before, after) in enumerate(zip(original, restored, strict=True)):
                np.testing.assert_array_equal(after.w, before.w, err_msg=f"w{i}")
                np.testing.assert_array_equal(after.b, before.b, err_msg=f"b{i}")
        elif isinstance(original, np.ndarray):
            np.testing.assert_array_equal(restored, original, err_msg=field.name)
        elif isinstance(original, float):
            assert restored == float(np.float32(original)), field.name
        else:
            assert restored == original, field.name


def test_saved_file_carries_the_current_version(export, tmp_path):
    with np.load(export.save(tmp_path / "actor.npz")) as data:
        assert int(data["version"]) == EXPORT_VERSION


def test_future_version_is_rejected(export, tmp_path):
    path = replace(export, version=EXPORT_VERSION + 1).save(tmp_path / "actor.npz")
    with pytest.raises(ValueError, match="version"):
        ActorExport.load(path)


def test_unstamped_file_is_rejected(export, tmp_path):
    """The pre-version layout: readable npz, unknown provenance."""
    path = export.save(tmp_path / "actor.npz")
    with np.load(path) as data:
        payload = {k: data[k] for k in data.files if k != "version"}
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="version"):
        ActorExport.load(path)
