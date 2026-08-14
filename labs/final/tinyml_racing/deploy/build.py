"""Compile a trained run into something an Arduino Nano 33 BLE can run.

    tinyml-build [--reuse-export] [--port auto] [--flash]

export, quantize, codegen, evaluate, compile. It drives before it flashes, because
per-step quantization error either washes out or integrates into a wall.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tinyml_racing import progress
from tinyml_racing.deploy.artifact import ActorExport
from tinyml_racing.deploy.codegen import write_header
from tinyml_racing.deploy.evaluate import evaluate_run, format_table, write_report
from tinyml_racing.deploy.export import export_actor
from tinyml_racing.deploy.manifest import bundle, write_manifest
from tinyml_racing.deploy.onnx_export import onnx_error, write_onnx
from tinyml_racing.deploy.quantize import float_actor, quantize_model
from tinyml_racing.utils import Run, setup_logging

SKETCH_DIR = Path(__file__).resolve().parents[2] / "arduino" / "deploy"
FQBN = "arduino:mbed_nano:nano33ble"

# The core builds at -Os, right for the mbed runtime, wrong for a fixed-shape MAC
# loop. No fast-math: the float rules stay IEEE-exact because the kernel is held to
# bit-exact agreement with the NumPy emulator. No newer `-std` either, since the core's
# `Arduino.h` defines `abs(x)` as a macro that collides with libstdc++ from C++17 on.
SKETCH_FLAGS = "-O3 -funroll-loops"

# Compiled by the system `arm-none-eabi-g++`, not the mbed core's 2017 gcc 7, whose
# `arm_acle.h` neither declares the DSP intrinsics `tinyml.h` uses nor compiles as C++.
# This TU is the check: exact, and cheaper than a version table. The core's prebuilt
# libmbed.a stays gcc 7, so a link failure means the ABIs disagree.
ACLE_PROBE = """#include <arm_acle.h>
int probe(unsigned v, int acc) {
  return __smlad(__sxtb16(v), __sxtb16(__ror(v, 8)), acc);
}
"""

# A real disagreement between the two writers misses by 0.1 or more, where float32
# summation order moves at most 2e-4, well under one int8 LSB. Scored against
# `float_actor`, which has no clip_obs fold.
ONNX_TOLERANCE = 1e-3

logger = logging.getLogger(__name__)


def build(
    run: Run,
    n_tracks: int = 8,
    max_steps: int | None = None,
    reuse_export: bool = False,
    port: str | None = None,
    clean_sensor: bool = True,
) -> dict[str, Any]:
    if not (reuse_export and run.actor_npz.is_file()):
        export_actor(run)

    with progress.step("quantize") as bar:
        export = ActorExport.load(run.actor_npz)
        model = quantize_model(export)
        provenance = f"run {run.name}, {export.num_timesteps:,} PPO steps"
        bar.advance(0.0, note=f"{model.arch} {model.activation}, {model.deployed_flash_bytes} B")

    # Written beside their final names and promoted in `record`: the ONNX gate can
    # still reject this build, and until the report describes it the artifacts must
    # stay the pair the surviving manifest hashes.
    header_tmp = run.header.with_name(run.header.name + ".tmp")
    onnx_tmp = run.actor_onnx.with_name(run.actor_onnx.name + ".tmp")
    try:
        with progress.step("artifacts", note="model.h, actor.onnx") as bar:
            write_header(model, header_tmp, provenance=provenance)
            write_onnx(export, onnx_tmp, provenance=provenance)

            # The same references the board replays over USB, against the portable
            # graph: an artifact nobody checked is an artifact nobody can trust.
            float_act = float_actor(export)
            float_out = np.stack([float_act(obs) for obs in export.reference_in]).astype(np.float32)
            onnx_err = onnx_error(onnx_tmp, replace(export, reference_out=float_out))
            bar.advance(0.0, note=f"onnx vs float32 {onnx_err['max']:.1e}")
        if onnx_err["max"] > ONNX_TOLERANCE:
            raise ValueError(
                f"{run.actor_onnx} disagrees with the float32 actor by "
                f"{onnx_err['max']:.3e} (tolerance {ONNX_TOLERANCE:.0e})"
            )

        board = None
        if port is not None:
            # Opened here, not by the caller: `Board.open` verifies the flashed
            # digest against a freshly quantized `actor.npz`, and only from this
            # line on does that file describe `run.snapshot`.
            from tinyml_racing.deploy.board import Board

            board = Board.open(None if port == "auto" else port, run)

        try:
            results = evaluate_run(
                run,
                n_tracks=n_tracks,
                max_steps=max_steps,
                board=board,
                clean_sensor=clean_sensor,
            )
        finally:
            if board is not None:
                board.close()

        results["digest"] = f"0x{model.digest():08x}"
        results["onnx_vs_float_max"] = onnx_err["max"]
        # One stage, in dependency order: `Path.replace` promotes each artifact
        # atomically within `artifacts/`, and the manifest is written last,
        # after the header, the graph and the report it hashes.
        with progress.step("record", note="model.h, actor.onnx, report.json, manifest.json"):
            header_tmp.replace(run.header)
            onnx_tmp.replace(run.actor_onnx)
            write_report(results, run)
            write_manifest(run, export, model, results)
    finally:
        # A stage that raised leaves no half-written sibling in `artifacts/` for
        # the next build, or for a reader listing the directory, to trip over.
        header_tmp.unlink(missing_ok=True)
        onnx_tmp.unlink(missing_ok=True)
    return results


@contextmanager
def update_sketch(run: Run) -> Iterator[Path]:
    """Point `arduino/deploy/model.h` at this run for the body of the block.

    `arduino-cli` compiles the sketch directory in place, so the tracked header is
    replaced before `compile_sketch` runs and only stays replaced if the block
    succeeds. The tracked sketch is never on a model that was never built.
    """
    dest = SKETCH_DIR / "model.h"
    previous = dest.read_bytes() if dest.is_file() else None
    shutil.copyfile(run.header, dest)
    compiled = False
    try:
        yield dest
        compiled = True
    finally:
        if not compiled:
            if previous is None:
                dest.unlink(missing_ok=True)
            else:
                dest.write_bytes(previous)


def toolchain_prefix() -> str:
    """The `compiler.path` the sketch must be built with, or a refusal saying why.

    arduino-cli concatenates this prefix onto the tool names, so keep the
    trailing separator.
    """
    gxx = shutil.which("arm-none-eabi-g++")
    if gxx is None:
        raise RuntimeError(
            "arm-none-eabi-g++ is not on PATH. The sketch is built with a current "
            "arm-none-eabi toolchain, not the gcc 7.2 the mbed core ships: install "
            "one from your distribution (Arch: arm-none-eabi-gcc)."
        )
    probe = subprocess.run(  # noqa: S603
        [gxx, "-x", "c++", "-mcpu=cortex-m4", "-fsyntax-only", "-"],
        input=ACLE_PROBE,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"{gxx} cannot compile the ACLE intrinsics `tinyml.h` needs "
            f"(__smlad/__sxtb16/__ror via <arm_acle.h>); it is too old for this "
            f"sketch. Install a current arm-none-eabi toolchain.\n{probe.stderr}"
        )
    return f"{Path(gxx).parent}{os.sep}"


def compile_sketch(fqbn: str = FQBN) -> str:
    """`arduino-cli compile`, returning its output. Needs no hardware.

    Its size lines are the flash/RAM figures `main` prints. They do not reach
    `report.json`, which the manifest has already hashed.
    """
    return _arduino_cli(
        [
            "compile",
            "--fqbn",
            fqbn,
            "--build-property",
            f"compiler.path={toolchain_prefix()}",
            "--build-property",
            f"compiler.cpp.extra_flags={SKETCH_FLAGS}",
            str(SKETCH_DIR),
        ]
    )


def _arduino_cli(args: list[str]) -> str:
    exe = shutil.which("arduino-cli")
    if exe is None:
        raise RuntimeError(
            "arduino-cli is not on PATH. Install it, then: "
            "arduino-cli core install arduino:mbed_nano"
        )
    # `check=False`: the non-zero path is reported below with the tool's own
    # output, which `CalledProcessError` would swallow.
    proc = subprocess.run([exe, *args], capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(f"arduino-cli {args[0]} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout + proc.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--n-tracks", type=int, default=8)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="rollout cap in steps (default: the run's own truncation limit)",
    )
    parser.add_argument(
        "--reuse-export", action="store_true", help="reuse artifacts/actor.npz if it exists"
    )
    parser.add_argument(
        "--port", metavar="PORT", help="also score the flashed board ('auto' probes for one)"
    )
    parser.add_argument(
        "--sensor",
        choices=("clean", "noisy"),
        default="clean",
        help="detector the held-out laps drive behind (noisy = the training sensor model)",
    )
    parser.add_argument(
        "--flash",
        nargs="?",
        const="auto",
        default=None,
        metavar="PORT",
        help="upload to the board after compiling ('auto' detects the port)",
    )
    args = parser.parse_args()

    try:
        run = Run.resolve(args.run_dir)
        # One live display for the pipeline: stages announce themselves, log lines
        # print above them, and every number is printed after it closes, so the summary
        # is plain text a report can quote.
        with progress.session(skip=False) as reporter:
            setup_logging(console=reporter.console)
            logger.info("building %s from %s", run.name, run.snapshot)
            results = build(
                run,
                args.n_tracks,
                args.max_steps,
                args.reuse_export,
                args.port,
                clean_sensor=args.sensor == "clean",
            )
            # The tracked header is replaced for the compile and rolled back if
            # `arduino-cli` rejects it, so `arduino/deploy/model.h` and the last
            # successful compile always name the same model.
            with update_sketch(run) as sketch, progress.step("compile", note="arduino-cli"):
                sizes = [
                    line.strip()
                    for line in compile_sketch().splitlines()
                    if "Sketch uses" in line or "Global variables" in line
                ]
            uploaded = None
            if args.flash:
                # Imported here, not at module scope: a build without --flash
                # needs no serial port and no pyserial.
                from tinyml_racing.deploy.board import find_port

                uploaded = find_port() if args.flash == "auto" else args.flash
                with progress.step("upload", note=uploaded):
                    _arduino_cli(["upload", "-p", uploaded, "--fqbn", FQBN, str(SKETCH_DIR)])
    # `ProtocolError` is a `RuntimeError`, so a stale flash lands here too.
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        progress.fail(exc)

    print(
        f"actor {results['arch']} ({results['activation']}) "
        f"@ {results['num_timesteps']:,} steps, digest {results['digest']}"
    )
    print(f"VecNormalize clipping rate on calibration set: {results['clipping_rate']:.3%}")
    print(f"onnx vs float32 export: {results['onnx_vs_float_max']:.2e} max action error\n")
    print(format_table(results))

    print(f"\n{run.artifacts}/")
    for entry in bundle(run):
        print(f"  {entry['name']:<14s} {entry['bytes'] / 1024:8.1f} KB  {entry['what']}")
    print(f"  {run.manifest.name:<14s} {run.manifest.stat().st_size / 1024:8.1f} KB  index")
    print(f"\ncompiled from {sketch}")
    for line in sizes:
        print(line)
    if uploaded is not None:
        print(f"\nflashed {uploaded}. next: tinyml-board --run-dir {run.root}")


if __name__ == "__main__":
    main()
