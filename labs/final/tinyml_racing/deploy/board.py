"""Talk to the Arduino Nano 33 BLE over USB: the host half of `arduino/deploy/link.h`, plus a
handshake that refuses to run against a board flashed with different weights than
the ones on disk.

`self_test` replays the exported reference observations through the hardware, the only
check that can catch a `tinyml.h`/`quantize.py` divergence.
"""

from __future__ import annotations

import argparse
import operator
import struct
import time
from dataclasses import asdict, dataclass
from functools import reduce
from typing import Any

import numpy as np
import serial
from serial.tools import list_ports

from tinyml_racing import progress
from tinyml_racing.deploy.artifact import ActorExport
from tinyml_racing.deploy.quantize import QuantModel, quantize_model
from tinyml_racing.utils import Run, setup_logging

ARDUINO_VIDS = (0x2341, 0x2A03)
BAUD = 500000

# The nRF52840's full-speed CDC bulk endpoint size; see `Board._write_paced`.
USB_PACKET = 64
USB_FRAME_S = 0.00005

# What `link.h`'s `reject()` puts in the error frame instead of a byte count. A
# count alone cannot distinguish "everything arrived and the xor was wrong" from
# "the checksum byte never arrived": both report `got == want`. No genuine count
# can collide, since it is bounded by the payload width `4 * n_in`.
REJECT_NO_CHECKSUM = 0xFFFE
REJECT_BAD_CHECKSUM = 0xFFFF

# 'E' + uint16 count, the whole error frame.
ERROR_FRAME = 3


class ProtocolError(RuntimeError):
    pass


def xor8(data: bytes) -> int:
    # `int(...)`: `operator.xor` is overloaded over anything with `__xor__`, so the
    # fold's static type is only as narrow as that. Over `bytes` it is always an int.
    return int(reduce(operator.xor, data, 0))


def find_port() -> str:
    ports = [p for p in list_ports.comports() if p.vid in ARDUINO_VIDS]
    if not ports:
        seen = ", ".join(f"{p.device} ({p.description})" for p in list_ports.comports()) or "none"
        raise RuntimeError(f"no Arduino found; serial ports present: {seen}")
    if len(ports) > 1:
        devices = ", ".join(p.device for p in ports)
        raise RuntimeError(f"several Arduinos connected ({devices}); pass --port")
    return str(ports[0].device)


@dataclass(frozen=True)
class BoardIdentity:
    arch: str
    act: str
    n_in: int
    n_out: int
    digest: int

    @classmethod
    def parse(cls, line: str) -> BoardIdentity:
        """Parse `link.h`'s `IDENTITY` line. Every failure is a `ProtocolError`."""
        if not line.startswith("tinyml"):
            raise ProtocolError(
                f"the board did not answer the handshake (got {line!r}); is the "
                "sketch flashed, and is another program holding the port?"
            )
        fields = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
        missing = {"arch", "act", "n_in", "n_out", "digest"} - fields.keys()
        if missing:
            raise ProtocolError(
                f"handshake line is missing {sorted(missing)} (got {line!r}); the board "
                "is running a different link.h than this host"
            )
        try:
            return cls(
                arch=fields["arch"],
                act=fields["act"],
                n_in=int(fields["n_in"]),
                n_out=int(fields["n_out"]),
                # The digest is stringified straight out of `model.h`, so it
                # still wears the `u` of the C literal it is over there.
                digest=int(fields["digest"].rstrip("uU"), 16),
            )
        except ValueError as exc:
            raise ProtocolError(f"unreadable handshake line {line!r}: {exc}") from exc


class Board:
    def __init__(self, port: str, n_in: int, n_out: int, timeout: float = 2.0):
        self.port = port
        self.model: QuantModel | None = None
        self.last_round_trip_ms = 0.0
        # What the device reported for `tinyml_infer` alone, us. The round trip
        # above is what a host control loop actually pays for it.
        self.last_infer_us = 0

        self._req = struct.Struct(f"<{int(n_in)}f")
        # Reply from `link.h`: n_out float32 actions, then the board's own two
        # uint16 timers (us_read, us_infer).
        self._resp = struct.Struct(f"<{int(n_out)}fHH")
        self._serial = serial.Serial(port, BAUD, timeout=timeout, write_timeout=timeout)
        # Everything after the open can fail, `_identify` raises `ProtocolError`
        # on the stale-flash case, and `Serial` has no `__del__`, so a
        # half-constructed board would leak the fd and hold DTR (the Nano's
        # reset line) asserted for the life of the process.
        try:
            time.sleep(1.5)  # opening the port resets the Nano; wait out the reboot
            self._serial.reset_input_buffer()
            self.identity = self._identify()
        except BaseException:
            self._serial.close()
            raise

    @classmethod
    def open(cls, port: str | None, run: Run) -> Board:
        model = quantize_model(ActorExport.load(run.actor_npz))
        board = cls(port or find_port(), n_in=model.n_in, n_out=model.n_out)
        # Same reason as in `__init__`: a digest mismatch is the expected outcome
        # of pointing --run-dir at the wrong run, and it must not cost the port.
        try:
            board.verify(model)
        except BaseException:
            board.close()
            raise
        board.model = model
        return board

    def _identify(self) -> BoardIdentity:
        self._serial.write(b"?")
        self._serial.flush()
        return BoardIdentity.parse(self._serial.readline().decode("ascii", "replace").strip())

    def verify(self, model: QuantModel) -> None:
        got = self.identity
        problems = [
            f"{name}: board={a!r} expected={b!r}"
            for name, a, b in (
                # `arch` is `MODEL_ARCH`, i.e. `QuantModel.arch`. The digest
                # would catch a changed hidden width too, but only as an opaque
                # hex mismatch; this line names it.
                ("arch", got.arch, model.arch),
                ("n_in", got.n_in, model.n_in),
                ("n_out", got.n_out, model.n_out),
                ("act", got.act, model.activation),
                ("digest", f"0x{got.digest:08x}", f"0x{model.digest():08x}"),
            )
            if a != b
        ]
        if problems:
            raise ProtocolError(
                f"board is running a different model ({'; '.join(problems)}). "
                "Re-flash it with `tinyml-build --flash`, or point --run-dir at "
                "the run it was flashed from."
            )

    def _write_paced(self, frame: bytes) -> None:
        """Write one endpoint-sized packet at a time, 50 us apart.

        `USB_PACKET` is the device's own bulk endpoint size, so a request of
        `2 + 4 * n_in` bytes leaves as ceil(len / 64) writes with a gap between
        them (246 bytes -> 64+64+64+54 at the shipped n_in = 61) and no single
        write can outrun the core's 256-byte ring however far behind the sketch
        is. The gap busy-waits because 50 us is below the scheduler's resolution.

        Kept because it is free, not because it is fast: paced, unpaced and one
        unbroken write all measure 2.85-2.89 ms per step, and only the unpaced
        run has ever dropped a frame. The step's cost is the device's read loop
        (docs/findings/link-latency.md), not how the host hands the bytes over.
        """
        for start in range(0, len(frame), USB_PACKET):
            self._serial.write(frame[start : start + USB_PACKET])
            self._serial.flush()
            if start + USB_PACKET < len(frame):
                spin_until = time.perf_counter() + USB_FRAME_S
                while time.perf_counter() < spin_until:
                    pass

    def _desync(self, message: str) -> ProtocolError:
        """Drop whatever is still in the RX ring, then describe the failure.

        `self_test` and `closed_loop` call `infer` in a loop, so without this
        flush one stray byte shifts every later frame and the run fails from the
        second frame on, blaming a protocol that already recovered.
        """
        self._serial.reset_input_buffer()
        return ProtocolError(message)

    def infer(self, obs: np.ndarray) -> tuple[np.ndarray, int]:
        """One control step: actions, and the board's own inference time in us."""
        payload = self._req.pack(*np.asarray(obs, dtype=np.float32).reshape(-1))
        want = 1 + self._resp.size + 1

        t0 = time.perf_counter_ns()
        self._write_paced(b"R" + payload + bytes([xor8(payload)]))
        # The tag first, and the rest sized from it: an error frame is 3 bytes,
        # so reading the reply width would burn the whole serial timeout on every
        # rejected frame before the diagnosis below can print.
        tag = self._serial.read(1)
        if tag == b"E":
            raise self._desync(f"board rejected the frame: {self._why(payload)}")
        if not tag:
            raise self._desync(f"no reply within {self._serial.timeout} s")
        if tag != b"A":
            raise self._desync(f"expected an 'A' frame, got {tag!r} (link out of sync)")

        resp = self._serial.read(want - 1)
        t1 = time.perf_counter_ns()
        if len(resp) != want - 1:
            raise self._desync(f"short reply: {1 + len(resp)} of {want} bytes")
        if xor8(resp[:-1]) != resp[-1]:
            raise self._desync("reply checksum mismatch")

        *action, _us_read, us_infer = self._resp.unpack(resp[:-1])
        self.last_round_trip_ms = (t1 - t0) / 1e6
        self.last_infer_us = int(us_infer)
        return np.asarray(action, dtype=np.float32), self.last_infer_us

    def _why(self, payload: bytes) -> str:
        """Render the count in an 'E' frame whose tag has already been read."""
        rest = self._serial.read(ERROR_FRAME - 1)
        if len(rest) != ERROR_FRAME - 1:
            return "the board sent a truncated error frame"
        n = int.from_bytes(rest, "little")
        if n == REJECT_BAD_CHECKSUM:
            return "the payload arrived, but its checksum byte did not match"
        if n == REJECT_NO_CHECKSUM:
            return "the payload arrived, but its checksum byte never did"
        return f"only {n} of {len(payload)} payload bytes arrived"

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.infer(obs)[0]

    def close(self) -> None:
        self._serial.close()


def self_test(run: Run, board: Board, n: int = 128) -> dict[str, Any]:
    model = board.model
    if model is None:
        raise ValueError("board has no model; construct it with `Board.open(run=...)`")
    export = ActorExport.load(run.actor_npz)
    reference_in = export.reference_in[:n]
    reference_out = export.reference_out[:n]

    device_out, micros, round_trips = [], [], []
    with progress.stage("replay", len(reference_in)) as bar:
        for i, obs in enumerate(reference_in, start=1):
            action, us = board.infer(obs)
            device_out.append(action)
            micros.append(us)
            round_trips.append(board.last_round_trip_ms)
            bar.set(i, note=f"{us} us infer, {board.last_round_trip_ms:.2f} ms round trip")
    device_out = np.stack(device_out)
    host_out = model(reference_in)

    return {
        "n": len(reference_in),
        "identity": asdict(board.identity),
        "vs_emulator_max": float(np.abs(device_out - host_out).max()),
        "vs_torch_max": float(np.abs(device_out - reference_out).max()),
        "vs_torch_mae": float(np.abs(device_out - reference_out).mean()),
        # `link.h` times `tinyml_infer` alone; the checksum pass over the
        # observation buffer is in `us_read`, not here.
        "inference_us_mean": float(np.mean(micros)),
        "inference_us_max": int(np.max(micros)),
        # Aggregated over the same `n` frames as `inference_us_*`: one table row
        # reporting the last sample next to means of everything else reads as one
        # statistic and is not.
        "round_trip_ms_mean": float(np.mean(round_trips)),
        "round_trip_ms_max": float(np.max(round_trips)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--port", default=None, help="serial device (default: autodetect)")
    parser.add_argument("--n", type=int, default=128, help="frames to replay")
    args = parser.parse_args()

    try:
        run = Run.resolve(args.run_dir)
        with progress.session(skip=False) as reporter:
            setup_logging(console=reporter.console)
            board = Board.open(args.port, run)
            try:
                result = self_test(run, board, args.n)
            finally:
                board.close()
    except (ProtocolError, FileNotFoundError, serial.SerialException) as exc:
        progress.fail(exc)

    for label, value in (
        ("frames", result["n"]),
        ("vs emulator (max)", f"{result['vs_emulator_max']:.3e}"),
        ("vs pytorch  (max)", f"{result['vs_torch_max']:.5f}"),
        ("vs pytorch  (mae)", f"{result['vs_torch_mae']:.5f}"),
        (
            "inference",
            f"{result['inference_us_mean']:.0f} us mean, {result['inference_us_max']} us max",
        ),
        (
            "round trip",
            f"{result['round_trip_ms_mean']:.2f} ms mean, {result['round_trip_ms_max']:.2f} ms max",
        ),
    ):
        print(f"{label:<20s} {value}")


if __name__ == "__main__":
    main()
