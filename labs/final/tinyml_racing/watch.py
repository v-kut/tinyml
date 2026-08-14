"""Watch policies drive, several at once on the same layout and the same spawn.

tinyml-watch --policy expert ppo student  # `board` drives the Nano over serial
"""

from __future__ import annotations

import argparse
import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from tinyml_racing.ml.config import RacingEnvConfig
from tinyml_racing.ml.env import RacingEnv
from tinyml_racing.ml.regression.dataset import pure_pursuit_teacher, snapshot_teacher
from tinyml_racing.ml.snapshot import load_snapshot
from tinyml_racing.render.hud import Latency, PanelLine
from tinyml_racing.render.theme import CONTENDER
from tinyml_racing.render.viewer import Ghost, PygameViewer, Trail
from tinyml_racing.utils import Run

# `(env, obs) -> action`, the signature `ml/regression/dataset.py` defines for a
# teacher: the expert reads privileged simulator state, a network the observation.
type Driver = Callable[[RacingEnv, np.ndarray], np.ndarray]

# Every source of a driver this repo can produce, in pipeline order. Resolved
# lazily, so `--policy all` on a run that skipped a stage reports the missing
# file and shows the drivers that do exist.
POLICIES = ("expert", "pretrain", "ppo", "student", "float32", "int8", "board")


@dataclass(frozen=True)
class Policy:
    name: str
    drive: Driver
    # What computes the action, for the table's own column: an actor's layer
    # widths (`61-64-64-2`) or, for a driver that is not a network, what it is.
    arch: str
    # Where the driver came from, for the followed row: training steps, the
    # numeric format it runs in, the port it is spoken to over.
    detail: str
    # Whatever the driver holds open. Only the board has one, and leaving a
    # serial port open on exit is how the next command fails to find it.
    close: Callable[[], None] | None = None
    # The driver's own report of what inference cost it, us. Only the board has
    # one: every other driver runs in this process, where the wall clock around
    # `drive` is the whole story.
    device_us: Callable[[], float] | None = None


def _snapshot_policy(name: str, path: Path, detail: str) -> Policy:
    snapshot = load_snapshot(path)
    return Policy(
        name,
        snapshot_teacher(snapshot),
        snapshot.arch,
        f"{detail}, {snapshot.num_timesteps:,} steps, float32 on host",
    )


def resolve(name: str, run: Run, port: str | None) -> Policy:
    """One named driver, or a `FileNotFoundError` naming what is missing."""
    match name:
        case "expert":
            return Policy(
                name, pure_pursuit_teacher(), "pure pursuit", "privileged state, no network"
            )
        case "pretrain":
            return _snapshot_policy(name, run.pretrain, "cloned expert")
        case "ppo":
            return _snapshot_policy(name, run.ppo, "reinforcement")
        case "student":
            return _snapshot_policy(name, run.student, "distilled from ppo")
        case "float32" | "int8" | "board":
            # The deploy bundle. Imported here rather than at module scope so
            # watching a training run costs no onnxruntime and no serial port.
            from tinyml_racing.deploy.artifact import ActorExport
            from tinyml_racing.deploy.quantize import float_actor, quantize_model

            export = ActorExport.load(run.actor_npz)
            model = quantize_model(export)
            if name == "float32":
                act = float_actor(export)
                return Policy(
                    name, lambda _env, obs: act(obs), model.arch, "exported actor, float32 on host"
                )
            if name == "int8":
                return Policy(
                    name,
                    lambda _env, obs: model.act(obs),
                    model.arch,
                    f"{model.activation}, int8 emulated on host",
                )

            from tinyml_racing.deploy.board import Board

            board = Board.open(port, run)
            return Policy(
                name,
                lambda _env, obs: board.act(obs),
                model.arch,
                f"int8 on {board.port}, {model.deployed_flash_bytes} B flash",
                close=board.close,
                device_us=lambda: board.last_infer_us,
            )
        case _:
            raise ValueError(f"unknown policy {name!r}; choose from {', '.join(POLICIES)}")


@dataclass
class Runner:
    """One contender's car, and what it has done with it so far."""

    policy: Policy
    env: RacingEnv
    shade: tuple[int, int, int]
    obs: np.ndarray | None = None
    action: np.ndarray | None = None
    trail: Trail = field(default_factory=Trail)
    reward: float = 0.0
    laps: float = 0.0
    steps: int = 0
    crashed: bool = False
    done: bool = False
    # Wall clock around `Policy.drive`, which for the board is the USB round
    # trip, and beside it whatever the device reported for the kernel alone.
    infer: Latency = field(default_factory=Latency)
    device: Latency = field(default_factory=Latency)

    def reset(self, seed: int) -> None:
        # Same seed into every runner, so the layout *and* the spawn are shared
        # and a divergence later is the policy's doing rather than the draw's.
        self.obs, _ = self.env.reset(seed=seed)
        self.trail.clear()
        self.action = None
        self.reward = self.laps = 0.0
        self.steps = 0
        self.crashed = self.done = False

    def step(self) -> None:
        if self.done or self.obs is None:
            return
        started = time.perf_counter_ns()
        self.action = np.asarray(self.policy.drive(self.env, self.obs), dtype=np.float32)
        self.infer.hit((time.perf_counter_ns() - started) / 1e6)
        if self.policy.device_us is not None:
            self.device.hit(self.policy.device_us() / 1e3)
        self.obs, reward, terminated, truncated, info = self.env.step(self.action)
        self.reward += float(reward)
        self.laps = float(info.get("lap_progress", self.laps))
        self.steps += 1
        self.crashed = bool(terminated)
        self.done = bool(terminated or truncated)
        # `Trail` owns the distance-sampling rule; this is every step it sees.
        self.trail.append(np.array([self.env.state.x, self.env.state.y]))

    @property
    def status(self) -> str:
        if self.crashed:
            return "OFF"
        return "out" if self.done else "on"


# (title, width, cell); a negative width is left-aligned. One row of widths, so
# the header cannot drift from the cells the way two format strings could.
type Column = tuple[str, int, Callable[[Runner], str]]


def _infer_cell(r: Runner) -> str:
    """`0.16/0.31`, the host's mean and worst per control step."""
    return f"{r.infer.mean_ms:.2f}/{r.infer.worst_ms:.2f}" if r.infer else "--"


def _device_cell(r: Runner) -> str:
    """The same pair as the device reported it, us. Only the board fills it."""
    if not r.device:
        return "--"
    return f"{1000 * r.device.mean_ms:.0f}/{1000 * r.device.worst_ms:.0f}"


# No column repeats the panel above it: the followed car's speed, slip and
# odometer are the viewer's own readout, and this table is what the panel
# cannot say, one row per contender.
_COLUMNS: tuple[Column, ...] = (
    ("policy", -9, lambda r: r.policy.name),
    # Widest real value is `61-256-256-2`; `pure pursuit` is the same 12.
    ("arch", -14, lambda r: r.policy.arch),
    ("reward", 9, lambda r: f"{r.reward:.0f}"),
    ("laps", 7, lambda r: f"{r.laps:.3f}"),
    # Mean and worst of the same window in one cell: a control loop is late
    # whenever a single step is, and that is a second number, not a column.
    ("infer ms", 12, _infer_cell),
    ("mcu us", 11, _device_cell),
)


def _columns(runners: Sequence[Runner]) -> tuple[Column, ...]:
    """The columns this field needs. `mcu us` exists only when something in it
    reports a device time: without a board it is a column of dashes.
    """
    if any(r.device for r in runners):
        return _COLUMNS
    return tuple(column for column in _COLUMNS if column[0] != "mcu us")


def _cells(values: Sequence[str], columns: Sequence[Column]) -> str:
    return "".join(
        f"{v:<{-w}s}" if w < 0 else f"{v:>{w}s}"
        for v, (_, w, _) in zip(values, columns, strict=True)
    )


def hud_table(runners: Sequence[Runner], focus: int) -> list[PanelLine]:
    """One row per contender in the order asked for, the followed one marked `>`."""
    columns = _columns(runners)
    rows = [PanelLine(f"  {_cells([title for title, _, _ in columns], columns)}  status")]
    for i, r in enumerate(runners):
        marker = ">" if i == focus else " "
        cells = _cells([cell(r) for _, _, cell in columns], columns)
        rows.append(PanelLine(f"{marker} {cells}  {r.status}", r.shade))
    lead = runners[focus]
    rows.append(PanelLine(f"[tab] {lead.policy.name}: {lead.policy.detail}", lead.shade))
    rows.append(PanelLine("[r]estart  [t]rack"))
    return rows


def _wait_until(target: float) -> float:
    """Sleep until `target`, and return the deadline the next step counts from.

    Lateness is never repaid: each step needs its own round trip.
    """
    now = time.perf_counter()
    if target > now:
        time.sleep(target - now)
        return target
    return now


def watch(
    policies: Sequence[Policy],
    env_cfg: RacingEnvConfig,
    track_seed: int | None = None,
    render_fps: float = 60.0,
    sim_speed: float = 1.0,
    max_steps: int | None = None,
) -> None:
    """Drive every policy through the same layout in lockstep, and draw it.

    One `ExitStack`, entered before anything that can fail: a board arrives open.
    """
    rng = np.random.default_rng()
    seed = track_seed if track_seed is not None else int(rng.integers(*env_cfg.track_seed_range))

    with contextlib.ExitStack() as stack:
        # Registered first, so they are given back last.
        for p in policies:
            if p.close is not None:
                stack.callback(p.close)

        runners = []
        for i, p in enumerate(policies):
            # One env each: they share a layout, not a car. `fixed_track_seed`
            # is what makes the pool hand all of them the same one.
            env = RacingEnv(config=replace(env_cfg, fixed_track_seed=seed))
            stack.callback(env.close)
            runners.append(Runner(policy=p, env=env, shade=CONTENDER[i % len(CONTENDER)]))
        dt = runners[0].env.params.dt
        limit = runners[0].env.max_steps if max_steps is None else max_steps
        step_wall = dt / sim_speed if sim_speed > 0 else 0.0
        focus = 0

        names = ", ".join(p.name for p in policies)
        viewer = PygameViewer(caption=f"tinyml-racing - {names}").open()
        stack.callback(viewer.close)

        while viewer.running:
            for r in runners:
                r.env.cfg.fixed_track_seed = seed
                r.reset(seed)
            # A `[r]` restart keeps the seed, so `set_track` early-returns and
            # cannot see this episode boundary.
            viewer.begin_episode()
            deadline = time.perf_counter()
            # Counted here rather than read off `runners[0]`: `Runner.step`
            # returns early once that runner is done, so its `steps` freezes
            # the moment it goes off and a limit read from it never arrives.
            steps = 0

            while viewer.running and not all(r.done for r in runners):
                keys = viewer.pump()
                if "r" in keys:
                    break
                if "t" in keys:
                    seed = int(rng.integers(*env_cfg.track_seed_range))
                    break
                if "tab" in keys:
                    focus = (focus + 1) % len(runners)
                    # The odometer and trail integrate against the previous
                    # position, which now belongs to a car tens of metres away.
                    viewer.begin_episode()

                for r in runners:
                    r.step()
                steps += 1
                lead = runners[focus]
                # Per control step, both of them: a no-op unless the layout
                # changed, and the viewer's simulation-rate accumulators.
                viewer.set_track(lead.env.track)
                viewer.observe(lead.env.state, dt)

                if viewer.frame_due(render_fps):
                    viewer.draw(
                        lead.env.state,
                        scan=lead.env.last_scan,
                        lidar=env_cfg.lidar,
                        action=lead.action,
                        trail=lead.trail,
                        # The followed car keeps the shade it had as a ghost, so
                        # `[tab]` moves the camera and nothing else.
                        shade=lead.shade,
                        ghosts=[
                            Ghost(r.env.state, r.shade, r.trail)
                            for i, r in enumerate(runners)
                            if i != focus
                        ],
                        hud_lines=hud_table(runners, focus),
                    )
                if steps >= limit:
                    break
                deadline = _wait_until(deadline + step_wall)


def main() -> None:
    from tinyml_racing.deploy.evaluate import load_env_config

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--policy",
        nargs="+",
        default=["expert", "ppo"],
        metavar="NAME",
        help=f"one or more of: {', '.join(POLICIES)}, or 'all'",
    )
    parser.add_argument("--track-seed", type=int, default=None)
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument(
        "--sim-speed",
        type=float,
        default=1.0,
        help="times real time; 0 runs flat out",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        # Per episode, not per run: the loop redraws and restarts until the
        # window is closed, which is what makes [r] and [t] useful.
        help="step cap per episode (default: the env's own truncation limit)",
    )
    parser.add_argument(
        "--sensor",
        choices=("clean", "noisy"),
        default="clean",
        help="detector the drivers see; noisy is the training sensor model, "
        "and the LiDAR fan then draws the dropouts they have to ride over",
    )
    parser.add_argument("--port", default=None, help="serial port for --policy board")
    args = parser.parse_args()
    if args.sim_speed < 0:
        # `sim_speed > 0` is what picks the paced branch, so a negative value
        # would quietly run flat out instead of the reverse-time nobody meant.
        parser.error("--sim-speed must be 0 (flat out) or positive")

    run = Run.resolve(args.run_dir)
    env_cfg = load_env_config(run).evaluation_variant(clean_sensor=args.sensor == "clean")
    # `all` never includes the board: opening a serial port is not something
    # a convenience flag should do behind the user's back.
    wanted = [p for p in POLICIES if p != "board"] if args.policy == ["all"] else args.policy

    # From the first `resolve` on, because that is where a `board` opens a
    # serial port: an unavailable policy later in the list must not abandon it.
    with contextlib.ExitStack() as stack:
        policies = []
        for name in wanted:
            try:
                policy = resolve(name, run, args.port)
            except FileNotFoundError as missing:
                # A run that skipped a stage is the normal case, not an error.
                print(f"  {name:<9s} unavailable: {missing}")
                continue
            if policy.close is not None:
                stack.callback(policy.close)
            policies.append(policy)
        if not policies:
            raise SystemExit(f"none of {', '.join(wanted)} exist under {run.root}")

        for p in policies:
            print(f"  {p.name:<9s} {p.arch:<14s} {p.detail}")
        # `watch` puts the same callbacks on its own stack before anything in
        # it can fail, so ownership is handed over rather than a port closed
        # twice. Nothing between here and the call can raise.
        stack.pop_all()
        watch(
            policies,
            env_cfg,
            track_seed=args.track_seed,
            render_fps=args.render_fps,
            sim_speed=args.sim_speed,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
