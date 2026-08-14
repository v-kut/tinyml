"""Live per-stage progress bars for a CLI run, plus the one key that steers a
training run.

`SkipKey` makes `s` mean "this stage has converged, move on": stages check
`Stage.skipped` only where stopping cannot leave a half-written artifact, and
`session(skip=False)` disables it for a pipeline with nothing to skip.
`stage()` is a silent no-op outside a `session()`, so library code calls it
unconditionally the way `logging` behaves for an unconfigured library.
"""

from __future__ import annotations

import contextlib
import os
import select
import sys
import time
from collections.abc import Iterator
from typing import NoReturn, Self

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

SKIP_KEY = "s"
# `Stage.advance` runs once per sample, 200k times in a cloning stage, so accumulation
# is batched at this interval, which is also how often the skip key is polled.
_FLUSH_S = 0.05


class SkipKey:
    """`s` on the terminal, meaning "wrap up the current stage".

    `cbreak` rather than a signal: a signal reaches the whole foreground process
    group, which is PPO's `SubprocVecEnv` workers too. ISIG is left alone, so
    Ctrl-C still raises. A no-op when stdin is not a terminal.
    """

    def __init__(self, key: str = SKIP_KEY) -> None:
        self.key = key
        self._fd: int | None = None
        self._saved = None

    def __enter__(self) -> Self:
        if not sys.stdin.isatty():
            return self
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None and self._saved is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        self._fd = None

    def pressed(self) -> bool:
        """Whether the key is waiting in the input buffer. Drains what is."""
        if self._fd is None:
            return False
        hit = False
        while select.select([self._fd], [], [], 0)[0]:
            char = os.read(self._fd, 1)
            if not char:
                break
            hit |= char.decode("utf-8", "ignore").lower() == self.key
        return hit


class Stage:
    """One bar, and the answer to "should I stop early".

    `advance` takes the metrics to show beside the bar, so there is no second
    path by which a number reaches the display.
    """

    def __init__(self, reporter: Reporter | None, task_id) -> None:
        self._reporter = reporter
        self._task = task_id
        self._pending = 0.0
        self._done = 0.0
        self._note = ""
        self._flushed = 0.0
        self._skipped = False

    def advance(self, step: float = 1.0, note: str = "") -> None:
        self._pending += step
        if note:
            self._note = note
        self._flush()

    def set(self, completed: float, note: str = "") -> None:
        """Report an absolute position, for a caller that counts for itself."""
        self._pending = completed - self._done
        if note:
            self._note = note
        self._flush()

    def _flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._flushed < _FLUSH_S:
            return
        self._flushed = now
        self._done += self._pending
        self._pending = 0.0
        if self._reporter is not None:
            self._reporter.update(self._task, completed=self._done, note=self._note)
            self._skipped = self._skipped or self._reporter.skip_requested()

    @property
    def skipped(self) -> bool:
        """True once the skip key has been pressed during this stage.

        Latched: two checks in one stage must see the same answer, and the
        keystroke is consumed by whichever runs first.
        """
        self._flush()
        return self._skipped


class Reporter:
    """The live display: one bar per stage, kept on screen as they complete."""

    def __init__(self, console: Console, skip: SkipKey | None = None) -> None:
        self.console = console
        self._skip = skip
        self._progress = Progress(
            TextColumn("[bold cyan]{task.fields[label]:<13}"),
            BarColumn(bar_width=18, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[note]}"),
            console=console,
        )
        hint = Text("")
        if skip is not None and sys.stdin.isatty():
            hint = Text(f"  {SKIP_KEY} finish this stage and move on", style="dim")
        self._live = Live(
            Group(self._progress, hint), console=console, refresh_per_second=8, transient=False
        )

    def __enter__(self) -> Self:
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._live.__exit__(*exc)

    def update(self, task_id, **fields) -> None:
        self._progress.update(task_id, **fields)

    def skip_requested(self) -> bool:
        return self._skip is not None and self._skip.pressed()

    @contextlib.contextmanager
    def stage(self, label: str, total: float) -> Iterator[Stage]:
        task = self._progress.add_task("", label=label, total=total, note="")
        st = Stage(self, task)
        try:
            yield st
        finally:
            # `_flush(force=True)` already pushed the final count. The bar is
            # left on screen rather than removed: the finished stages are the
            # run's shape, and a bar that vanishes takes its timing with it.
            st._flush(force=True)


_active: Reporter | None = None


@contextlib.contextmanager
def session(skip: bool = True) -> Iterator[Reporter]:
    """Own the terminal for the duration of a run.

    `skip=False` for a pipeline whose stages cannot be cut short, the key
    would do nothing and the hint would lie.
    """
    global _active  # noqa: PLW0603, one display owns the terminal per process
    console = Console()
    with contextlib.ExitStack() as stack:
        key = stack.enter_context(SkipKey()) if skip else None
        reporter = stack.enter_context(Reporter(console, key))
        previous, _active = _active, reporter
        try:
            yield reporter
        finally:
            # The previous reporter, not `None`: a nested session must not leave
            # the outer run's bars silently disabled.
            _active = previous


@contextlib.contextmanager
def stage(label: str, total: float) -> Iterator[Stage]:
    """A progress bar for `label`, or a silent stand-in outside a session."""
    if _active is None:
        yield Stage(None, None)
        return
    with _active.stage(label, total) as st:
        yield st


@contextlib.contextmanager
def step(label: str, note: str = "") -> Iterator[Stage]:
    """A bar for one indivisible step, completed on exit.

    For a stage with no natural count, a compile, one file written. The
    elapsed timer is the feedback while it runs; the bar fills when it lands.
    """
    with stage(label, 1.0) as st:
        if note:
            st.advance(0.0, note=note)
        yield st
        st.advance(1.0)


def fail(exc: Exception) -> NoReturn:
    """Print an operator error and exit 1, without a traceback.

    For the conditions a CLI is expected to hit, no board, a stale flash, a
    missing run, a compiler that is not installed. A traceback names our frames;
    these messages already name the fix.
    """
    Console(stderr=True).print(f"[bold red]error[/] {exc}")
    raise SystemExit(1)
