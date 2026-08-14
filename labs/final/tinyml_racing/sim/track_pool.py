"""Pre-generated track cache. Building one track costs ~12 ms (an analytically sampled
corner sequence with its offset walls, plus the racing-line solve) against ~0.05 ms for
a simulation step, so generating a layout per `reset()` spent most of the wall clock on
geometry, worst when the car crashed early. A pool generated lazily per worker keeps
the variety that stops the policy overfitting one circuit and makes `reset()` free.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from tinyml_racing.sim.geometry import ArcLengthLUT
from tinyml_racing.sim.racing_line import add_racing_line
from tinyml_racing.sim.track import Track, TrackConfig, generate_track


# `eq=False` for the same reason as `Track`: ndarray fields make a generated
# `__eq__` return an array and `__hash__` raise. Identity is what callers mean.
@dataclass(frozen=True, eq=False)
class PooledTrack:
    """A track plus the derived structures every episode on it needs.

    The arc-length lookup table is built once here rather than per reset;
    it is read-only during an episode, so sharing it is safe.
    """

    seed: int
    track: Track
    progress: ArcLengthLUT


def build_track(seed: int, config: TrackConfig | None = None) -> PooledTrack:
    track = add_racing_line(generate_track(seed, config))
    return PooledTrack(seed=seed, track=track, progress=ArcLengthLUT(track.racing_line))


class TrackPool:
    """A fixed set of layouts, generated lazily and cached by seed.

    Lazy generation matters for the viewer and for tests, which only ever
    touch one or two layouts and should not pay for sixty-four.
    """

    def __init__(self, seeds: Iterable[int], config: TrackConfig | None = None) -> None:
        self.seeds = tuple(int(s) for s in seeds)
        if not self.seeds:
            raise ValueError("TrackPool needs at least one seed")
        self.config = config or TrackConfig()
        self._cache: dict[int, PooledTrack] = {}

    @classmethod
    def from_seed_range(
        cls,
        n_tracks: int,
        seed_range: tuple[int, int],
        rng: np.random.Generator,
        config: TrackConfig | None = None,
    ) -> TrackPool:
        """`n_tracks` layouts drawn from `[lo, hi)`, one distinct seed each.

        Distinct, which `rng.integers` is not: `get` caches by seed, so a repeated draw
        makes the pool quietly smaller than `n_tracks` and gives that layout double
        weight in `sample`. Unlikely at the shipped 2e9 range, likely at a narrow one.
        """
        lo, hi = seed_range
        span = int(hi) - int(lo)
        if n_tracks > span:
            raise ValueError(f"cannot draw {n_tracks} distinct seeds from range [{lo}, {hi})")
        # `replace=False` over a range this wide is Floyd's algorithm inside numpy, so
        # it stays O(n_tracks) rather than materializing 2e9 candidates, and it is a
        # pure function of the generator state, so the pool stays reproducible.
        seeds = rng.choice(span, size=n_tracks, replace=False) + lo
        return cls(seeds, config)

    def get(self, seed: int) -> PooledTrack:
        seed = int(seed)
        pooled = self._cache.get(seed)
        if pooled is None:
            pooled = build_track(seed, self.config)
            self._cache[seed] = pooled
        return pooled

    def sample(self, rng: np.random.Generator) -> PooledTrack:
        return self.get(self.seeds[int(rng.integers(len(self.seeds)))])
