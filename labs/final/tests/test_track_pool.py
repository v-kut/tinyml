"""Contracts of the layout cache: identity, keying, and how its seeds are drawn. Every
failure here is silent in training, a cache that ignores its key shrinks the pool to one
circuit, duplicate draws give one layout double weight, and an arc-length table that
disagrees with its own vertices corrupts the progress reward. None of them raise, and all
look like a policy that simply did not generalise.
"""

import numpy as np
import pytest

from tinyml_racing.ml.config import EVAL_SEED_RANGE, TRAIN_SEED_RANGE
from tinyml_racing.sim.track_pool import PooledTrack, TrackPool, build_track


def test_same_seed_gives_identical_geometry():
    a = build_track(7)
    b = build_track(7)
    np.testing.assert_array_equal(a.track.centerline, b.track.centerline)
    np.testing.assert_array_equal(a.track.racing_line, b.track.racing_line)


def test_different_seeds_give_different_geometry():
    a = build_track(7)
    b = build_track(8)
    # Sample counts follow lap length, so the arrays need not even share a
    # shape, compare the lap the pool actually built.
    assert a.track.length != b.track.length
    assert a.progress.total_length != b.progress.total_length


def test_get_is_cached_and_actually_keyed_on_the_seed():
    # `pool.get(11) is pool.get(11)` on its own is satisfied by a one-entry cache handing
    # the same track back for every seed. The identity, the distinctness and the
    # labelling have to be asserted together.
    pool = TrackPool([11, 12])
    first = pool.get(11)
    assert pool.get(11) is first, "repeated get must not regenerate the track"

    other = pool.get(12)
    assert other is not first, "distinct seeds must not share a layout"
    assert pool.get(11) is first, "a second seed must not evict the first"
    assert (first.seed, other.seed) == (11, 12)

    # And the label has to match the geometry, not just the argument: `get`
    # must build the track the seed names.
    np.testing.assert_array_equal(first.track.centerline, build_track(11).track.centerline)
    np.testing.assert_array_equal(other.track.centerline, build_track(12).track.centerline)
    assert first.track.length != other.track.length


def test_sample_only_returns_pooled_seeds():
    pool = TrackPool([21, 22, 23])
    rng = np.random.default_rng(0)
    drawn = {pool.sample(rng).seed for _ in range(40)}
    assert drawn <= {21, 22, 23}
    assert len(drawn) > 1, "sampling should not collapse onto a single layout"


def test_from_seed_range_is_a_pure_function_of_the_generator_state():
    # The pool has to be recoverable from the run's seed alone, an evaluation
    # rerun must land on the layouts that were trained on, or against.
    lo, hi = TRAIN_SEED_RANGE
    a = TrackPool.from_seed_range(5, TRAIN_SEED_RANGE, np.random.default_rng(3))
    b = TrackPool.from_seed_range(5, TRAIN_SEED_RANGE, np.random.default_rng(3))
    c = TrackPool.from_seed_range(5, TRAIN_SEED_RANGE, np.random.default_rng(4))
    assert a.seeds == b.seeds
    assert a.seeds != c.seeds, "the draw ignores the generator: every run is one pool"
    assert all(lo <= s < hi for s in a.seeds)


def test_from_seed_range_draws_distinct_seeds():
    """Duplicates make the pool quietly smaller than it claims to be.

    `get` caches by seed, so a repeated draw is one layout occupying two slots: the pool
    holds fewer circuits than `n_tracks` and `sample` weights the duplicate twice. The
    draw below spans exactly its range, so any replacement at all shows up.
    """
    exact = TrackPool.from_seed_range(8, (100, 108), np.random.default_rng(0))
    assert sorted(exact.seeds) == list(range(100, 108))

    narrow = TrackPool.from_seed_range(32, (500, 540), np.random.default_rng(1))
    assert len(set(narrow.seeds)) == 32
    assert all(500 <= s < 540 for s in narrow.seeds)


def test_from_seed_range_rejects_a_request_the_range_cannot_satisfy():
    # One more track than distinct seeds available. Silently returning a
    # short pool would misreport the variety the run was trained on.
    with pytest.raises(ValueError, match="cannot draw 9 distinct seeds"):
        TrackPool.from_seed_range(9, (100, 108), np.random.default_rng(0))


def test_pool_rejects_empty_seed_set():
    with pytest.raises(ValueError, match="at least one seed"):
        TrackPool([])


def test_train_and_eval_seed_ranges_are_disjoint():
    # A policy evaluated on layouts it trained on would report memorization as
    # generalization, so the split must hold by construction.
    assert TRAIN_SEED_RANGE[1] <= EVAL_SEED_RANGE[0]


def test_pooled_track_carries_a_prebuilt_arc_length_lut():
    # `0.0 <= s <= total` was satisfied by `return 0.0`. The table's actual
    # contract is the identity on its own vertices: `arc_length_at(line[k])`
    # is `cum_lengths[k]`, wrap vertex included.
    pooled = build_track(31)
    assert isinstance(pooled, PooledTrack)
    line = pooled.track.racing_line
    assert line is not None, "build_track must attach a racing line"
    progress = pooled.progress
    assert progress.total_length > 0

    n = len(line)
    for k in (0, 1, n // 3, n // 2, n - 2, n - 1):
        s = progress.arc_length_at(tuple(line[k]))
        assert s == pytest.approx(progress.cum_lengths[k], abs=1e-6), f"vertex {k}"


def test_arc_length_rises_to_the_lap_across_the_start_finish_line():
    """The seam, on the geometry the env actually runs.

    A car three quarters of the way along the closing segment is nearest to vertex 0, so a
    projection that only looks at the segment leaving the nearest vertex calls it arc
    length 0.0, a lap of apparent backwards travel one sample before the line.
    """
    pooled = build_track(31)
    progress, line = pooled.progress, pooled.track.racing_line
    assert line is not None, "the pool builds the racing line alongside the track"
    closing = progress.seg_lengths[-1]
    for frac in (0.25, 0.5, 0.75):
        probe = (1.0 - frac) * line[-1] + frac * line[0]
        s = progress.arc_length_at(tuple(probe))
        assert s == pytest.approx(progress.cum_lengths[-1] + frac * closing, abs=1e-6)
        assert progress.cum_lengths[-1] <= s < progress.total_length
