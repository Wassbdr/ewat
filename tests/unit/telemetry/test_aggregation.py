"""Unit tests for telemetry.features.aggregation."""

import numpy as np
import pytest

from telemetry.features.aggregation import (
    aggregate_max,
    aggregate_median,
    aggregate_p99_union,
    aggregate_volume_weighted,
    reconstruct_from_histogram,
)


class TestAggregateMax:
    def test_returns_max(self):
        vals = np.array([0.1, 0.9, 0.5])
        assert aggregate_max(vals) == pytest.approx(0.9)

    def test_single_element(self):
        assert aggregate_max(np.array([0.42])) == pytest.approx(0.42)

    def test_empty_returns_nan(self):
        assert np.isnan(aggregate_max(np.array([])))

    def test_ignores_nan(self):
        vals = np.array([0.3, float("nan"), 0.7])
        assert aggregate_max(vals) == pytest.approx(0.7)


class TestAggregateVolumeWeighted:
    def test_basic_weighting(self):
        rates = np.array([0.1, 0.5])
        volumes = np.array([100.0, 0.0])  # all weight on first pod
        result = aggregate_volume_weighted(rates, volumes)
        assert result == pytest.approx(0.1)

    def test_equal_volumes(self):
        rates = np.array([0.2, 0.4])
        volumes = np.array([10.0, 10.0])
        assert aggregate_volume_weighted(rates, volumes) == pytest.approx(0.3)

    def test_zero_volumes_falls_back_to_mean(self):
        rates = np.array([0.2, 0.4])
        volumes = np.array([0.0, 0.0])
        assert aggregate_volume_weighted(rates, volumes) == pytest.approx(0.3)

    def test_empty_returns_nan(self):
        result = aggregate_volume_weighted(np.array([]), np.array([]))
        assert np.isnan(result)


class TestAggregateP99Union:
    def test_known_distribution(self):
        # All samples in [0, 1]: P99 should be close to 1.0
        rng = np.random.default_rng(42)
        a = rng.uniform(0, 1, 1000).astype(np.float32)
        b = rng.uniform(0, 1, 1000).astype(np.float32)
        p99 = aggregate_p99_union([a, b])
        assert 0.95 < p99 <= 1.0

    def test_empty_returns_nan(self):
        assert np.isnan(aggregate_p99_union([]))

    def test_empty_arrays_returns_nan(self):
        assert np.isnan(aggregate_p99_union([np.array([]), np.array([])]))

    def test_single_list(self):
        samples = np.array([0.01, 0.02, 0.50, 0.99], dtype=np.float32)
        p99 = aggregate_p99_union([samples])
        assert p99 == pytest.approx(np.percentile(samples, 99))


class TestAggregateMedian:
    def test_basic(self):
        vals = np.array([1.0, 3.0, 2.0])
        assert aggregate_median(vals) == pytest.approx(2.0)

    def test_empty_returns_nan(self):
        assert np.isnan(aggregate_median(np.array([])))

    def test_ignores_nan(self):
        vals = np.array([1.0, float("nan"), 3.0])
        assert aggregate_median(vals) == pytest.approx(2.0)


class TestReconstructFromHistogram:
    def test_output_shape_positive_counts(self):
        bounds = np.array([0.1, 0.5, 1.0, 5.0])
        counts = np.array([10.0, 40.0, 30.0, 20.0])
        samples = reconstruct_from_histogram(bounds, counts, n_samples=100, rng=0)
        assert samples.size > 0

    def test_values_within_bounds(self):
        bounds = np.array([1.0, 2.0, 5.0])
        counts = np.array([10.0, 10.0, 10.0])
        samples = reconstruct_from_histogram(bounds, counts, n_samples=300, rng=0)
        assert float(samples.max()) <= 5.0
        assert float(samples.min()) >= 0.0

    def test_zero_counts_returns_empty(self):
        bounds = np.array([1.0, 2.0])
        counts = np.array([0.0, 0.0])
        samples = reconstruct_from_histogram(bounds, counts, rng=0)
        assert samples.size == 0

    def test_seeded_reproducibility(self):
        bounds = np.array([0.1, 0.5, 1.0, 5.0])
        counts = np.array([10.0, 40.0, 30.0, 20.0])
        a = reconstruct_from_histogram(bounds, counts, n_samples=200, rng=42)
        b = reconstruct_from_histogram(bounds, counts, n_samples=200, rng=42)
        assert np.array_equal(a, b)

    def test_warns_when_no_rng(self):
        import warnings
        bounds = np.array([0.1, 0.5, 1.0, 5.0])
        counts = np.array([10.0, 40.0, 30.0, 20.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reconstruct_from_histogram(bounds, counts, n_samples=10)
            assert any(
                issubclass(warning.category, RuntimeWarning)
                and "without an explicit rng" in str(warning.message)
                for warning in w
            )
