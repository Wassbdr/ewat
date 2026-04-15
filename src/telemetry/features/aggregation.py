"""Aggregation functions for reducing pod-level samples to service-level features.

Rules (per CLAUDE.md):
- Saturation (CPU, RAM, net_sat, disk_io, queue_depth) → max
- Rates (error_rate, warn_rate, retry_rate, abnormal_rate) → volume-weighted sum
- Latency (latency_p99, span_dur_med) → P99 on the *union* of raw duration lists
- Structural (trace_depth, fan_out, latency_cv, semantic_anomaly, lexical_entropy) → median

Never use simple mean. Never compute percentile-of-percentiles for latency.
"""

from __future__ import annotations

import numpy as np


def aggregate_max(values: np.ndarray) -> float:
    """Return the max of an array of scalar per-pod values.

    Parameters
    ----------
    values:
        1-D array of shape (P,) where P is the number of pods.

    Returns
    -------
    float
        Maximum value across pods.
    """
    if values.size == 0:
        return float("nan")
    return float(np.nanmax(values))


def aggregate_volume_weighted(rates: np.ndarray, volumes: np.ndarray) -> float:
    """Volume-weighted sum of per-pod rates.

    Parameters
    ----------
    rates:
        1-D array (P,) of per-pod rate values (fractions, not counts).
    volumes:
        1-D array (P,) of per-pod request/event counts for the window.

    Returns
    -------
    float
        Weighted average rate; falls back to simple mean when all volumes are 0.
    """
    total_volume = float(np.nansum(volumes))
    if total_volume == 0.0 or volumes.size == 0:
        return float(np.nanmean(rates)) if rates.size > 0 else float("nan")
    weights = volumes / total_volume
    return float(np.nansum(weights * rates))


def aggregate_p99_union(sample_lists: list[np.ndarray]) -> float:
    """P99 latency on the *union* of all per-pod raw duration samples.

    Never compute percentile-of-percentiles. This function requires the caller
    to pass actual raw samples (e.g. from histograms reconstructed via
    `_reconstruct_from_histogram`).

    Parameters
    ----------
    sample_lists:
        List of 1-D arrays, one per pod, each containing raw duration samples
        in seconds.

    Returns
    -------
    float
        99th percentile of the combined distribution.
    """
    if not sample_lists or all(arr.size == 0 for arr in sample_lists):
        return float("nan")
    union = np.concatenate([arr for arr in sample_lists if arr.size > 0])
    return float(np.nanpercentile(union, 99))


def aggregate_median(values: np.ndarray) -> float:
    """Median of an array of scalar per-pod values.

    Parameters
    ----------
    values:
        1-D array (P,) of per-pod structural feature values.

    Returns
    -------
    float
        Median across pods.
    """
    if values.size == 0:
        return float("nan")
    return float(np.nanmedian(values))


def reconstruct_from_histogram(
    bucket_bounds: np.ndarray,
    bucket_counts: np.ndarray,
    n_samples: int = 200,
) -> np.ndarray:
    """Reconstruct approximate raw samples from a Prometheus histogram.

    Prometheus exposes cumulative histograms (le buckets). We approximate
    the raw distribution by uniform-sampling within each bucket proportional
    to the bucket's count. This yields a sample-level distribution suitable
    for `aggregate_p99_union` without computing percentile-of-percentiles.

    Parameters
    ----------
    bucket_bounds:
        1-D array of upper-bound values for each finite bucket (in seconds).
        The +Inf bucket is handled separately.
    bucket_counts:
        1-D array of **incremental** (not cumulative) counts per bucket.
        Must be same length as bucket_bounds (exclude +Inf bucket count).
    n_samples:
        Target number of synthetic samples to generate. Actual count may
        differ slightly due to integer rounding.

    Returns
    -------
    np.ndarray
        Approximate raw sample array.
    """
    total = float(np.sum(bucket_counts))
    if total == 0:
        return np.array([])

    samples: list[np.ndarray] = []
    prev_bound = 0.0
    for bound, count in zip(bucket_bounds, bucket_counts):
        if count <= 0:
            prev_bound = float(bound)
            continue
        k = max(1, round(n_samples * count / total))
        samples.append(np.random.uniform(prev_bound, float(bound), k))
        prev_bound = float(bound)

    return np.concatenate(samples) if samples else np.array([])
