"""Tests du registre de schémas de features (D1, audit 2026-06)."""

from __future__ import annotations

import pytest

from telemetry.feature_names import (
    AGGREGATION_RULE,
    FEATURE_NAMES,
    FEATURE_NAMES_V4,
    MODALITY_SLICES,
    SCHEMA_V4,
    SCHEMA_V5_1,
    SCHEMAS,
    SIGNAL_DIM,
    get_schema,
    schema_for_dim,
    signal_dim,
)


def test_v4_schema_unchanged_backward_compat():
    assert FEATURE_NAMES_V4 is FEATURE_NAMES
    assert signal_dim(SCHEMA_V4) == SIGNAL_DIM == 17
    assert get_schema(SCHEMA_V4) == FEATURE_NAMES
    # Toutes les features v4 ont une règle d'agrégation
    assert set(FEATURE_NAMES_V4) == set(AGGREGATION_RULE)


def test_v5_1_schema_dimensions():
    assert signal_dim(SCHEMA_V5_1) == 18
    names = get_schema(SCHEMA_V5_1)
    assert len(names) == len(set(names)), "noms dupliqués dans v5.1"
    # Features structurantes du schéma v5.1
    assert names[6] == "mem_limit_ratio"
    assert "jvm_heap_ratio" in names and "restart_count" in names
    # Features v4 retirées en v5.1
    for dropped in ("span_dur_p99", "retry_rate", "queue_depth", "log_warn_rate"):
        assert dropped not in names


def test_modality_slices_cover_schema():
    for version, names in SCHEMAS.items():
        sl = MODALITY_SLICES[version]
        assert sl["M"].start == 0
        assert sl["M"].stop == sl["T"].start
        assert sl["T"].stop == sl["L"].start
        assert sl["L"].stop == len(names)


def test_schema_for_dim_roundtrip():
    # En cas de collision de dimension (v5.1 et v5.2 font toutes deux 18),
    # schema_for_dim résout vers la PLUS ANCIENNE : l'inférence par dimension
    # ne sert qu'aux épisodes legacy sans metadata, qui prédatent le schéma
    # récent par construction (tout épisode v5.2 déclare sa version).
    for version in SCHEMAS:
        dim = signal_dim(version)
        oldest_with_dim = next(v for v in SCHEMAS if signal_dim(v) == dim)
        assert schema_for_dim(dim) == oldest_with_dim


def test_schema_for_dim_collision_prefers_legacy():
    from telemetry.feature_names import SCHEMA_V5_1, SCHEMA_V5_2

    assert signal_dim(SCHEMA_V5_1) == signal_dim(SCHEMA_V5_2) == 18
    assert schema_for_dim(18) == SCHEMA_V5_1


def test_unknown_schema_raises():
    with pytest.raises(KeyError):
        get_schema("v99")
    with pytest.raises(KeyError):
        schema_for_dim(5)


def test_get_schema_returns_copy():
    names = get_schema(SCHEMA_V4)
    names.append("mutant")
    assert get_schema(SCHEMA_V4) != names
