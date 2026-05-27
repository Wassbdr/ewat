"""SPARQL queries canonical to the EWAT anomaly ontology.

Five reference queries are exposed; each is also documented in the plan
``oublie-la-phase-jury-tidy-reef.md`` §Phase 5 and tested in
``tests/unit/ontology/test_queries.py``. The query strings use the default
ontology IRI ``http://ewat.devoteam.com/ontology#`` aliased as ``ewat:``.

Use :func:`run_query` to evaluate a query against an owlready2 world; it
returns a list of dicts keyed by SELECT variable names so the consumer
does not need to know the underlying tuple format.
"""

from __future__ import annotations

from typing import Any

from owlready2 import World


PREFIXES = """
PREFIX ewat: <http://ewat.devoteam.com/ontology#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
"""


# ---------------------------------------------------------------------------
# Five canonical queries (Plan §Phase 5)
# ---------------------------------------------------------------------------


# 1. All Composite_Anomaly instances (including Drift_With_Anomaly, CascadingFailure)
QUERY_ALL_COMPOSITES = PREFIXES + """
SELECT DISTINCT ?anomaly WHERE {
    ?anomaly rdf:type ?subClass .
    ?subClass rdfs:subClassOf* ewat:Composite_Anomaly .
}
"""

# 2. Anomalies causally downstream of Memory_Saturation (transitive)
QUERY_DOWNSTREAM_MEMORY = PREFIXES + """
SELECT DISTINCT ?downstream WHERE {
    ?mem rdf:type ?memClass .
    ?memClass rdfs:subClassOf* ewat:Memory_Saturation .
    ?mem ewat:causes+ ?downstream .
}
"""

# 3. Services affected by any CascadingFailure
QUERY_SERVICES_CASCADING = PREFIXES + """
SELECT DISTINCT ?service WHERE {
    ?anomaly rdf:type ?subClass .
    ?subClass rdfs:subClassOf* ewat:CascadingFailure .
    ?anomaly ewat:affects ?service .
}
"""

# 4. Signatures sharing >=3 features whose weight > 0.2
QUERY_SHARED_HEAVY_FEATURES = PREFIXES + """
SELECT ?signature ?featureName WHERE {
    ?signature ewat:hasFeatureWeight ?fw .
    ?fw ewat:featureName ?featureName .
    ?fw ewat:weightValue ?w .
    FILTER (?w > 0.2)
}
"""

# 5. Precursors (precedes) of any Composite_Anomaly with lead time <= 5 min
QUERY_FAST_COMPOSITE_PRECURSORS = PREFIXES + """
SELECT DISTINCT ?precursor ?leadTime WHERE {
    ?precursor ewat:precedes ?composite .
    ?composite rdf:type ?subClass .
    ?subClass rdfs:subClassOf* ewat:Composite_Anomaly .
    ?precursor ewat:temporalLeadTime ?leadTime .
    FILTER (?leadTime <= 300.0)
}
"""


CANONICAL_QUERIES: dict[str, str] = {
    "all_composites": QUERY_ALL_COMPOSITES,
    "downstream_of_memory_saturation": QUERY_DOWNSTREAM_MEMORY,
    "services_affected_by_cascading": QUERY_SERVICES_CASCADING,
    "signatures_sharing_heavy_features": QUERY_SHARED_HEAVY_FEATURES,
    "fast_precursors_of_composite": QUERY_FAST_COMPOSITE_PRECURSORS,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_query(
    world: World,
    query: str,
    *,
    select_vars: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run a SPARQL SELECT against an owlready2 World and return dicts.

    If ``select_vars`` is omitted, returned dict keys are positional
    ``var_0, var_1, ...``. The underlying owlready2 backend returns tuples
    of ``Thing | str | float`` values; pass-through is preserved.
    """
    raw = list(world.sparql(query))
    if not raw:
        return []
    if select_vars is None:
        ncols = len(raw[0])
        select_vars = [f"var_{i}" for i in range(ncols)]
    return [dict(zip(select_vars, row)) for row in raw]
