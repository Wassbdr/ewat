"""EWAT v5 — anomalie sémantique des logs (SentenceBERT).

Conforme à la formalisation : e(ℓ) = SentenceBERT(ℓ) ∈ ℝ^384, et
score(service, bin) = distance cosinus moyenne des logs du bin au centroïde
« normal » du service (appris sur les bins baseline).

Coût borné : on **templatise** les lignes (masque nombres/hex/timestamps) et on
n'encode que les templates **uniques** de l'épisode (les logs TT sont très
répétitifs → typiquement quelques centaines de templates pour des milliers de
lignes). Modèle léger all-MiniLM-L6-v2 (384-dim).
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

_NUM = re.compile(r"\b\d[\d.:,_-]*\b")
_HEX = re.compile(r"\b[0-9a-f]{8,}\b", re.I)
_WS = re.compile(r"\s+")

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _templatize(line: str) -> str:
    line = _HEX.sub("<H>", line)
    line = _NUM.sub("<N>", line)
    return _WS.sub(" ", line).strip()[:200]


def compute_semantic(buckets: dict, services: list[str], n_t: int,
                     baseline_bins: set[int]) -> np.ndarray:
    """Calcule la plane (N, T) d'anomalie sémantique.

    Parameters
    ----------
    buckets : dict[(svc_idx, bin)] -> list[str]
        Lignes de log par (service, bin).
    services : list[str]
    n_t : int
        Nombre de bins T.
    baseline_bins : set[int]
        Indices de bins considérés « normal » pour le centroïde par service.

    Returns
    -------
    (N, T) float32 — NaN si le service n'a pas de log dans le bin.
    """
    N = len(services)
    out = np.full((N, n_t), np.nan, np.float32)
    if not buckets:
        return out

    # templates uniques sur tout l'épisode
    cell_templates: dict[tuple[int, int], list[str]] = {}
    uniq: dict[str, int] = {}
    for (si, b), lines in buckets.items():
        tpls = [_templatize(x) for x in lines if x.strip()]
        cell_templates[(si, b)] = tpls
        for t in tpls:
            if t not in uniq:
                uniq[t] = len(uniq)
    if not uniq:
        return out

    model = _model()
    emb = model.encode(list(uniq.keys()), batch_size=256, show_progress_bar=False,
                       normalize_embeddings=True).astype(np.float32)  # (U, 384)

    # centroïde normal par service (moyenne des embeddings des bins baseline)
    centroids: dict[int, np.ndarray] = {}
    acc: dict[int, list[int]] = defaultdict(list)
    for (si, b), tpls in cell_templates.items():
        if b in baseline_bins:
            acc[si].extend(uniq[t] for t in tpls)
    for si, ids in acc.items():
        if ids:
            centroids[si] = emb[ids].mean(axis=0)

    # fallback centroïde : moyenne globale du service (si pas de baseline)
    glob: dict[int, list[int]] = defaultdict(list)
    for (si, b), tpls in cell_templates.items():
        glob[si].extend(uniq[t] for t in tpls)
    for si, ids in glob.items():
        if si not in centroids and ids:
            centroids[si] = emb[ids].mean(axis=0)

    # distance cosinus moyenne au centroïde par cellule
    for (si, b), tpls in cell_templates.items():
        if not tpls or si not in centroids:
            continue
        c = centroids[si]
        ids = [uniq[t] for t in tpls]
        # embeddings normalisés → cos sim = produit scalaire ; distance = 1 - cos
        sims = emb[ids] @ c / (np.linalg.norm(c) + 1e-9)
        out[si, b] = float(np.mean(1.0 - sims))
    return out
