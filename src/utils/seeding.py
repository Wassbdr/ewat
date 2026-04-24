"""Global random seed initialisation for reproducible EWAT runs.

Call :func:`seed_everything` once at the start of each training or collection
script to guarantee reproducibility across Python, NumPy, and PyTorch.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Set the random seed for all relevant libraries.

    Sets:
    - ``PYTHONHASHSEED`` (best-effort; export it in the shell before Python
      starts for full effect on dict/set iteration order)
    - ``random`` (Python stdlib)
    - ``numpy``
    - ``torch`` (CPU + CUDA, when available)

    Parameters
    ----------
    seed:
        Integer seed value. Configured via ``random.seed`` in
        ``configs/default.yaml``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.debug("seeding.seed_everything: torch seeded with %d", seed)
    except ImportError:
        pass  # PyTorch not installed — skip silently

    logger.info("seeding.seed_everything: seed=%d applied", seed)
