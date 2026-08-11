"""Multi-dataset motor-imagery architecture research package.

The package is intentionally separate from :mod:`deepnet`: its datasets,
splits, environments, model selection, and confirmation artifacts form a new
study rather than extending an already-observed benchmark.
"""

from .config import DATASETS, DatasetSpec

__all__ = ["DATASETS", "DatasetSpec"]
