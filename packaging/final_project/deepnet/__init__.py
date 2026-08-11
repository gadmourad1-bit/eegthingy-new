"""Research package for causal geometric motor-imagery EEG decoding.

The package is intentionally isolated from the acquisition GUI and robot runtime so
offline experiments cannot accidentally depend on hardware state.  The eventual
deployment adapter lives behind the same covariance/logit interface.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
