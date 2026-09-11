"""Standalone four-command SSVEP data collector.

The package deliberately does not import the motor-imagery collector.  Its only
project-level integration point is the launcher entry in :mod:`main`.
"""

from .config import COMMANDS, PHASES, Settings

__all__ = ["COMMANDS", "PHASES", "Settings"]
