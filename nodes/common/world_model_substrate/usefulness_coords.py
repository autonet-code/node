"""Moved to ``nodes.common.library.usefulness_coords`` (this path is an alias)."""
import sys

from nodes.common.library import usefulness_coords as _moved

sys.modules[__name__] = _moved
