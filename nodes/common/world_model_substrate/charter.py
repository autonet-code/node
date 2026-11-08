"""Moved to ``nodes.common.library.charter`` (this path is an alias)."""
import sys

from nodes.common.library import charter as _moved

sys.modules[__name__] = _moved
