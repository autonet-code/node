"""Moved to ``nodes.common.library.charter_version`` (this path is an alias)."""
import sys

from nodes.common.library import charter_version as _moved

sys.modules[__name__] = _moved
