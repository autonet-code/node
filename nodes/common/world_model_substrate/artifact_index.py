"""Moved to ``nodes.common.library.artifact_index`` (this path is an alias)."""
import sys

from nodes.common.library import artifact_index as _moved

sys.modules[__name__] = _moved
