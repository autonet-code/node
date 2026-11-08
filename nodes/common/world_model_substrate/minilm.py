"""Moved to ``nodes.common.library.minilm`` (this path is an alias)."""
import sys

from nodes.common.library import minilm as _moved

sys.modules[__name__] = _moved
