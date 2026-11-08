"""Moved to ``nodes.common.library.tool_manifest`` (this path is an alias)."""
import sys

from nodes.common.library import tool_manifest as _moved

sys.modules[__name__] = _moved
