"""ATN — Agent framework."""
# This literal is the LOOKUP KEY for the on-chain integrity check
# (see _cache.validate), so it must never drift from the version the
# package actually ships. Read it from the installed distribution
# metadata; the literal below is only the source-checkout fallback and
# is kept in sync with pyproject.toml.
try:
    from importlib.metadata import PackageNotFoundError as _PkgNotFound
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("autonet-computer")
    except _PkgNotFound:
        __version__ = "0.8.0"
    del _PkgNotFound, _pkg_version
except Exception:
    __version__ = "0.8.0"
