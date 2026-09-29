"""Version information for openzync-core.

Precedence: OPENZYNC_VERSION env var (stripped, ignored when empty)
> importlib.metadata version of the ``openzync`` dist > ``"0.0.0"``.

Docker/CD sets OPENZYNC_VERSION via the APP_VERSION build-arg because the
build context has no git history, so hatch-vcs cannot derive the tag.
"""

import os


def _resolve_version() -> str:
    """Resolve the package version.

    Returns:
        The OPENZYNC_VERSION env value when non-blank, else the installed
        dist version, else ``"0.0.0"`` as a last resort.
    """
    env_version = os.environ.get("OPENZYNC_VERSION", "").strip()
    if env_version:
        return env_version
    try:
        from importlib.metadata import version as _metadata_version

        return _metadata_version("openzync")
    except Exception:
        return "0.0.0"


__version__ = _resolve_version()
