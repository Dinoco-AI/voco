"""Plugin loader for VOCO."""


def discover_plugins(plugin_dir: str | None = None) -> list[str]:
    """Discover and load VOCO plugins.

    Currently a placeholder. Plugins must be imported manually.
    """
    return []


def load_plugin(name: str) -> bool:
    """Load a specific plugin by name.

    Currently a placeholder.
    """
    return False


__all__ = [
    "discover_plugins",
    "load_plugin",
]
