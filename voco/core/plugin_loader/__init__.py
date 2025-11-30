from importlib.metadata import entry_points


def discover_plugins() -> list[str]:
    loaded = []
    eps = entry_points()
    group = eps.select(group="voco.plugins") if hasattr(eps, "select") else eps.get("voco.plugins", [])

    for ep in group:
        try:
            ep.load()
            loaded.append(ep.name)
        except Exception:
            pass

    return loaded


__all__ = ["discover_plugins"]
