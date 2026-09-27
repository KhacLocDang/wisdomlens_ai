"""Provider factory and registry for WisdomLens AI."""

from __future__ import annotations

from app.services.providers.gemini import GeminiProvider
from app.services.providers.claude import ClaudeProvider

_REGISTRY: dict[str, type] = {
    "gemini": GeminiProvider,
    "claude": ClaudeProvider,
}


def list_supported_providers() -> list[str]:
    """Return the list of supported provider names."""
    return list(_REGISTRY.keys())


def get_provider(name: str):
    """Instantiate and return a provider by name.

    Raises ``ValueError`` if the provider is unknown.
    """
    key = (name or "").strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown provider: {name!r}. Supported: {list(_REGISTRY)}")
    return cls()


def list_models_for_provider(provider: str | None) -> list[dict]:
    """Return model list for a given provider (or gemini by default)."""
    from app.services.wisdom_service import list_gemini_models

    if provider is None or provider.strip().lower() == "gemini":
        return list_gemini_models()

    prov = get_provider(provider)
    return prov.list_models()


def list_all_models() -> list[dict]:
    """Return models for all registered providers."""
    result: list[dict] = []
    for name in _REGISTRY:
        prov = _REGISTRY[name]()
        result.extend(prov.list_models())
    return result

