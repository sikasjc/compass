from __future__ import annotations

from compass.data.base import MarketDataProvider


class ProviderRegistry:
    """Insertion-ordered provider registry with explicit duplicate handling."""

    def __init__(self) -> None:
        self._providers: dict[str, MarketDataProvider] = {}

    def register(self, provider: MarketDataProvider) -> None:
        name = provider.name
        if not name or name != name.strip().lower():
            raise ValueError("provider name must be normalized lowercase text")
        if name in self._providers:
            raise ValueError(f"provider {name!r} is already registered")
        self._providers[name] = provider

    def get(self, name: str) -> MarketDataProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(f"unknown market data provider: {name!r}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(self._providers)
