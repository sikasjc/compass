from __future__ import annotations

from pathlib import Path

import pytest

from compass.data.providers.csv_provider import CsvProvider
from compass.data.registry import ProviderRegistry


def test_registry_preserves_registration_order() -> None:
    registry = ProviderRegistry()
    first = CsvProvider(Path("first"))
    second = type("SecondProvider", (), {"name": "second"})()

    registry.register(first)
    registry.register(second)

    assert registry.names() == ("csv", "second")
    assert registry.get("csv") is first


def test_registry_rejects_duplicate_names_without_changing_precedence() -> None:
    registry = ProviderRegistry()
    first = CsvProvider(Path("first"))
    registry.register(first)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(CsvProvider(Path("second")))

    assert registry.get("csv") is first
    assert registry.names() == ("csv",)


def test_registry_unknown_name_is_deterministic() -> None:
    registry = ProviderRegistry()

    with pytest.raises(KeyError) as error:
        registry.get("missing")

    assert error.value.args == ("unknown market data provider: 'missing'",)


@pytest.mark.parametrize("name", ["", " ", "AKShare"])
def test_registry_requires_normalized_nonempty_names(name: str) -> None:
    provider = type("Provider", (), {"name": name})()

    with pytest.raises(ValueError, match="provider name"):
        ProviderRegistry().register(provider)
