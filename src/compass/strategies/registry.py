from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from compass.strategies.base import Strategy, StrategyMetadata, StrategyParameters


_STRATEGY_TYPE = re.compile(r"^[a-z][a-z0-9_]*$")
StrategyFactory = Callable[..., Strategy]


class StrategyRegistry:
    """Explicit registry for built-in strategies; plugin discovery is intentionally absent."""

    def __init__(self) -> None:
        self._factories: dict[str, StrategyFactory] = {}
        self._metadata: dict[str, StrategyMetadata] = {}

    def register(self, strategy_type: str, factory: StrategyFactory) -> None:
        if not isinstance(strategy_type, str) or not _STRATEGY_TYPE.fullmatch(strategy_type):
            raise ValueError("strategy type must be a non-empty lower snake identifier")
        if not callable(factory):
            raise ValueError("strategy factory must be callable")
        if strategy_type in self._factories:
            raise ValueError(f"strategy type already registered: {strategy_type}")
        metadata = self._factory_metadata(factory)
        if metadata.strategy_type != strategy_type:
            raise ValueError(
                f"factory metadata strategy type {metadata.strategy_type!r} does not match {strategy_type!r}"
            )
        self._validate_parameters_type(metadata.parameters_type)
        self._factories[strategy_type] = factory
        self._metadata[strategy_type] = metadata

    def strategy_types(self) -> tuple[str, ...]:
        """Return registered type identifiers in deterministic order."""

        return tuple(sorted(self._factories))

    def lookup(self, strategy_type: str) -> StrategyFactory:
        try:
            return self._factories[strategy_type]
        except KeyError:
            raise KeyError(f"unknown strategy type: {strategy_type}") from None

    def describe(self, strategy_type: str) -> StrategyMetadata:
        """Get immutable metadata without constructing a strategy instance."""

        try:
            return self._metadata[strategy_type]
        except KeyError:
            raise KeyError(f"unknown strategy type: {strategy_type}") from None

    def list_metadata(self) -> tuple[StrategyMetadata, ...]:
        """List immutable strategy definitions in deterministic type order."""

        return tuple(self._metadata[strategy_type] for strategy_type in self.strategy_types())

    def create(self, strategy_type: str, *args: Any, **kwargs: Any) -> Strategy:
        strategy = self.lookup(strategy_type)(*args, **kwargs)
        self._validate_strategy(strategy_type, self.describe(strategy_type), strategy)
        return strategy

    @staticmethod
    def _factory_metadata(factory: StrategyFactory) -> StrategyMetadata:
        metadata = getattr(factory, "metadata", None)
        if type(metadata) is not StrategyMetadata:
            raise TypeError("strategy factory must expose exact StrategyMetadata as metadata")
        return metadata

    @staticmethod
    def _validate_parameters_type(parameters_type: type[StrategyParameters]) -> None:
        config = parameters_type.model_config
        if config.get("frozen") is not True:
            raise ValueError("strategy parameters_type must remain frozen")
        if config.get("extra") != "forbid":
            raise ValueError("strategy parameters_type must forbid extra fields")
        for name, field in parameters_type.model_fields.items():
            if not isinstance(field.description, str) or not field.description.strip():
                raise ValueError(f"strategy parameter {name!r} must have a description")

    @staticmethod
    def _validate_strategy(
        strategy_type: str, metadata: StrategyMetadata, strategy: object
    ) -> None:
        if not isinstance(strategy, Strategy):
            raise TypeError("strategy factory returned an object that does not implement Strategy")
        if type(strategy.strategy_type) is not str:
            raise TypeError("strategy strategy_type must be an exact str")
        if type(strategy.metadata) is not StrategyMetadata:
            raise TypeError("strategy metadata must be exact StrategyMetadata")
        if (
            type(strategy.minimum_history) is not int
            or strategy.minimum_history <= 0
        ):
            raise ValueError("strategy minimum_history must be a positive exact int")
        if (
            type(strategy.required_history) is not int
            or strategy.required_history < metadata.minimum_history
        ):
            raise ValueError(
                "strategy required_history must be an exact int at least metadata.minimum_history"
            )
        if not isinstance(strategy.parameters_type, type) or not issubclass(
            strategy.parameters_type, StrategyParameters
        ):
            raise TypeError("strategy parameters_type must subclass StrategyParameters")
        if not callable(strategy.generate_targets):
            raise TypeError("strategy generate_targets must be callable")
        if strategy.strategy_type != strategy_type:
            raise ValueError(
                f"factory strategy type {strategy.strategy_type!r} does not match {strategy_type!r}"
            )
        if strategy.metadata != metadata:
            raise ValueError("factory strategy metadata does not match registered metadata")
        if strategy.parameters_type is not metadata.parameters_type:
            raise ValueError("factory strategy parameters_type does not match registered metadata")
        if strategy.minimum_history != metadata.minimum_history:
            raise ValueError("factory strategy minimum_history does not match registered metadata")
