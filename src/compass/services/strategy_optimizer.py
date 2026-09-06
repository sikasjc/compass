from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import product
import json
from math import isfinite
import os
from pathlib import Path
from statistics import median, pstdev
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from compass.backtest.engine import ExecutionTiming
from compass.domain.market import InstrumentId
from compass.services.local_strategy_lab import LocalStrategyLabGateway
from compass.services.safe_display import safe_display_text, safe_identifier
from compass.storage.canonical_json import canonical_json, content_hash, decode_canonical_json
from compass.strategies.kronos_forecast import KronosForecastParameters
from compass.strategies.rule_dsl import RuleDslParameters
from compass.ui.pages.strategy_lab import (
    StrategyLabConfiguration,
    StrategyLabKind,
    StrategyLabRebalanceMode,
    StrategyLegConfiguration,
)

if TYPE_CHECKING:
    from compass.ui.pages.strategies import (
        StrategyDraft,
        StrategyInstance,
        StrategyPool,
    )


Clock = Callable[[], datetime]
IdFactory = Callable[[str], str]
_SCHEMA_VERSION = 3
_READABLE_SCHEMA_VERSIONS = {1, 2, 3}
_MAX_TRIALS = 50
_VALIDATION_FOLDS = 3
_PROGRESS_PHASES = {"preparing", "training", "validation", "frozen_test", "saving"}
_DUAL_MA_PARAMETERS = frozenset({"short_window", "long_window", "confirmation_days"})
_KRONOS_PARAMETERS = frozenset(
    {
        "entry_return",
        "exit_return",
        "minimum_path_positive_ratio",
        "trend_window",
        "rebalance_interval",
        "horizon",
    }
)
OptimizationValue = int | Decimal


def _strategy_parameters_json(parameters: Mapping[str, object]) -> str:
    from compass.ui.pages.strategies import strategy_parameters_json

    return strategy_parameters_json(parameters)


class StrategyOptimizationGateway(Protocol):
    def list(self) -> Sequence[StrategyInstance]: ...
    def pool(self, watchlist_id: str) -> StrategyPool: ...
    def pool_instruments(self, instance_id: str) -> tuple[InstrumentId, ...]: ...
    def create_version(
        self,
        instance_id: str,
        draft: StrategyDraft,
    ) -> StrategyInstance: ...


def _checked_parameter_value(value: object) -> OptimizationValue:
    if type(value) is int:
        return value
    if type(value) is Decimal and value.is_finite():
        return value
    raise TypeError("optimization values must be exact integers or finite Decimals")


def _freeze_parameter_values(
    values: Mapping[str, Sequence[OptimizationValue]],
) -> Mapping[str, tuple[OptimizationValue, ...]]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("optimization parameter space must not be empty")
    checked: dict[str, tuple[OptimizationValue, ...]] = {}
    for name, candidates in values.items():
        if type(name) is not str or not name or name != name.strip():
            raise ValueError("optimization parameter names must be stable strings")
        items = tuple(_checked_parameter_value(item) for item in candidates)
        if not items or len(set(items)) != len(items):
            raise ValueError("optimization parameter candidates must be non-empty and unique")
        checked[name] = tuple(sorted(items))
    return MappingProxyType(dict(sorted(checked.items())))


def _freeze_trial_parameters(
    values: Mapping[str, OptimizationValue],
) -> Mapping[str, OptimizationValue]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("optimization trial parameters must not be empty")
    checked = {
        name: _checked_parameter_value(value)
        for name, value in values.items()
        if type(name) is str and name and name == name.strip()
    }
    if len(checked) != len(values):
        raise ValueError("optimization trial parameter names are invalid")
    return MappingProxyType(dict(sorted(checked.items())))


@dataclass(frozen=True, slots=True)
class OptimizationSearchSpace:
    parameter_values: Mapping[str, tuple[OptimizationValue, ...]]

    def __post_init__(self) -> None:
        frozen = _freeze_parameter_values(self.parameter_values)
        object.__setattr__(self, "parameter_values", frozen)
        if self.trial_count > _MAX_TRIALS:
            raise ValueError("OPTIMIZATION_TRIAL_LIMIT_EXCEEDED")
        if not self.candidates:
            raise ValueError(
                "OPTIMIZATION_WINDOW_SPACE_INVALID"
                if set(frozen) == _DUAL_MA_PARAMETERS
                else "OPTIMIZATION_PARAMETER_SPACE_INVALID"
            )

    @classmethod
    def dual_ma(
        cls,
        short_windows: Sequence[int],
        long_windows: Sequence[int],
        confirmation_days: Sequence[int],
    ) -> OptimizationSearchSpace:
        for values in (short_windows, long_windows, confirmation_days):
            if any(type(item) is not int or item <= 0 for item in values):
                raise ValueError("dual-MA candidates must be positive integers")
        return cls(
            {
                "short_window": tuple(short_windows),
                "long_window": tuple(long_windows),
                "confirmation_days": tuple(confirmation_days),
            }
        )

    @property
    def candidates(self) -> tuple[Mapping[str, OptimizationValue], ...]:
        names = tuple(self.parameter_values)
        candidates = []
        for values in product(*(self.parameter_values[name] for name in names)):
            item = dict(zip(names, values, strict=True))
            short = item.get("short_window")
            long = item.get("long_window")
            if short is not None and long is not None and int(short) >= int(long):
                continue
            entry = item.get("entry_return")
            exit_value = item.get("exit_return")
            if entry is not None and exit_value is not None and Decimal(exit_value) >= Decimal(entry):
                continue
            candidates.append(MappingProxyType(item))
        return tuple(candidates)

    @property
    def trial_count(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True, slots=True)
class OptimizationRequest:
    source_instance_id: str
    start: date
    end: date
    search_space: OptimizationSearchSpace

    def __post_init__(self) -> None:
        safe_identifier(self.source_instance_id, label="optimization source strategy")
        if type(self.start) is not date or type(self.end) is not date:
            raise TypeError("optimization range must use exact dates")
        if self.end - self.start < timedelta(days=365):
            raise ValueError("OPTIMIZATION_RANGE_TOO_SHORT")
        if type(self.search_space) is not OptimizationSearchSpace:
            raise TypeError("optimization search space must be exact")


@dataclass(frozen=True, slots=True)
class OptimizationMetrics:
    total_return: float | None
    calmar_ratio: float | None
    sharpe_ratio: float | None
    maximum_drawdown: float | None
    total_turnover: float | None
    trade_count: int

    def __post_init__(self) -> None:
        if type(self.trade_count) is not int or self.trade_count < 0:
            raise ValueError("optimization trade count must be non-negative")


@dataclass(frozen=True, slots=True)
class OptimizationValidationFold:
    fold: int
    start: date
    end: date
    metrics: OptimizationMetrics

    def __post_init__(self) -> None:
        if type(self.fold) is not int or self.fold <= 0:
            raise ValueError("optimization fold must be a positive integer")
        if type(self.start) is not date or type(self.end) is not date or self.start > self.end:
            raise ValueError("optimization fold dates are invalid")
        if type(self.metrics) is not OptimizationMetrics:
            raise TypeError("optimization fold metrics must be exact")


@dataclass(frozen=True, slots=True)
class OptimizationTrial:
    rank: int
    parameters: Mapping[str, OptimizationValue]
    training: OptimizationMetrics
    validation: OptimizationMetrics
    frozen_test: OptimizationMetrics | None
    eligible: bool
    score: float
    rejection_reason: str | None = None
    validation_folds: tuple[OptimizationValidationFold, ...] = ()

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("optimization trial rank must be positive")
        parameters = _freeze_trial_parameters(self.parameters)
        object.__setattr__(self, "parameters", parameters)
        if _DUAL_MA_PARAMETERS.issubset(parameters) and self.short_window >= self.long_window:
            raise ValueError("optimization trial windows are invalid")
        if type(self.eligible) is not bool or type(self.score) is not float:
            raise TypeError("optimization trial eligibility and score must be exact")
        folds = tuple(self.validation_folds)
        if folds and tuple(item.fold for item in folds) != tuple(range(1, len(folds) + 1)):
            raise ValueError("optimization validation folds must be ordered without gaps")
        object.__setattr__(self, "validation_folds", folds)

    @property
    def short_window(self) -> int:
        return self._integer("short_window")

    @property
    def long_window(self) -> int:
        return self._integer("long_window")

    @property
    def confirmation_days(self) -> int:
        return self._integer("confirmation_days")

    def _integer(self, name: str) -> int:
        value = self.parameters.get(name)
        if type(value) is not int:
            raise AttributeError(name)
        return value

    @property
    def parameter_text(self) -> str:
        return " / ".join(f"{name}={value}" for name, value in self.parameters.items())


@dataclass(frozen=True, slots=True)
class OptimizationExperiment:
    experiment_id: str
    source_instance_id: str
    source_name: str
    strategy_type: str
    created_at: datetime
    start: date
    training_end: date
    validation_end: date
    end: date
    trials: tuple[OptimizationTrial, ...]
    published_instance_id: str | None = None

    def __post_init__(self) -> None:
        safe_identifier(self.experiment_id, label="optimization experiment id")
        safe_identifier(self.source_instance_id, label="optimization source strategy")
        safe_display_text(self.source_name, label="optimization source name")
        safe_identifier(self.strategy_type, label="optimization strategy type")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("optimization creation time must be timezone-aware")
        if not self.start < self.training_end < self.validation_end < self.end:
            raise ValueError("optimization split dates must be ordered")
        trials = tuple(self.trials)
        if not trials or tuple(item.rank for item in trials) != tuple(range(1, len(trials) + 1)):
            raise ValueError("optimization trials must be ranked without gaps")
        object.__setattr__(self, "trials", trials)
        if self.published_instance_id is not None:
            safe_identifier(self.published_instance_id, label="published strategy instance")


@dataclass(frozen=True, slots=True)
class OptimizationProgress:
    experiment_id: str
    phase: str
    current_trial: int
    trial_count: int
    parameters: Mapping[str, OptimizationValue] = field(default_factory=dict)
    current_fold: int = 0
    fold_count: int = _VALIDATION_FOLDS

    def __post_init__(self) -> None:
        safe_identifier(self.experiment_id, label="optimization progress experiment id")
        if self.phase not in _PROGRESS_PHASES:
            raise ValueError("optimization progress phase is invalid")
        if type(self.trial_count) is not int or self.trial_count <= 0:
            raise ValueError("optimization progress trial count must be positive")
        if (
            type(self.current_trial) is not int
            or self.current_trial < 0
            or self.current_trial > self.trial_count
        ):
            raise ValueError("optimization progress current trial is invalid")
        if self.parameters:
            object.__setattr__(self, "parameters", _freeze_trial_parameters(self.parameters))
        elif not isinstance(self.parameters, Mapping):
            raise TypeError("optimization progress parameters must be a mapping")
        if type(self.fold_count) is not int or self.fold_count <= 0:
            raise ValueError("optimization progress fold count must be positive")
        if type(self.current_fold) is not int or not 0 <= self.current_fold <= self.fold_count:
            raise ValueError("optimization progress current fold is invalid")

    @property
    def parameter_text(self) -> str:
        return " / ".join(f"{name}={value}" for name, value in self.parameters.items())

    @property
    def fraction(self) -> float:
        if self.phase == "preparing":
            return 0.0
        if self.phase == "training":
            return max(0.0, (self.current_trial - 1) / self.trial_count)
        if self.phase == "validation":
            completed = self.current_trial - 1 + self.current_fold / self.fold_count
            return min(0.98, completed / self.trial_count)
        return 0.99


class LocalStrategyOptimizer:
    """Run bounded, strategy-aware searches without polluting backtest history."""

    def __init__(
        self,
        path: Path,
        *,
        strategies: StrategyOptimizationGateway,
        backtests: LocalStrategyLabGateway,
        clock: Clock,
        id_factory: IdFactory,
    ) -> None:
        self._path = path
        self._strategies = strategies
        self._backtests = backtests
        self._clock = clock
        self._id_factory = id_factory
        self._lock = RLock()
        self._progress: dict[str, OptimizationProgress] = {}
        if not path.exists():
            self._write(())

    def list(self) -> tuple[OptimizationExperiment, ...]:
        with self._lock:
            return tuple(
                sorted(self._read(), key=lambda item: item.created_at, reverse=True)
            )

    def default_range(self, instance_id: str) -> tuple[date, date]:
        source = self._source(instance_id)
        instruments = set(self._strategies.pool_instruments(source.instance_id))
        available = tuple(
            item for item in self._backtests.instruments() if item.instrument in instruments
        )
        if not available:
            raise LookupError("OPTIMIZATION_MARKET_DATA_MISSING")
        return max(item.first_day for item in available), min(item.last_day for item in available)

    def new_experiment_id(self) -> str:
        return safe_identifier(
            self._id_factory("optimization"),
            label="optimization experiment id",
        )

    def progress(self, experiment_id: str) -> OptimizationProgress | None:
        checked = safe_identifier(experiment_id, label="optimization experiment id")
        with self._lock:
            return self._progress.get(checked)

    def run(self, experiment_id: str, request: OptimizationRequest) -> None:
        checked_id = safe_identifier(experiment_id, label="optimization experiment id")
        source = self._source(request.source_instance_id)
        if source.strategy_type not in {
            StrategyLabKind.DUAL_MA.value,
            StrategyLabKind.RULE_DSL.value,
            StrategyLabKind.KRONOS_FORECAST.value,
        }:
            raise ValueError("OPTIMIZATION_STRATEGY_UNSUPPORTED")
        self._validate_search_space(source, request.search_space)
        trial_count = request.search_space.trial_count
        self._set_progress(OptimizationProgress(checked_id, "preparing", 0, trial_count))
        span = (request.end - request.start).days
        training_end = request.start + timedelta(days=int(span * 0.40))
        validation_end = request.start + timedelta(days=int(span * 0.80))
        fold_span = (validation_end - training_end).days
        validation_ranges = tuple(
            (
                training_end
                + timedelta(days=int(fold_span * position / _VALIDATION_FOLDS))
                + timedelta(days=1),
                validation_end
                if position == _VALIDATION_FOLDS - 1
                else training_end
                + timedelta(days=int(fold_span * (position + 1) / _VALIDATION_FOLDS)),
            )
            for position in range(_VALIDATION_FOLDS)
        )
        test_start = validation_end + timedelta(days=1)
        base = self._base_configuration(source, request.start, request.end)
        trials: list[OptimizationTrial] = []
        for position, parameters in enumerate(request.search_space.candidates, start=1):
            configured = self._with_parameters(base, source.strategy_type, parameters)
            self._set_progress(
                OptimizationProgress(
                    checked_id,
                    "training",
                    position,
                    trial_count,
                    parameters,
                )
            )
            training = self._metrics(
                self._backtests.evaluate(
                    f"{checked_id}-trial-{position}-train",
                    replace(configured, start=request.start, end=training_end),
                )
            )
            validation_folds = tuple(
                self._evaluate_validation_fold(
                    checked_id,
                    position,
                    trial_count,
                    parameters,
                    configured,
                    fold,
                    fold_start,
                    fold_end,
                )
                for fold, (fold_start, fold_end) in enumerate(validation_ranges, 1)
            )
            validation = self._aggregate_metrics(
                tuple(item.metrics for item in validation_folds)
            )
            eligible, reason = self._eligible(validation, validation_folds)
            score = (
                self._robust_score(validation_folds)
                if eligible
                else -1_000_000_000.0
            )
            trials.append(
                OptimizationTrial(
                    1,
                    parameters,
                    training,
                    validation,
                    None,
                    eligible,
                    score,
                    reason,
                    validation_folds,
                )
            )
        ordered = sorted(
            trials,
            key=lambda item: (
                item.eligible,
                item.score,
                (
                    item.validation.total_return
                    if item.validation.total_return is not None
                    else -1_000_000_000.0
                ),
                (
                    item.validation.maximum_drawdown
                    if item.validation.maximum_drawdown is not None
                    else -1_000_000_000.0
                ),
            ),
            reverse=True,
        )
        ranked = tuple(replace(item, rank=rank) for rank, item in enumerate(ordered, 1))
        if ranked[0].eligible:
            best = ranked[0]
            self._set_progress(
                OptimizationProgress(
                    checked_id,
                    "frozen_test",
                    trial_count,
                    trial_count,
                    best.parameters,
                )
            )
            test_metrics = self._metrics(
                self._backtests.evaluate(
                    f"{checked_id}-frozen-test",
                    replace(
                        self._with_parameters(base, source.strategy_type, best.parameters),
                        start=test_start,
                        end=request.end,
                    ),
                )
            )
            ranked = (replace(best, frozen_test=test_metrics), *ranked[1:])
        experiment = OptimizationExperiment(
            checked_id,
            source.instance_id,
            source.name,
            source.strategy_type,
            self._timestamp(),
            request.start,
            training_end,
            validation_end,
            request.end,
            ranked,
        )
        self._set_progress(
            OptimizationProgress(
                checked_id,
                "saving",
                trial_count,
                trial_count,
            )
        )
        with self._lock:
            experiments = self._read()
            if any(item.experiment_id == checked_id for item in experiments):
                raise ValueError("OPTIMIZATION_EXPERIMENT_ID_CONFLICT")
            self._write((*experiments, experiment))

    def _set_progress(self, value: OptimizationProgress) -> None:
        with self._lock:
            self._progress[value.experiment_id] = value

    def publish(self, experiment_id: str, rank: int = 1) -> StrategyInstance:
        from compass.ui.pages.strategies import StrategyDraft

        checked = safe_identifier(experiment_id, label="optimization experiment id")
        if type(rank) is not int or rank <= 0:
            raise ValueError("optimization candidate rank must be positive")
        with self._lock:
            experiments = self._read()
            experiment = next(
                (item for item in experiments if item.experiment_id == checked),
                None,
            )
            if experiment is None:
                raise LookupError("OPTIMIZATION_EXPERIMENT_MISSING")
            if experiment.published_instance_id is not None:
                raise ValueError("OPTIMIZATION_EXPERIMENT_ALREADY_PUBLISHED")
            trial = next((item for item in experiment.trials if item.rank == rank), None)
            if trial is None or not trial.eligible:
                raise ValueError("OPTIMIZATION_CANDIDATE_NOT_ELIGIBLE")
            source = self._source(experiment.source_instance_id)
            pool = self._strategies.pool(source.watchlist_id)
            parameters = dict(source.parameters)
            parameters = self._published_parameters(source.strategy_type, parameters, trial)
            published = self._strategies.create_version(
                source.instance_id,
                StrategyDraft(
                    name=f"{source.name} · 调优",
                    strategy_type=source.strategy_type,
                    strategy_version=source.strategy_version,
                    watchlist_id=source.watchlist_id,
                    pool_snapshot_id=pool.snapshot_id,
                    frequency=source.frequency,
                    parameters=parameters,
                ),
            )
            updated = replace(experiment, published_instance_id=published.instance_id)
            self._write(
                tuple(updated if item.experiment_id == checked else item for item in experiments)
            )
            return published

    def _source(self, instance_id: str) -> StrategyInstance:
        checked = safe_identifier(instance_id, label="optimization source strategy")
        matches = tuple(item for item in self._strategies.list() if item.instance_id == checked)
        if len(matches) != 1:
            raise LookupError("OPTIMIZATION_SOURCE_MISSING")
        return matches[0]

    def _base_configuration(
        self,
        source: StrategyInstance,
        start: date,
        end: date,
    ) -> StrategyLabConfiguration:
        instruments = tuple(self._strategies.pool_instruments(source.instance_id))
        if not instruments:
            raise LookupError("OPTIMIZATION_MARKET_DATA_MISSING")
        signal = instruments[0]
        target_weight = Decimal(str(source.parameters.get("target_weight", "1")))
        kind = StrategyLabKind(source.strategy_type)
        leg_arguments: dict[str, object] = {}
        if kind is StrategyLabKind.DUAL_MA:
            leg_arguments.update(
                short_window=self._integer_parameter(source, "short_window"),
                long_window=self._integer_parameter(source, "long_window"),
                confirmation_days=self._integer_parameter(source, "confirmation_days"),
            )
        elif kind is StrategyLabKind.RULE_DSL:
            parameters = RuleDslParameters.model_validate_json(
                _strategy_parameters_json(source.parameters), strict=True
            )
            leg_arguments.update(
                buy_expression=parameters.buy_expression,
                sell_expression=parameters.sell_expression,
                variables=parameters.variables,
                dsl_rules=parameters.rules,
            )
        elif kind is StrategyLabKind.KRONOS_FORECAST:
            leg_arguments["kronos_parameters"] = KronosForecastParameters.model_validate_json(
                _strategy_parameters_json(source.parameters), strict=True
            )
        return StrategyLabConfiguration(
            strategies=(
                StrategyLegConfiguration(
                    strategy_id=source.instance_id,
                    strategy=kind,
                    instruments=instruments,
                    budget=target_weight,
                    signal_instrument=(
                        signal
                        if kind in {StrategyLabKind.DUAL_MA, StrategyLabKind.RULE_DSL}
                        else None
                    ),
                    template_instance_id=source.instance_id,
                    template_name=source.name,
                    **leg_arguments,  # type: ignore[arg-type]
                ),
            ),
            benchmark=signal,
            start=start,
            end=end,
            initial_cash=Decimal("1000000.00"),
            commission_rate=Decimal("0.0003"),
            minimum_commission=Decimal("5"),
            slippage_bps=Decimal("2"),
            execution_timing=ExecutionTiming.NEXT_OPEN,
            rebalance_mode=StrategyLabRebalanceMode.SIGNAL_CHANGE,
        )

    @staticmethod
    def _with_parameters(
        configuration: StrategyLabConfiguration,
        strategy_type: str,
        parameters: Mapping[str, OptimizationValue],
    ) -> StrategyLabConfiguration:
        leg = configuration.strategies[0]
        if strategy_type == StrategyLabKind.DUAL_MA.value:
            leg = replace(
                leg,
                short_window=int(parameters["short_window"]),
                long_window=int(parameters["long_window"]),
                confirmation_days=int(parameters["confirmation_days"]),
            )
        elif strategy_type == StrategyLabKind.RULE_DSL.value:
            updated_variables = tuple(
                variable.model_copy(
                    update={"value": Decimal(parameters.get(f"variable.{variable.name}", variable.value))}
                )
                for variable in leg.variables
            )
            leg = replace(leg, variables=updated_variables)
        elif strategy_type == StrategyLabKind.KRONOS_FORECAST.value:
            if leg.kronos_parameters is None:
                raise ValueError("OPTIMIZATION_KRONOS_PARAMETERS_MISSING")
            payload = leg.kronos_parameters.model_dump()
            payload.update(parameters)
            leg = replace(
                leg,
                kronos_parameters=KronosForecastParameters.model_validate(payload, strict=True),
            )
        else:
            raise ValueError("OPTIMIZATION_STRATEGY_UNSUPPORTED")
        return replace(configuration, strategies=(leg,))

    @staticmethod
    def _validate_search_space(
        source: StrategyInstance,
        search_space: OptimizationSearchSpace,
    ) -> None:
        names = set(search_space.parameter_values)
        if source.strategy_type == StrategyLabKind.DUAL_MA.value:
            if names != _DUAL_MA_PARAMETERS:
                raise ValueError("OPTIMIZATION_DUAL_MA_SPACE_INVALID")
            if any(
                type(value) is not int or value <= 0
                for values in search_space.parameter_values.values()
                for value in values
            ):
                raise ValueError("OPTIMIZATION_DUAL_MA_SPACE_INVALID")
            return
        if source.strategy_type == StrategyLabKind.RULE_DSL.value:
            parsed_rule = RuleDslParameters.model_validate_json(
                _strategy_parameters_json(source.parameters), strict=True
            )
            variables = {
                f"variable.{item.name}": item
                for item in parsed_rule.optimization_variables
            }
            if not names or not names.issubset(variables):
                raise ValueError("OPTIMIZATION_DSL_SPACE_INVALID")
            for name, values in search_space.parameter_values.items():
                variable = variables[name]
                for raw in values:
                    value = Decimal(raw)
                    if (
                        value < variable.minimum
                        or value > variable.maximum
                    ):
                        raise ValueError("OPTIMIZATION_DSL_SPACE_INVALID")
            return
        if source.strategy_type == StrategyLabKind.KRONOS_FORECAST.value:
            if not names or not names.issubset(_KRONOS_PARAMETERS):
                raise ValueError("OPTIMIZATION_KRONOS_SPACE_INVALID")
            parsed_kronos = KronosForecastParameters.model_validate_json(
                _strategy_parameters_json(source.parameters), strict=True
            )
            for candidate in search_space.candidates:
                payload = parsed_kronos.model_dump()
                payload.update(candidate)
                try:
                    KronosForecastParameters.model_validate(payload, strict=True)
                except ValueError:
                    raise ValueError("OPTIMIZATION_KRONOS_SPACE_INVALID") from None
            return
        raise ValueError("OPTIMIZATION_STRATEGY_UNSUPPORTED")

    @staticmethod
    def _published_parameters(
        strategy_type: str,
        source_parameters: Mapping[str, object],
        trial: OptimizationTrial,
    ) -> dict[str, object]:
        if strategy_type == StrategyLabKind.DUAL_MA.value:
            result = dict(source_parameters)
            result.update(trial.parameters)
            return result
        if strategy_type == StrategyLabKind.RULE_DSL.value:
            parsed_rule = RuleDslParameters.model_validate_json(
                _strategy_parameters_json(source_parameters), strict=True
            )
            variables = tuple(
                variable.model_copy(
                    update={
                        "value": Decimal(
                            trial.parameters.get(
                                f"variable.{variable.name}", variable.value
                            )
                        )
                    }
                )
                for variable in parsed_rule.variables
            )
            updated_rule = parsed_rule.model_copy(update={"variables": variables})
            return dict(updated_rule.model_dump(mode="json"))
        if strategy_type == StrategyLabKind.KRONOS_FORECAST.value:
            parsed_kronos = KronosForecastParameters.model_validate_json(
                _strategy_parameters_json(source_parameters), strict=True
            )
            payload = parsed_kronos.model_dump()
            payload.update(trial.parameters)
            updated_kronos = KronosForecastParameters.model_validate(payload, strict=True)
            return dict(updated_kronos.model_dump(mode="json"))
        raise ValueError("OPTIMIZATION_STRATEGY_UNSUPPORTED")

    @staticmethod
    def _integer_parameter(source: StrategyInstance, name: str) -> int:
        value = source.parameters[name]
        if type(value) is not int:
            raise TypeError(f"optimization parameter {name} must be an exact integer")
        return value

    @staticmethod
    def _metrics(report: object) -> OptimizationMetrics:
        metrics = getattr(report, "metrics")
        result = getattr(report, "result")
        return OptimizationMetrics(
            metrics.total_return,
            metrics.calmar_ratio,
            metrics.sharpe_ratio,
            metrics.maximum_drawdown,
            metrics.total_turnover,
            len(result.fills),
        )

    def _evaluate_validation_fold(
        self,
        experiment_id: str,
        trial: int,
        trial_count: int,
        parameters: Mapping[str, OptimizationValue],
        configured: StrategyLabConfiguration,
        fold: int,
        start: date,
        end: date,
    ) -> OptimizationValidationFold:
        self._set_progress(
            OptimizationProgress(
                experiment_id,
                "validation",
                trial,
                trial_count,
                parameters,
                current_fold=fold,
            )
        )
        metrics = self._metrics(
            self._backtests.evaluate(
                f"{experiment_id}-trial-{trial}-validate-{fold}",
                replace(configured, start=start, end=end),
            )
        )
        return OptimizationValidationFold(fold, start, end, metrics)

    @staticmethod
    def _median(values: Sequence[float | None]) -> float | None:
        usable = tuple(value for value in values if value is not None and isfinite(value))
        return None if not usable else float(median(usable))

    @classmethod
    def _aggregate_metrics(
        cls,
        folds: Sequence[OptimizationMetrics],
    ) -> OptimizationMetrics:
        if not folds:
            raise ValueError("optimization validation folds must not be empty")
        drawdowns = tuple(
            item.maximum_drawdown
            for item in folds
            if item.maximum_drawdown is not None and isfinite(item.maximum_drawdown)
        )
        return OptimizationMetrics(
            cls._median(tuple(item.total_return for item in folds)),
            cls._median(tuple(item.calmar_ratio for item in folds)),
            cls._median(tuple(item.sharpe_ratio for item in folds)),
            None if not drawdowns else min(drawdowns),
            cls._median(tuple(item.total_turnover for item in folds)),
            sum(item.trade_count for item in folds),
        )

    @staticmethod
    def _eligible(
        metrics: OptimizationMetrics,
        folds: Sequence[OptimizationValidationFold] = (),
    ) -> tuple[bool, str | None]:
        if folds:
            evaluable = sum(
                item.metrics.trade_count > 0 and item.metrics.total_return is not None
                for item in folds
            )
            if evaluable < 2:
                return False, "至少需要两个有成交的滚动验证窗口"
        if metrics.trade_count == 0:
            return False, "验证区间没有成交"
        if metrics.total_return is None:
            return False, "验证区间收益不可计算"
        if metrics.maximum_drawdown is None or metrics.maximum_drawdown < -0.40:
            return False, "验证区间最大回撤超过 40%"
        return True, None

    @staticmethod
    def _robust_score(folds: Sequence[OptimizationValidationFold]) -> float:
        calmars = tuple(
            item.metrics.calmar_ratio
            for item in folds
            if item.metrics.calmar_ratio is not None and isfinite(item.metrics.calmar_ratio)
        )
        values = calmars
        if len(values) < 2:
            values = tuple(
                item.metrics.total_return
                for item in folds
                if item.metrics.total_return is not None and isfinite(item.metrics.total_return)
            )
        if not values:
            return -1_000_000_000.0
        return float(median(values) - (pstdev(values) if len(values) > 1 else 0.0))

    def _timestamp(self) -> datetime:
        value = self._clock()
        if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("optimization clock must be timezone-aware")
        return value

    def _read(self) -> tuple[OptimizationExperiment, ...]:
        try:
            wrapper = json.loads(self._path.read_text("utf-8"))
            payload = decode_canonical_json(wrapper["payload_json"], wrapper["content_hash"])
            schema_version = payload.get("schema_version")
            if type(schema_version) is not int or schema_version not in _READABLE_SCHEMA_VERSIONS:
                raise ValueError
            raw_experiments = payload["experiments"]
            if not isinstance(raw_experiments, list):
                raise ValueError
            return tuple(
                self._decode(item, schema_version)
                for item in raw_experiments
                if isinstance(item, Mapping)
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("OPTIMIZATION_REGISTRY_INTEGRITY") from None

    def _write(self, experiments: Sequence[OptimizationExperiment]) -> None:
        payload_json = canonical_json(
            {
                "experiments": [self._encode(item) for item in experiments],
                "schema_version": _SCHEMA_VERSION,
            }
        )
        document = canonical_json(
            {"content_hash": content_hash(payload_json), "payload_json": payload_json}
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(document, "utf-8")
            os.replace(temporary, self._path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _metric_payload(value: OptimizationMetrics) -> dict[str, object]:
        return {
            "calmar_ratio": value.calmar_ratio,
            "maximum_drawdown": value.maximum_drawdown,
            "sharpe_ratio": value.sharpe_ratio,
            "total_return": value.total_return,
            "total_turnover": value.total_turnover,
            "trade_count": value.trade_count,
        }

    @classmethod
    def _encode(cls, value: OptimizationExperiment) -> dict[str, object]:
        return {
            "created_at": value.created_at.isoformat(),
            "end": value.end.isoformat(),
            "experiment_id": value.experiment_id,
            "published_instance_id": value.published_instance_id,
            "strategy_type": value.strategy_type,
            "source_instance_id": value.source_instance_id,
            "source_name": value.source_name,
            "start": value.start.isoformat(),
            "training_end": value.training_end.isoformat(),
            "trials": [
                {
                    "eligible": item.eligible,
                    "frozen_test": (
                        None
                        if item.frozen_test is None
                        else cls._metric_payload(item.frozen_test)
                    ),
                    "parameters": {
                        name: {
                            "type": "integer" if type(parameter) is int else "decimal",
                            "value": parameter if type(parameter) is int else str(parameter),
                        }
                        for name, parameter in item.parameters.items()
                    },
                    "rank": item.rank,
                    "rejection_reason": item.rejection_reason,
                    "score": item.score,
                    "training": cls._metric_payload(item.training),
                    "validation": cls._metric_payload(item.validation),
                    "validation_folds": [
                        {
                            "end": fold.end.isoformat(),
                            "fold": fold.fold,
                            "metrics": cls._metric_payload(fold.metrics),
                            "start": fold.start.isoformat(),
                        }
                        for fold in item.validation_folds
                    ],
                }
                for item in value.trials
            ],
            "validation_end": value.validation_end.isoformat(),
        }

    @staticmethod
    def _decode_metrics(value: Mapping[str, object]) -> OptimizationMetrics:
        return OptimizationMetrics(
            value["total_return"],  # type: ignore[arg-type]
            value["calmar_ratio"],  # type: ignore[arg-type]
            value["sharpe_ratio"],  # type: ignore[arg-type]
            value["maximum_drawdown"],  # type: ignore[arg-type]
            value["total_turnover"],  # type: ignore[arg-type]
            value["trade_count"],  # type: ignore[arg-type]
        )

    @classmethod
    def _decode(
        cls,
        value: Mapping[str, object],
        schema_version: int,
    ) -> OptimizationExperiment:
        raw_trials = value["trials"]
        if not isinstance(raw_trials, list):
            raise ValueError
        trials = []
        for raw in raw_trials:
            if not isinstance(raw, Mapping):
                raise ValueError
            frozen = raw["frozen_test"]
            if schema_version == 1:
                parameters: Mapping[str, OptimizationValue] = {
                    "short_window": raw["short_window"],
                    "long_window": raw["long_window"],
                    "confirmation_days": raw["confirmation_days"],
                }
            else:
                raw_parameters = raw["parameters"]
                if not isinstance(raw_parameters, Mapping):
                    raise ValueError
                decoded_parameters: dict[str, OptimizationValue] = {}
                for name, encoded in raw_parameters.items():
                    if type(name) is not str or not isinstance(encoded, Mapping):
                        raise ValueError
                    kind = encoded.get("type")
                    parameter = encoded.get("value")
                    if kind == "integer" and type(parameter) is int:
                        decoded_parameters[name] = parameter
                    elif kind == "decimal" and type(parameter) is str:
                        parsed = Decimal(parameter)
                        if not parsed.is_finite() or str(parsed) != parameter:
                            raise ValueError
                        decoded_parameters[name] = parsed
                    else:
                        raise ValueError
                parameters = decoded_parameters
            raw_folds = raw.get("validation_folds", []) if schema_version >= 3 else []
            if not isinstance(raw_folds, list):
                raise ValueError
            validation_folds = tuple(
                OptimizationValidationFold(
                    fold=item["fold"],
                    start=date.fromisoformat(item["start"]),
                    end=date.fromisoformat(item["end"]),
                    metrics=cls._decode_metrics(item["metrics"]),
                )
                for item in raw_folds
                if isinstance(item, Mapping)
            )
            if len(validation_folds) != len(raw_folds):
                raise ValueError
            trials.append(
                OptimizationTrial(
                    raw["rank"],
                    parameters,
                    cls._decode_metrics(raw["training"]),
                    cls._decode_metrics(raw["validation"]),
                    None if frozen is None else cls._decode_metrics(frozen),
                    raw["eligible"],
                    raw["score"],
                    raw["rejection_reason"],
                    validation_folds,
                )
            )
        return OptimizationExperiment(
            experiment_id=value["experiment_id"],  # type: ignore[arg-type]
            source_instance_id=value["source_instance_id"],  # type: ignore[arg-type]
            source_name=value["source_name"],  # type: ignore[arg-type]
            strategy_type=(
                StrategyLabKind.DUAL_MA.value
                if schema_version == 1
                else value["strategy_type"]  # type: ignore[arg-type]
            ),
            created_at=datetime.fromisoformat(value["created_at"]),  # type: ignore[arg-type]
            start=date.fromisoformat(value["start"]),  # type: ignore[arg-type]
            training_end=date.fromisoformat(value["training_end"]),  # type: ignore[arg-type]
            validation_end=date.fromisoformat(value["validation_end"]),  # type: ignore[arg-type]
            end=date.fromisoformat(value["end"]),  # type: ignore[arg-type]
            trials=tuple(trials),
            published_instance_id=value["published_instance_id"],  # type: ignore[arg-type]
        )
