from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import product
import json
import os
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from compass.backtest.engine import ExecutionTiming
from compass.domain.market import InstrumentId
from compass.services.local_strategy_lab import LocalStrategyLabGateway
from compass.services.safe_display import safe_display_text, safe_identifier
from compass.storage.canonical_json import canonical_json, content_hash, decode_canonical_json
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
_SCHEMA_VERSION = 1
_MAX_TRIALS = 50
_PROGRESS_PHASES = {"preparing", "training", "validation", "frozen_test", "saving"}


class StrategyOptimizationGateway(Protocol):
    def list(self) -> Sequence[StrategyInstance]: ...
    def pool(self, watchlist_id: str) -> StrategyPool: ...
    def pool_instruments(self, instance_id: str) -> tuple[InstrumentId, ...]: ...
    def create_version(
        self,
        instance_id: str,
        draft: StrategyDraft,
    ) -> StrategyInstance: ...


@dataclass(frozen=True, slots=True)
class OptimizationSearchSpace:
    short_windows: tuple[int, ...]
    long_windows: tuple[int, ...]
    confirmation_days: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("short_windows", "long_windows", "confirmation_days"):
            values = tuple(getattr(self, name))
            if (
                not values
                or any(type(item) is not int or item <= 0 for item in values)
                or values != tuple(sorted(set(values)))
            ):
                raise ValueError(f"{name} must contain unique sorted positive integers")
            object.__setattr__(self, name, values)
        if self.trial_count > _MAX_TRIALS:
            raise ValueError("OPTIMIZATION_TRIAL_LIMIT_EXCEEDED")
        if not any(
            short < long
            for short, long in product(self.short_windows, self.long_windows)
        ):
            raise ValueError("OPTIMIZATION_WINDOW_SPACE_INVALID")

    @property
    def trial_count(self) -> int:
        return sum(
            1
            for short, long, _ in product(
                self.short_windows,
                self.long_windows,
                self.confirmation_days,
            )
            if short < long
        )


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
class OptimizationTrial:
    rank: int
    short_window: int
    long_window: int
    confirmation_days: int
    training: OptimizationMetrics
    validation: OptimizationMetrics
    frozen_test: OptimizationMetrics | None
    eligible: bool
    score: float
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("optimization trial rank must be positive")
        if self.short_window >= self.long_window:
            raise ValueError("optimization trial windows are invalid")
        if type(self.eligible) is not bool or type(self.score) is not float:
            raise TypeError("optimization trial eligibility and score must be exact")


@dataclass(frozen=True, slots=True)
class OptimizationExperiment:
    experiment_id: str
    source_instance_id: str
    source_name: str
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
    short_window: int | None = None
    long_window: int | None = None
    confirmation_days: int | None = None

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
        parameters = (
            self.short_window,
            self.long_window,
            self.confirmation_days,
        )
        if any(value is not None and (type(value) is not int or value <= 0) for value in parameters):
            raise ValueError("optimization progress parameters must be positive integers")

    @property
    def fraction(self) -> float:
        if self.phase == "preparing":
            return 0.0
        if self.phase == "training":
            return max(0.0, (self.current_trial - 1) / self.trial_count)
        if self.phase == "validation":
            return min(0.98, (self.current_trial - 0.5) / self.trial_count)
        return 0.99


class LocalStrategyOptimizer:
    """Run bounded dual-MA searches without polluting ordinary backtest history."""

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
        trial_count = request.search_space.trial_count
        self._set_progress(
            OptimizationProgress(checked_id, "preparing", 0, trial_count)
        )
        source = self._source(request.source_instance_id)
        if source.strategy_type != StrategyLabKind.DUAL_MA.value:
            raise ValueError("OPTIMIZATION_STRATEGY_UNSUPPORTED")
        span = (request.end - request.start).days
        training_end = request.start + timedelta(days=int(span * 0.60))
        validation_end = request.start + timedelta(days=int(span * 0.80))
        validation_start = training_end + timedelta(days=1)
        test_start = validation_end + timedelta(days=1)
        base = self._base_configuration(source, request.start, request.end)
        trials: list[OptimizationTrial] = []
        for position, (short, long, confirmation) in enumerate(
            (
                values
                for values in product(
                    request.search_space.short_windows,
                    request.search_space.long_windows,
                    request.search_space.confirmation_days,
                )
                if values[0] < values[1]
            ),
            start=1,
        ):
            configured = self._with_parameters(base, short, long, confirmation)
            self._set_progress(
                OptimizationProgress(
                    checked_id,
                    "training",
                    position,
                    trial_count,
                    short,
                    long,
                    confirmation,
                )
            )
            training = self._metrics(
                self._backtests.evaluate(
                    f"{checked_id}-trial-{position}-train",
                    replace(configured, start=request.start, end=training_end),
                )
            )
            self._set_progress(
                OptimizationProgress(
                    checked_id,
                    "validation",
                    position,
                    trial_count,
                    short,
                    long,
                    confirmation,
                )
            )
            validation = self._metrics(
                self._backtests.evaluate(
                    f"{checked_id}-trial-{position}-validate",
                    replace(configured, start=validation_start, end=validation_end),
                )
            )
            eligible, reason = self._eligible(validation)
            score = self._score(validation) if eligible else -1_000_000_000.0
            trials.append(
                OptimizationTrial(
                    1,
                    short,
                    long,
                    confirmation,
                    training,
                    validation,
                    None,
                    eligible,
                    score,
                    reason,
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
                    best.short_window,
                    best.long_window,
                    best.confirmation_days,
                )
            )
            test_metrics = self._metrics(
                self._backtests.evaluate(
                    f"{checked_id}-frozen-test",
                    replace(
                        self._with_parameters(
                            base,
                            best.short_window,
                            best.long_window,
                            best.confirmation_days,
                        ),
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
            parameters.update(
                short_window=trial.short_window,
                long_window=trial.long_window,
                confirmation_days=trial.confirmation_days,
            )
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
        return StrategyLabConfiguration(
            strategies=(
                StrategyLegConfiguration(
                    strategy_id=source.instance_id,
                    strategy=StrategyLabKind.DUAL_MA,
                    instruments=instruments,
                    budget=target_weight,
                    signal_instrument=signal,
                    short_window=self._integer_parameter(source, "short_window"),
                    long_window=self._integer_parameter(source, "long_window"),
                    confirmation_days=self._integer_parameter(source, "confirmation_days"),
                    template_instance_id=source.instance_id,
                    template_name=source.name,
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
        short: int,
        long: int,
        confirmation: int,
    ) -> StrategyLabConfiguration:
        leg = replace(
            configuration.strategies[0],
            short_window=short,
            long_window=long,
            confirmation_days=confirmation,
        )
        return replace(configuration, strategies=(leg,))

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

    @staticmethod
    def _eligible(metrics: OptimizationMetrics) -> tuple[bool, str | None]:
        if metrics.trade_count == 0:
            return False, "验证区间没有成交"
        if metrics.total_return is None:
            return False, "验证区间收益不可计算"
        if metrics.maximum_drawdown is None or metrics.maximum_drawdown < -0.40:
            return False, "验证区间最大回撤超过 40%"
        return True, None

    @staticmethod
    def _score(metrics: OptimizationMetrics) -> float:
        return (
            metrics.calmar_ratio
            if metrics.calmar_ratio is not None
            else metrics.total_return
            if metrics.total_return is not None
            else -1_000_000_000.0
        )

    def _timestamp(self) -> datetime:
        value = self._clock()
        if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("optimization clock must be timezone-aware")
        return value

    def _read(self) -> tuple[OptimizationExperiment, ...]:
        try:
            wrapper = json.loads(self._path.read_text("utf-8"))
            payload = decode_canonical_json(wrapper["payload_json"], wrapper["content_hash"])
            if payload.get("schema_version") != _SCHEMA_VERSION:
                raise ValueError
            raw_experiments = payload["experiments"]
            if not isinstance(raw_experiments, list):
                raise ValueError
            return tuple(
                self._decode(item)
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
            "source_instance_id": value.source_instance_id,
            "source_name": value.source_name,
            "start": value.start.isoformat(),
            "training_end": value.training_end.isoformat(),
            "trials": [
                {
                    "confirmation_days": item.confirmation_days,
                    "eligible": item.eligible,
                    "frozen_test": (
                        None
                        if item.frozen_test is None
                        else cls._metric_payload(item.frozen_test)
                    ),
                    "long_window": item.long_window,
                    "rank": item.rank,
                    "rejection_reason": item.rejection_reason,
                    "score": item.score,
                    "short_window": item.short_window,
                    "training": cls._metric_payload(item.training),
                    "validation": cls._metric_payload(item.validation),
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
    def _decode(cls, value: Mapping[str, object]) -> OptimizationExperiment:
        raw_trials = value["trials"]
        if not isinstance(raw_trials, list):
            raise ValueError
        trials = []
        for raw in raw_trials:
            if not isinstance(raw, Mapping):
                raise ValueError
            frozen = raw["frozen_test"]
            trials.append(
                OptimizationTrial(
                    raw["rank"],
                    raw["short_window"],
                    raw["long_window"],
                    raw["confirmation_days"],
                    cls._decode_metrics(raw["training"]),
                    cls._decode_metrics(raw["validation"]),
                    None if frozen is None else cls._decode_metrics(frozen),
                    raw["eligible"],
                    raw["score"],
                    raw["rejection_reason"],
                )
            )
        return OptimizationExperiment(
            value["experiment_id"],  # type: ignore[arg-type]
            value["source_instance_id"],  # type: ignore[arg-type]
            value["source_name"],  # type: ignore[arg-type]
            datetime.fromisoformat(value["created_at"]),  # type: ignore[arg-type]
            date.fromisoformat(value["start"]),  # type: ignore[arg-type]
            date.fromisoformat(value["training_end"]),  # type: ignore[arg-type]
            date.fromisoformat(value["validation_end"]),  # type: ignore[arg-type]
            date.fromisoformat(value["end"]),  # type: ignore[arg-type]
            tuple(trials),
            value["published_instance_id"],  # type: ignore[arg-type]
        )
