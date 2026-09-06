from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import json
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from compass.domain.market import AssetType, InstrumentId
from compass.services.strategy_optimizer import (
    LocalStrategyOptimizer,
    OptimizationRequest,
    OptimizationSearchSpace,
)
from compass.strategies.base import StrategyFrequency
from compass.strategies.kronos_forecast import KronosForecastParameters
from compass.strategies.rule_dsl import DslVariable, RuleDslParameters
from compass.storage.canonical_json import canonical_json, content_hash
from compass.ui.pages.strategies import (
    StrategyDraft,
    StrategyInstance,
    StrategyPool,
    strategy_parameters_json,
)
from compass.ui.pages.strategy_lab import StrategyLabInstrument


NOW = datetime(2026, 8, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
INSTRUMENT = InstrumentId.parse("SSE.510300")


class Strategies:
    def __init__(self, instance: StrategyInstance | None = None) -> None:
        self.instance = instance or StrategyInstance(
            "strategy-dual-v1",
            "strategy-dual",
            1,
            "沪深 300 双均线",
            "dual_ma",
            "1.1.0",
            "pool-main",
            "pool-main-snapshot",
            StrategyFrequency.DAILY,
            {
                "short_window": 20,
                "long_window": 60,
                "confirmation_days": 1,
                "target_weight": "1",
            },
            True,
            NOW,
        )
        self.published: StrategyDraft | None = None

    def list(self):  # type: ignore[no-untyped-def]
        return (self.instance,)

    def pool(self, watchlist_id: str) -> StrategyPool:
        assert watchlist_id == "pool-main"
        return StrategyPool(
            "pool-main",
            "pool-main-snapshot",
            (INSTRUMENT,),
            AssetType.ETF,
            StrategyFrequency.DAILY,
        )

    def pool_instruments(self, instance_id: str) -> tuple[InstrumentId, ...]:
        assert instance_id == self.instance.instance_id
        return (INSTRUMENT,)

    def create_version(self, instance_id: str, draft: StrategyDraft) -> StrategyInstance:
        assert instance_id == self.instance.instance_id
        self.published = draft
        return StrategyInstance(
            "strategy-dual-v2",
            "strategy-dual",
            2,
            draft.name,
            draft.strategy_type,
            draft.strategy_version,
            draft.watchlist_id,
            draft.pool_snapshot_id,
            draft.frequency,
            draft.parameters,
            True,
            NOW,
        )


class Backtests:
    def __init__(self) -> None:
        self.run_ids: list[str] = []

    def instruments(self):  # type: ignore[no-untyped-def]
        return (
            StrategyLabInstrument(
                INSTRUMENT,
                "沪深300ETF",
                AssetType.ETF,
                date(2020, 1, 2),
                date(2026, 8, 11),
                1600,
            ),
        )

    def evaluate(self, run_id, configuration):  # type: ignore[no-untyped-def]
        self.run_ids.append(run_id)
        leg = configuration.strategies[0]
        if leg.strategy.value == "dual_ma":
            score_parameter = leg.short_window
        elif leg.strategy.value == "rule_dsl":
            score_parameter = leg.variables[0].value
        else:
            assert leg.kronos_parameters is not None
            score_parameter = leg.kronos_parameters.entry_return * 100
        total_return = float(score_parameter) / 100
        metrics = SimpleNamespace(
            total_return=total_return,
            calmar_ratio=total_return,
            sharpe_ratio=total_return,
            maximum_drawdown=-0.10,
            total_turnover=0.50,
        )
        result = SimpleNamespace(fills=(object(), object()))
        return SimpleNamespace(metrics=metrics, result=result)


class UnstableBacktests(Backtests):
    def evaluate(self, run_id, configuration):  # type: ignore[no-untyped-def]
        self.run_ids.append(run_id)
        short_window = configuration.strategies[0].short_window
        fold = next(
            (position for position in (1, 2, 3) if f"validate-{position}" in run_id),
            None,
        )
        if fold is None:
            score = 0.5
        else:
            score = {
                10: (2.0, -1.0, 2.0),
                20: (0.8, 0.8, 0.8),
            }[short_window][fold - 1]
        metrics = SimpleNamespace(
            total_return=score / 10,
            calmar_ratio=score,
            sharpe_ratio=score,
            maximum_drawdown=-0.10,
            total_turnover=0.50,
        )
        return SimpleNamespace(
            metrics=metrics,
            result=SimpleNamespace(fills=(object(), object())),
        )


def test_search_space_is_bounded_and_requires_valid_windows() -> None:
    space = OptimizationSearchSpace.dual_ma((10, 20), (20, 30), (1, 2))
    assert space.trial_count == 6

    with pytest.raises(ValueError, match="TRIAL_LIMIT"):
        OptimizationSearchSpace.dual_ma(tuple(range(1, 9)), tuple(range(10, 18)), (1,))
    with pytest.raises(ValueError, match="WINDOW_SPACE"):
        OptimizationSearchSpace.dual_ma((20,), (10,), (1,))


def test_optimizer_ranks_validation_runs_frozen_test_once_and_publishes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    strategies = Strategies()
    backtests = Backtests()
    path = tmp_path / "experiments.json"
    optimizer = LocalStrategyOptimizer(
        path,
        strategies=strategies,
        backtests=backtests,  # type: ignore[arg-type]
        clock=lambda: NOW,
        id_factory=lambda prefix: f"{prefix}-1",
    )
    request = OptimizationRequest(
        strategies.instance.instance_id,
        date(2020, 1, 2),
        date(2026, 8, 11),
        OptimizationSearchSpace.dual_ma((10, 20), (40,), (1,)),
    )

    optimizer.run("optimization-1", request)

    experiment = optimizer.list()[0]
    progress = optimizer.progress("optimization-1")
    assert progress is not None
    assert progress.phase == "saving"
    assert progress.current_trial == 2
    assert progress.fraction == 0.99
    assert experiment.trials[0].short_window == 20
    assert experiment.trials[0].frozen_test is not None
    assert experiment.trials[1].frozen_test is None
    assert len(experiment.trials[0].validation_folds) == 3
    assert [item.fold for item in experiment.trials[0].validation_folds] == [1, 2, 3]
    assert len(backtests.run_ids) == 9
    assert backtests.run_ids[-1] == "optimization-1-frozen-test"

    reloaded = LocalStrategyOptimizer(
        path,
        strategies=strategies,
        backtests=backtests,  # type: ignore[arg-type]
        clock=lambda: NOW,
        id_factory=lambda prefix: f"{prefix}-2",
    )
    assert reloaded.list() == optimizer.list()

    published = reloaded.publish("optimization-1")
    assert published.instance_id == "strategy-dual-v2"
    assert strategies.published is not None
    assert strategies.published.parameters["short_window"] == 20
    assert reloaded.list()[0].published_instance_id == published.instance_id


def test_optimizer_prefers_stable_performance_across_validation_windows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    strategies = Strategies()
    backtests = UnstableBacktests()
    optimizer = LocalStrategyOptimizer(
        tmp_path / "stable-experiments.json",
        strategies=strategies,
        backtests=backtests,  # type: ignore[arg-type]
        clock=lambda: NOW,
        id_factory=lambda prefix: f"{prefix}-stable",
    )

    optimizer.run(
        "optimization-stable",
        OptimizationRequest(
            strategies.instance.instance_id,
            date(2020, 1, 2),
            date(2026, 8, 11),
            OptimizationSearchSpace.dual_ma((10, 20), (40,), (1,)),
        ),
    )

    experiment = optimizer.list()[0]
    assert experiment.trials[0].short_window == 20
    assert experiment.trials[0].score == pytest.approx(0.8)
    assert experiment.trials[1].score < experiment.trials[0].score


def _rule_instance() -> StrategyInstance:
    parameters = RuleDslParameters(
        buy_expression="rsi(close, rsi_window) < threshold",
        sell_expression="rsi(close, rsi_window) > 60",
        variables=(
            DslVariable(
                name="threshold",
                value=Decimal("30"),
                minimum=Decimal("20"),
                maximum=Decimal("40"),
                step=Decimal("10"),
                optimize=True,
            ),
            DslVariable(
                name="rsi_window",
                value=Decimal("14"),
                minimum=Decimal("14"),
                maximum=Decimal("14"),
                step=Decimal("1"),
                optimize=False,
            ),
        ),
    )
    return StrategyInstance(
        "strategy-rule-v1",
        "strategy-rule",
        1,
        "RSI 规则",
        "rule_dsl",
        "1.0.0",
        "pool-main",
        "pool-main-snapshot",
        StrategyFrequency.DAILY,
        parameters.model_dump(mode="json"),
        True,
        NOW,
    )


def _kronos_instance() -> StrategyInstance:
    parameters = KronosForecastParameters(
        model_size="mini",
        device="cpu",
        lookback=64,
        horizon=1,
        rebalance_interval=20,
        entry_return=Decimal("0.01"),
        exit_return=Decimal("-0.01"),
        minimum_path_positive_ratio=Decimal("0.5"),
        trend_window=20,
        top_n=1,
        sample_count=1,
    )
    return StrategyInstance(
        "strategy-kronos-v1",
        "strategy-kronos",
        1,
        "Kronos 沪深300",
        "kronos_forecast",
        "1.0.0",
        "pool-main",
        "pool-main-snapshot",
        StrategyFrequency.DAILY,
        parameters.model_dump(mode="json"),
        True,
        NOW,
    )


@pytest.mark.parametrize(
    ("instance", "space", "parameter_name", "best_value"),
    (
        (
            _rule_instance(),
            OptimizationSearchSpace(
                {"variable.threshold": (Decimal("20"), Decimal("40"))}
            ),
            "variable.threshold",
            Decimal("40"),
        ),
        (
            _kronos_instance(),
            OptimizationSearchSpace(
                {
                    "entry_return": (Decimal("0"), Decimal("0.02")),
                    "exit_return": (Decimal("-0.01"),),
                    "minimum_path_positive_ratio": (Decimal("0.5"),),
                    "trend_window": (20,),
                    "rebalance_interval": (20,),
                }
            ),
            "entry_return",
            Decimal("0.02"),
        ),
    ),
)
def test_optimizer_supports_dsl_and_kronos_and_publishes_parameters(
    tmp_path, instance, space, parameter_name, best_value  # type: ignore[no-untyped-def]
) -> None:
    strategies = Strategies(instance)
    backtests = Backtests()
    optimizer = LocalStrategyOptimizer(
        tmp_path / "experiments.json",
        strategies=strategies,
        backtests=backtests,  # type: ignore[arg-type]
        clock=lambda: NOW,
        id_factory=lambda prefix: f"{prefix}-multi",
    )

    optimizer.run(
        "optimization-multi",
        OptimizationRequest(
            instance.instance_id,
            date(2020, 1, 2),
            date(2026, 8, 11),
            space,
        ),
    )

    experiment = optimizer.list()[0]
    assert experiment.strategy_type == instance.strategy_type
    assert experiment.trials[0].parameters[parameter_name] == best_value
    assert experiment.trials[0].frozen_test is not None
    optimizer.publish(experiment.experiment_id)
    assert strategies.published is not None
    if instance.strategy_type == "rule_dsl":
        published = RuleDslParameters.model_validate_json(
            strategy_parameters_json(strategies.published.parameters), strict=True
        )
        assert published.variable_values["threshold"] == best_value
    else:
        published = KronosForecastParameters.model_validate_json(
            strategy_parameters_json(strategies.published.parameters), strict=True
        )
        assert published.entry_return == best_value


def test_optimizer_reads_existing_schema_one_experiments(tmp_path) -> None:  # type: ignore[no-untyped-def]
    metrics = {
        "calmar_ratio": 1.2,
        "maximum_drawdown": -0.1,
        "sharpe_ratio": 0.8,
        "total_return": 0.12,
        "total_turnover": 0.3,
        "trade_count": 3,
    }
    payload_json = canonical_json(
        {
            "experiments": [
                {
                    "created_at": NOW.isoformat(),
                    "end": "2026-08-11",
                    "experiment_id": "optimization-legacy",
                    "published_instance_id": None,
                    "source_instance_id": "strategy-dual-v1",
                    "source_name": "旧双均线实验",
                    "start": "2020-01-02",
                    "training_end": "2023-12-31",
                    "trials": [
                        {
                            "confirmation_days": 1,
                            "eligible": True,
                            "frozen_test": metrics,
                            "long_window": 60,
                            "rank": 1,
                            "rejection_reason": None,
                            "score": 1.2,
                            "short_window": 20,
                            "training": metrics,
                            "validation": metrics,
                        }
                    ],
                    "validation_end": "2025-04-30",
                }
            ],
            "schema_version": 1,
        }
    )
    path = tmp_path / "experiments.json"
    path.write_text(
        json.dumps(
            {"content_hash": content_hash(payload_json), "payload_json": payload_json}
        ),
        "utf-8",
    )

    optimizer = LocalStrategyOptimizer(
        path,
        strategies=Strategies(),
        backtests=Backtests(),  # type: ignore[arg-type]
        clock=lambda: NOW,
        id_factory=lambda prefix: f"{prefix}-legacy",
    )

    experiment = optimizer.list()[0]
    assert experiment.strategy_type == "dual_ma"
    assert experiment.trials[0].parameters == {
        "confirmation_days": 1,
        "long_window": 60,
        "short_window": 20,
    }
