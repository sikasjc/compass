from __future__ import annotations

from datetime import date, datetime
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
from compass.ui.pages.strategies import (
    StrategyDraft,
    StrategyInstance,
    StrategyPool,
)
from compass.ui.pages.strategy_lab import StrategyLabInstrument


NOW = datetime(2026, 8, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
INSTRUMENT = InstrumentId.parse("SSE.510300")


class Strategies:
    def __init__(self) -> None:
        self.instance = StrategyInstance(
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
        short = configuration.strategies[0].short_window
        total_return = float(short) / 100
        metrics = SimpleNamespace(
            total_return=total_return,
            calmar_ratio=total_return,
            sharpe_ratio=total_return,
            maximum_drawdown=-0.10,
            total_turnover=0.50,
        )
        result = SimpleNamespace(fills=(object(), object()))
        return SimpleNamespace(metrics=metrics, result=result)


def test_search_space_is_bounded_and_requires_valid_windows() -> None:
    space = OptimizationSearchSpace((10, 20), (20, 30), (1, 2))
    assert space.trial_count == 6

    with pytest.raises(ValueError, match="TRIAL_LIMIT"):
        OptimizationSearchSpace(tuple(range(1, 9)), tuple(range(10, 18)), (1,))
    with pytest.raises(ValueError, match="WINDOW_SPACE"):
        OptimizationSearchSpace((20,), (10,), (1,))


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
        OptimizationSearchSpace((10, 20), (40,), (1,)),
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
    assert len(backtests.run_ids) == 5
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
