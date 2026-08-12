from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]
import pytest

from compass.backtest.broker import InitialPosition
from compass.backtest.engine import BacktestEngine, BacktestRequest, DecisionTarget
from compass.backtest.market_rules import (
    MarketRuleBook,
    MarketRuleProfile,
    OddLotSellPolicy,
    PriceLimitMode,
    SettlementMode,
)
from compass.domain.market import AssetType, Exchange, Instrument, InstrumentId
from compass.risk.engine import RiskEngine
from compass.risk.rules import StrategyBudgetRule, TradabilityRule, TurnoverCapRule
from compass.strategies.base import StrategyContext


DAY = date(2026, 7, 20)
FIRST = InstrumentId.parse("SSE.510300")
SECOND = InstrumentId.parse("SSE.510500")


def _instrument(symbol: InstrumentId) -> Instrument:
    return Instrument(symbol, AssetType.ETF, 100, False)


def _bars(price: str = "10") -> pd.DataFrame:
    value = Decimal(price)
    return pd.DataFrame(
        {
            "open": [value, value],
            "high": [value, value],
            "low": [value, value],
            "close": [value, value],
            "volume": [100_000, 100_000],
            "amount": [Decimal("1000000"), Decimal("1000000")],
        },
        index=pd.to_datetime(["2026-07-20", "2026-07-21"]),
    )


def _profile() -> MarketRuleProfile:
    return MarketRuleProfile(
        profile_id="ETF-RISK",
        exchange=Exchange.SSE,
        asset_type=AssetType.ETF,
        effective_from=date(2026, 1, 1),
        effective_to=None,
        buy_lot_size=100,
        odd_lot_sell_policy=OddLotSellPolicy.POSITION_REMAINDER_ONLY,
        settlement_mode=SettlementMode.T_PLUS_ONE,
        same_day_sell_eligible=False,
        price_limit_mode=PriceLimitMode.PERCENTAGE,
        price_limit_rate=Decimal("0.10"),
        risk_warning_price_limit_rate=Decimal("0.05"),
        commission_rate=Decimal("0"),
        minimum_commission=Decimal("0"),
        sell_stamp_duty_rate=Decimal("0"),
        transfer_fee_rate=Decimal("0"),
        slippage_bps=Decimal("0"),
        maximum_volume_participation=Decimal("1"),
        fee_profile_confirmed=True,
    )


class _TwoTargets:
    def targets(self, context: StrategyContext) -> DecisionTarget:
        return DecisionTarget(
            weights={SECOND: Decimal("0.40"), FIRST: Decimal("0.40")},
            sleeve_weights={
                SECOND: {"portfolio": Decimal("1")},
                FIRST: {"portfolio": Decimal("1")},
            },
        )


class _RotateTarget:
    def targets(self, context: StrategyContext) -> DecisionTarget:
        return DecisionTarget(
            weights={SECOND: Decimal("0.40")},
            sleeve_weights={SECOND: {"portfolio": Decimal("1")}},
        )


class _PlannedExposureTargets:
    def __init__(self, *, reverse: bool) -> None:
        self._reverse = reverse

    def targets(self, context: StrategyContext) -> DecisionTarget:
        items = [(FIRST, Decimal("0.31")), (SECOND, Decimal("0.40"))]
        if self._reverse:
            items.reverse()
        weights = dict(items)
        return DecisionTarget(
            weights=weights,
            sleeve_weights={symbol: {"portfolio": Decimal("1")} for symbol in weights},
        )


def _request(risk_engine: RiskEngine) -> BacktestRequest:
    return BacktestRequest(
        run_id="portfolio-risk",
        sessions=(DAY, date(2026, 7, 21)),
        instruments={SECOND: _instrument(SECOND), FIRST: _instrument(FIRST)},
        bars={SECOND: _bars(), FIRST: _bars()},
        initial_cash=Decimal("10000.00"),
        initial_positions=(),
        corporate_actions=(),
        decision_source=_TwoTargets(),
        risk_engine=risk_engine,
        rule_book=MarketRuleBook((_profile(),)),
    )


@pytest.mark.parametrize(
    ("rules", "expected_codes"),
    [
        ((TurnoverCapRule(maximum_turnover=Decimal("0.50")),), ("TURNOVER_CAP",)),
        ((StrategyBudgetRule(maximum_weight=Decimal("0.50")),), ("STRATEGY_BUDGET",)),
        (
            (
                StrategyBudgetRule(maximum_weight=Decimal("0.60")),
                TurnoverCapRule(maximum_turnover=Decimal("0.50")),
            ),
            ("STRATEGY_BUDGET", "TURNOVER_CAP"),
        ),
    ],
)
def test_backtest_accumulates_portfolio_risk_state_in_canonical_symbol_order(
    rules: tuple[object, ...], expected_codes: tuple[str, ...]
) -> None:
    result = BacktestEngine().run(_request(RiskEngine(rules)))  # type: ignore[arg-type]
    traces = tuple(trace for trace in result.risk_traces if trace.decision_date == DAY)

    assert tuple(trace.instrument for trace in traces) == (FIRST, SECOND)
    assert sum((trace.final_weight for trace in traces), Decimal("0")) <= Decimal("0.50")
    assert tuple(item.code for item in traces[0].adjustments) == ()
    assert tuple(item.code for item in traces[1].adjustments) == expected_codes
    assert traces[1].final_weight == Decimal("0.10")


def test_backtest_blocked_sell_keeps_current_exposure_in_later_strategy_budget() -> None:
    held_bars = _bars()
    held_bars["suspended"] = [True, False]
    request = BacktestRequest(
        run_id="blocked-holding-risk",
        sessions=(DAY, date(2026, 7, 21)),
        instruments={SECOND: _instrument(SECOND), FIRST: _instrument(FIRST)},
        bars={SECOND: _bars(), FIRST: held_bars},
        initial_cash=Decimal("4000.00"),
        initial_positions=(
            InitialPosition(FIRST, 600, 600, Decimal("10"), Decimal("10")),
        ),
        corporate_actions=(),
        decision_source=_RotateTarget(),
        risk_engine=RiskEngine(
            (TradabilityRule(), StrategyBudgetRule(maximum_weight=Decimal("0.70")))
        ),
        rule_book=MarketRuleBook((_profile(),)),
    )

    result = BacktestEngine().run(request)
    traces = tuple(trace for trace in result.risk_traces if trace.decision_date == DAY)

    assert traces[0].instrument == FIRST
    assert traces[0].result.blocked is True
    assert traces[1].instrument == SECOND
    assert traces[1].final_weight == Decimal("0.10")
    assert tuple(item.code for item in traces[1].adjustments) == ("STRATEGY_BUDGET",)


@pytest.mark.parametrize(
    ("rules", "expected_later_weight", "expected_codes"),
    [
        (
            (StrategyBudgetRule(maximum_weight=Decimal("0.70")),),
            Decimal("0.30"),
            ("STRATEGY_BUDGET",),
        ),
        (
            (
                StrategyBudgetRule(maximum_weight=Decimal("0.70")),
                TurnoverCapRule(maximum_turnover=Decimal("0.25")),
            ),
            Decimal("0.25"),
            ("STRATEGY_BUDGET", "TURNOVER_CAP"),
        ),
    ],
)
def test_backtest_accumulates_cent_rounded_planned_quantity_exposure(
    rules: tuple[object, ...],
    expected_later_weight: Decimal,
    expected_codes: tuple[str, ...],
) -> None:
    signatures: list[tuple[tuple[InstrumentId, Decimal, tuple[str, ...]], ...]] = []
    for reverse in (False, True):
        symbols = (SECOND, FIRST) if reverse else (FIRST, SECOND)
        result = BacktestEngine().run(
            BacktestRequest(
                run_id=f"planned-exposure-{reverse}",
                sessions=(DAY, date(2026, 7, 21)),
                instruments={symbol: _instrument(symbol) for symbol in symbols},
                bars={symbol: _bars("1") for symbol in symbols},
                initial_cash=Decimal("400.00"),
                initial_positions=(
                    InitialPosition(FIRST, 600, 600, Decimal("1"), Decimal("1")),
                ),
                corporate_actions=(),
                decision_source=_PlannedExposureTargets(reverse=reverse),
                risk_engine=RiskEngine(rules),  # type: ignore[arg-type]
                rule_book=MarketRuleBook((_profile(),)),
            )
        )
        traces = tuple(trace for trace in result.risk_traces if trace.decision_date == DAY)
        signatures.append(
            tuple(
                (
                    trace.instrument,
                    trace.final_weight,
                    tuple(item.code for item in trace.adjustments),
                )
                for trace in traces
            )
        )
        first_order = next(
            order
            for order in result.orders
            if order.created_on == DAY and order.instrument == FIRST
        )

        assert first_order.quantity == 200
        assert traces[1].final_weight == expected_later_weight
        assert tuple(item.code for item in traces[1].adjustments) == expected_codes
        assert Decimal("0.40") + traces[1].final_weight <= Decimal("0.70")

    assert signatures[0] == signatures[1]
