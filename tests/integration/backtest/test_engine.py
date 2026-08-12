import json
from dataclasses import FrozenInstanceError, replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from compass.backtest.engine import (
    BacktestEngine,
    BacktestRequest,
    BacktestResult,
    DecisionTarget,
    ExecutionTiming,
)
from compass.backtest.broker import InitialPosition
from compass.backtest.market_rules import (
    MarketRuleBook,
    MarketRuleProfile,
    OddLotSellPolicy,
    PriceLimitMode,
    SettlementMode,
)
from compass.backtest.orders import CancellationReason, OrderSide, OrderStatus
from compass.backtest.orders import round_money
from compass.domain.market import AssetType, Exchange, Instrument, InstrumentId
from compass.domain.trading import CorporateAction
from compass.risk.engine import RiskEngine
from compass.risk.rules import CashReserveRule, SingleStockCapRule, TradabilityRule
from compass.services.local_read_gateways import (
    _StrategyDecisionSource,
    _benchmark_curve,
    _trusted_backtest_sessions,
)
from compass.strategies.base import StrategyContext
from compass.strategies.base import StrategyDecision, StrategyDecisionStatus


FIXTURE = Path(__file__).parents[2] / "fixtures" / "backtest_scenario.json"
SYMBOL = InstrumentId.parse("SSE.600000")
INSTRUMENT = Instrument(SYMBOL, AssetType.STOCK, 100, False)


def _profile() -> MarketRuleProfile:
    return MarketRuleProfile(
        profile_id="SSE-ZERO-COST",
        exchange=Exchange.SSE,
        asset_type=AssetType.STOCK,
        effective_from=date(2020, 1, 1),
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


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [Decimal("4.00"), Decimal("4.10")],
            "high": [Decimal("4.10"), Decimal("4.30")],
            "low": [Decimal("3.90"), Decimal("4.00")],
            "close": [Decimal("4.00"), Decimal("4.20")],
            "volume": [100_000, 100_000],
            "amount": [Decimal("400000"), Decimal("420000")],
            "suspended": [False, False],
            "limit_up": [Decimal("4.40"), Decimal("4.40")],
            "limit_down": [Decimal("3.60"), Decimal("3.60")],
        },
        index=pd.to_datetime(["2026-07-20", "2026-07-21"]),
    )


class _AssertingSource:
    def __init__(self, schedule: dict[date, Decimal]) -> None:
        self.schedule = dict(schedule)
        self.seen: list[date] = []

    def targets(self, context: StrategyContext) -> DecisionTarget:
        assert context.history(SYMBOL).index.max().date() <= context.as_of
        assert not (context.history(SYMBOL).index.date > context.as_of).any()
        self.seen.append(context.as_of)
        return DecisionTarget(
            weights={SYMBOL: self.schedule.get(context.as_of, Decimal("0"))},
            sleeve_weights={SYMBOL: {"main": Decimal("1")}},
        )


def _request(
    source: object,
    *,
    risk_engine: RiskEngine | None = None,
    execution_timing: ExecutionTiming = ExecutionTiming.NEXT_OPEN,
) -> BacktestRequest:
    return BacktestRequest(
        run_id="scenario-run",
        sessions=(date(2026, 7, 20), date(2026, 7, 21)),
        instruments={SYMBOL: INSTRUMENT},
        bars={SYMBOL: _bars()},
        initial_cash=Decimal("1000"),
        initial_positions=(),
        corporate_actions=(),
        decision_source=source,  # type: ignore[arg-type]
        risk_engine=RiskEngine(()) if risk_engine is None else risk_engine,
        rule_book=MarketRuleBook((_profile(),)),
        execution_timing=execution_timing,
    )


def test_close_signal_cannot_fill_until_next_open_and_uses_fixture() -> None:
    expected = json.loads(FIXTURE.read_text("utf-8"))
    source = _AssertingSource({date(2026, 7, 20): Decimal("1")})

    result = BacktestEngine().run(_request(source))

    assert result.orders[0].created_on.isoformat() == expected["created_on"]
    assert result.fills[0].trading_day.isoformat() == expected["fill_day"]
    assert result.fills[0].price == Decimal(expected["fill_price"])
    assert result.fills[0].quantity == expected["fill_quantity"]
    assert result.ledger[-1].cash == Decimal(expected["final_cash"])
    assert source.seen == [date(2026, 7, 20), date(2026, 7, 21)]


def test_close_signal_can_be_configured_to_fill_at_next_close() -> None:
    source = _AssertingSource({date(2026, 7, 20): Decimal("1")})

    result = BacktestEngine().run(
        _request(source, execution_timing=ExecutionTiming.NEXT_CLOSE)
    )

    assert result.fills[0].trading_day == date(2026, 7, 21)
    assert result.fills[0].price == Decimal("4.2000")
    assert result.fills[0].price != _bars().iloc[1]["open"]


def test_skipped_decision_preserves_existing_target_instead_of_liquidating() -> None:
    """Dropping ``preserve_unspecified`` would recreate non-rebalance liquidation."""

    class Source:
        def targets(self, context: StrategyContext) -> DecisionTarget:
            if context.as_of == date(2026, 7, 20):
                return DecisionTarget({SYMBOL: Decimal("1")}, {SYMBOL: {"main": Decimal("1")}})
            return DecisionTarget({}, {}, preserve_unspecified=True)

    result = BacktestEngine().run(_request(Source()))

    assert tuple(fill.side for fill in result.fills) == (OrderSide.BUY,)
    assert result.ledger[-1].positions[0].quantity == result.fills[0].quantity


def test_local_strategy_adapter_preserves_targets_for_non_rebalance_status() -> None:
    """Ignoring StrategyDecision.status would turn weekly SKIPPED into cash."""

    class WeeklyStrategy:
        def generate_targets(self, context: StrategyContext) -> StrategyDecision:
            del context
            return StrategyDecision.empty(
                StrategyDecisionStatus.SKIPPED,
                "NOT_REBALANCE_SESSION",
            )

    context = StrategyContext(
        as_of=date(2026, 7, 21),
        bars={SYMBOL: _bars()},
        instruments=(SYMBOL,),
        account_equity=Decimal("1000"),
    )

    target = _StrategyDecisionSource(WeeklyStrategy()).targets(context)  # type: ignore[arg-type]

    assert target.weights == {}
    assert target.preserve_unspecified is True


def test_backtest_clock_keeps_trusted_session_missing_from_one_instrument() -> None:
    """Using the all-instrument intersection would silently delete July 21."""

    other = InstrumentId.parse("SZSE.159915")
    missing_middle = _bars().drop(pd.Timestamp("2026-07-21"))
    healthy = pd.concat(
        [
            _bars(),
            _bars().iloc[[-1]].rename(index={pd.Timestamp("2026-07-21"): pd.Timestamp("2026-07-22")}),
        ]
    )
    expected = pd.to_datetime(["2026-07-20", "2026-07-21", "2026-07-22"])

    sessions = _trusted_backtest_sessions(
        {SYMBOL: missing_middle, other: healthy},
        lambda request: expected,
    )

    assert sessions == (
        date(2026, 7, 20),
        date(2026, 7, 21),
        date(2026, 7, 22),
    )


def test_benchmark_curve_keeps_frozen_sessions_when_a_bar_is_missing() -> None:
    frame = _bars().drop(pd.Timestamp("2026-07-21"))

    curve = _benchmark_curve(
        frame,
        (date(2026, 7, 20), date(2026, 7, 21), date(2026, 7, 22)),
    )

    assert tuple(point.day for point in curve) == (
        "2026-07-20",
        "2026-07-21",
        "2026-07-22",
    )
    assert curve[1].value == curve[0].value


def test_strategy_context_uses_first_session_of_current_aggregated_holding() -> None:
    """Removing acquisition tracking would make max_holding_sessions inert again."""

    seen_holding_since: list[date | None] = []

    class Source:
        def targets(self, context: StrategyContext) -> DecisionTarget:
            holding = context.holding(SYMBOL)
            if holding is not None:
                seen_holding_since.append(holding.holding_since)
            return DecisionTarget(
                {SYMBOL: Decimal("1")},
                {SYMBOL: {"main": Decimal("1")}},
            )

    BacktestEngine().run(_request(Source()))

    assert seen_holding_since == [date(2026, 7, 21)]


def test_engine_integrates_real_risk_engine_and_keeps_trace_on_order() -> None:
    source = _AssertingSource({date(2026, 7, 20): Decimal("0.80")})
    risk = RiskEngine((SingleStockCapRule(maximum_weight=Decimal("0.40")),))

    result = BacktestEngine().run(_request(source, risk_engine=risk))

    assert result.orders[0].risk_codes == ("SINGLE_STOCK_CAP",)
    assert result.orders[0].quantity == 100
    assert result.risk_traces[0].final_weight == Decimal("0.40")


def test_order_generation_uses_next_session_profile_and_records_all_rule_versions() -> None:
    source = _AssertingSource({date(2026, 7, 20): Decimal("0.70")})
    old = replace(
        _profile(),
        profile_id="OLD",
        effective_to=date(2026, 7, 20),
    )
    new = replace(
        _profile(),
        profile_id="NEW",
        effective_from=date(2026, 7, 21),
        buy_lot_size=200,
        fee_profile_confirmed=False,
    )
    request = BacktestRequest(
        run_id="boundary-run",
        sessions=(date(2026, 7, 20), date(2026, 7, 21)),
        instruments={SYMBOL: INSTRUMENT},
        bars={SYMBOL: _bars()},
        initial_cash=Decimal("2000"),
        initial_positions=(),
        corporate_actions=(),
        decision_source=source,
        risk_engine=RiskEngine(()),
        rule_book=MarketRuleBook((old, new)),
    )

    result = BacktestEngine().run(request)

    assert result.orders[0].quantity == 200
    assert result.orders[0].status is OrderStatus.FILLED
    assert result.used_profile_ids == ("NEW", "OLD")
    assert result.warnings == (
        "COMPARABLE_PRICE_FACTOR_MISSING:SSE.600000",
        "FEE_PROFILE_UNCONFIRMED:NEW",
    )


def test_existing_position_omitted_from_target_is_sold() -> None:
    class EmptySource:
        def targets(self, context: StrategyContext) -> DecisionTarget:
            return DecisionTarget(weights={}, sleeve_weights={})

    request = _request(EmptySource())
    request = BacktestRequest(
        run_id=request.run_id,
        sessions=request.sessions,
        instruments=request.instruments,
        bars=request.bars,
        initial_cash=Decimal("0"),
        initial_positions=(
            # available at the first close and next open
            InitialPosition(SYMBOL, 100, 100, Decimal("4"), Decimal("4")),
        ),
        corporate_actions=(),
        decision_source=request.decision_source,
        risk_engine=request.risk_engine,
        rule_book=request.rule_book,
    )

    result = BacktestEngine().run(request)

    assert result.orders[0].side.value == "sell"
    assert result.fills[0].trading_day == date(2026, 7, 21)


def test_last_close_order_without_next_session_is_cancelled() -> None:
    source = _AssertingSource({date(2026, 7, 21): Decimal("1")})

    result = BacktestEngine().run(_request(source))

    last = result.orders[-1]
    assert last.created_on == date(2026, 7, 21)
    assert last.status is OrderStatus.CANCELLED
    assert last.cancellation_reason is CancellationReason.NO_NEXT_SESSION


def test_blocked_risk_result_never_turns_into_an_implicit_liquidation() -> None:
    bars = _bars()
    bars.loc[pd.Timestamp("2026-07-20"), "suspended"] = True

    class EmptySource:
        def targets(self, context: StrategyContext) -> DecisionTarget:
            return DecisionTarget(weights={}, sleeve_weights={})

    base = _request(EmptySource(), risk_engine=RiskEngine((TradabilityRule(),)))
    request = BacktestRequest(
        run_id=base.run_id,
        sessions=base.sessions,
        instruments=base.instruments,
        bars={SYMBOL: bars},
        initial_cash=Decimal("0"),
        initial_positions=(InitialPosition(SYMBOL, 100, 100, Decimal("4"), Decimal("4")),),
        corporate_actions=(),
        decision_source=base.decision_source,
        risk_engine=base.risk_engine,
        rule_book=base.rule_book,
    )

    result = BacktestEngine().run(request)

    assert not any(order.created_on == date(2026, 7, 20) for order in result.orders)
    assert result.risk_traces[0].result.blocked


def test_repeated_runs_are_stable_and_results_are_immutable() -> None:
    left = BacktestEngine().run(_request(_AssertingSource({date(2026, 7, 20): Decimal("1")})))
    right = BacktestEngine().run(_request(_AssertingSource({date(2026, 7, 20): Decimal("1")})))

    assert left == right
    with pytest.raises(FrozenInstanceError):
        left.run_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="orders must be a tuple"):
        BacktestResult(
            run_id=left.run_id,
            orders=list(left.orders),  # type: ignore[arg-type]
            fills=left.fills,
            ledger=left.ledger,
            risk_traces=left.risk_traces,
            used_profile_ids=left.used_profile_ids,
            warnings=left.warnings,
        )


def test_request_rejects_out_of_range_and_duplicate_sessions() -> None:
    source = _AssertingSource({})
    values = dict(
        run_id="bad",
        instruments={SYMBOL: INSTRUMENT},
        bars={SYMBOL: _bars()},
        initial_cash=Decimal("1000"),
        initial_positions=(),
        corporate_actions=(),
        decision_source=source,
        risk_engine=RiskEngine(()),
        rule_book=MarketRuleBook((_profile(),)),
    )
    with pytest.raises(ValueError, match="unique and increasing"):
        BacktestRequest(sessions=(date(2026, 7, 20), date(2026, 7, 20)), **values)
    with pytest.raises(ValueError, match="outside available bars"):
        BacktestRequest(sessions=(date(2026, 7, 19),), **values)


def test_directional_sell_sizing_never_crosses_risk_approved_target() -> None:
    class HalfTarget:
        def targets(self, context: StrategyContext) -> DecisionTarget:
            return DecisionTarget(weights={SYMBOL: Decimal("0.50")}, sleeve_weights={SYMBOL: {}})

    bars = _bars()
    bars.loc[:, ["open", "high", "low", "close"]] = Decimal("3")
    request = BacktestRequest(
        run_id="directional",
        sessions=(date(2026, 7, 20), date(2026, 7, 21)),
        instruments={SYMBOL: INSTRUMENT},
        bars={SYMBOL: bars},
        initial_cash=Decimal("400"),
        initial_positions=(InitialPosition(SYMBOL, 200, 200, Decimal("3"), Decimal("3")),),
        corporate_actions=(),
        decision_source=HalfTarget(),
        risk_engine=RiskEngine(()),
        rule_book=MarketRuleBook((_profile(),)),
    )

    result = BacktestEngine().run(request)

    assert not any(order.side.value == "sell" for order in result.orders)
    assert result.ledger[-1].positions[0].quantity == 200


def test_etf_price_limits_are_derived_from_previous_close_when_provider_omits_them() -> None:
    symbol = InstrumentId.parse("SSE.510300")
    instrument = Instrument(symbol, AssetType.ETF, 100, False)
    profile = replace(
        _profile(),
        profile_id="SSE-ETF-ZERO-COST",
        asset_type=AssetType.ETF,
        risk_warning_price_limit_rate=None,
    )
    bars = _bars().drop(columns=["limit_up", "limit_down"])
    bars["price_limit_rate"] = Decimal("0.10")
    bars.loc[pd.Timestamp("2026-07-20"), ["open", "high", "low", "close"]] = Decimal(
        "10"
    )
    bars.loc[pd.Timestamp("2026-07-21"), ["open", "high", "low", "close"]] = Decimal(
        "11"
    )

    class BuyEtf:
        def targets(self, context: StrategyContext) -> DecisionTarget:
            return DecisionTarget(
                weights={symbol: Decimal("1")},
                sleeve_weights={symbol: {"main": Decimal("1")}},
            )

    result = BacktestEngine().run(
        BacktestRequest(
            run_id="derived-etf-limit",
            sessions=(date(2026, 7, 20), date(2026, 7, 21)),
            instruments={symbol: instrument},
            bars={symbol: bars},
            initial_cash=Decimal("10000"),
            initial_positions=(),
            corporate_actions=(),
            decision_source=BuyEtf(),
            risk_engine=RiskEngine(()),
            rule_book=MarketRuleBook((profile,)),
        )
    )

    assert result.fills == ()
    assert result.orders[0].cancellation_reason is CancellationReason.LIMIT_UP


def test_chinext_etf_uses_attested_twenty_percent_regime_not_blanket_ten() -> None:
    symbol = InstrumentId.parse("SZSE.159915")
    instrument = Instrument(symbol, AssetType.ETF, 100, False)
    profile = replace(
        _profile(),
        profile_id="SZSE-CHINEXT-ETF-ZERO-COST",
        exchange=Exchange.SZSE,
        asset_type=AssetType.ETF,
        risk_warning_price_limit_rate=None,
    )
    bars = _bars().drop(columns=["limit_up", "limit_down"])
    bars.loc[pd.Timestamp("2026-07-20"), ["open", "high", "low", "close"]] = Decimal(
        "10"
    )
    bars.loc[pd.Timestamp("2026-07-21"), ["open", "high", "low", "close"]] = Decimal(
        "11.5"
    )
    bars["previous_close"] = [Decimal("10"), Decimal("10")]
    bars["price_limit_rate"] = [Decimal("0.20"), Decimal("0.20")]

    class BuyEtf:
        def targets(self, context: StrategyContext) -> DecisionTarget:
            return DecisionTarget(
                weights={symbol: Decimal("1")},
                sleeve_weights={symbol: {"main": Decimal("1")}},
            )

    result = BacktestEngine().run(
        BacktestRequest(
            run_id="chinext-etf-limit",
            sessions=(date(2026, 7, 20), date(2026, 7, 21)),
            instruments={symbol: instrument},
            bars={symbol: bars},
            initial_cash=Decimal("10000"),
            initial_positions=(),
            corporate_actions=(),
            decision_source=BuyEtf(),
            risk_engine=RiskEngine(()),
            rule_book=MarketRuleBook((profile,)),
        )
    )

    assert result.fills
    assert result.orders[0].cancellation_reason is not CancellationReason.LIMIT_UP


def test_corporate_action_day_rejects_generic_previous_close_as_limit_reference() -> None:
    symbol = InstrumentId.parse("SSE.510300")
    instrument = Instrument(symbol, AssetType.ETF, 100, False)
    profile = replace(
        _profile(),
        profile_id="SSE-ETF-ZERO-COST",
        asset_type=AssetType.ETF,
        risk_warning_price_limit_rate=None,
    )
    bars = _bars().drop(columns=["limit_up", "limit_down"])
    bars["previous_close"] = [Decimal("10"), Decimal("10")]

    class BuyEtf:
        def targets(self, context: StrategyContext) -> DecisionTarget:
            return DecisionTarget(
                weights={symbol: Decimal("1")},
                sleeve_weights={symbol: {"main": Decimal("1")}},
            )

    request = BacktestRequest(
            run_id="action-reference-unknown",
            sessions=(date(2026, 7, 20), date(2026, 7, 21)),
            instruments={symbol: instrument},
            bars={symbol: bars},
            initial_cash=Decimal("10000"),
            initial_positions=(),
            corporate_actions=(
                CorporateAction(
                    symbol,
                    date(2026, 7, 21),
                    cash_dividend_per_share=Decimal("0.5"),
                ),
            ),
            decision_source=BuyEtf(),
            risk_engine=RiskEngine(()),
            rule_book=MarketRuleBook((profile,)),
    )
    result = BacktestEngine().run(request)

    assert result.fills == ()
    assert result.orders[0].cancellation_reason is (
        CancellationReason.MARKET_STATUS_UNKNOWN
    )
    attested = bars.copy()
    attested["exchange_reference_price"] = [Decimal("10"), Decimal("10")]
    attested["price_limit_rate"] = [Decimal("0.10"), Decimal("0.10")]
    trusted = BacktestEngine().run(
        BacktestRequest(
            run_id="action-reference-attested",
            sessions=(date(2026, 7, 20), date(2026, 7, 21)),
            instruments={symbol: instrument},
            bars={symbol: attested},
            initial_cash=Decimal("10000"),
            initial_positions=(),
            corporate_actions=request.corporate_actions,
            decision_source=BuyEtf(),
            risk_engine=RiskEngine(()),
            rule_book=MarketRuleBook((profile,)),
        )
    )
    assert trusted.orders[0].cancellation_reason is not (
        CancellationReason.MARKET_STATUS_UNKNOWN
    )


def test_stock_without_risk_warning_metadata_fails_closed_before_execution() -> None:
    bars = _bars().drop(columns=["limit_up", "limit_down"])

    result = BacktestEngine().run(
        BacktestRequest(
            run_id="unknown-stock-limit-state",
            sessions=(date(2026, 7, 20), date(2026, 7, 21)),
            instruments={SYMBOL: INSTRUMENT},
            bars={SYMBOL: bars},
            initial_cash=Decimal("1000"),
            initial_positions=(),
            corporate_actions=(),
            decision_source=_AssertingSource(
                {date(2026, 7, 20): Decimal("1")}
            ),
            risk_engine=RiskEngine(()),
            rule_book=MarketRuleBook((_profile(),)),
        )
    )

    assert result.fills == ()
    assert result.orders[0].cancellation_reason is (
        CancellationReason.MARKET_STATUS_UNKNOWN
    )


def test_stock_without_known_listing_regime_does_not_guess_a_standard_limit() -> None:
    bars = _bars().drop(columns=["limit_up", "limit_down"])
    bars["risk_warning"] = False

    result = BacktestEngine().run(
        BacktestRequest(
            run_id="unknown-stock-listing-regime",
            sessions=(date(2026, 7, 20), date(2026, 7, 21)),
            instruments={SYMBOL: INSTRUMENT},
            bars={SYMBOL: bars},
            initial_cash=Decimal("1000"),
            initial_positions=(),
            corporate_actions=(),
            decision_source=_AssertingSource(
                {date(2026, 7, 20): Decimal("1")}
            ),
            risk_engine=RiskEngine(()),
            rule_book=MarketRuleBook((_profile(),)),
        )
    )

    assert result.fills == ()
    assert result.orders[0].cancellation_reason is (
        CancellationReason.MARKET_STATUS_UNKNOWN
    )


@pytest.mark.parametrize(
    ("symbol_text", "risk_warning", "limit_price"),
    (
        ("SSE.600000", True, Decimal("10.5")),
        ("SSE.688001", None, Decimal("12")),
    ),
    ids=("main-board-risk-warning-five-percent", "star-board-twenty-percent"),
)
def test_known_stock_regimes_apply_board_and_risk_warning_limits(
    symbol_text: str,
    risk_warning: bool | None,
    limit_price: Decimal,
) -> None:
    symbol = InstrumentId.parse(symbol_text)
    instrument = Instrument(symbol, AssetType.STOCK, 100, False)
    bars = _bars().drop(columns=["limit_up", "limit_down"])
    bars.loc[pd.Timestamp("2026-07-20"), ["open", "high", "low", "close"]] = Decimal(
        "10"
    )
    bars.loc[pd.Timestamp("2026-07-21"), ["open", "high", "low", "close"]] = limit_price
    bars["risk_warning"] = risk_warning
    bars["listing_regime_known"] = True

    class BuyStock:
        def targets(self, context: StrategyContext) -> DecisionTarget:
            return DecisionTarget(
                weights={symbol: Decimal("1")},
                sleeve_weights={symbol: {"main": Decimal("1")}},
            )

    result = BacktestEngine().run(
        BacktestRequest(
            run_id=f"known-limit-{symbol.code}",
            sessions=(date(2026, 7, 20), date(2026, 7, 21)),
            instruments={symbol: instrument},
            bars={symbol: bars},
            initial_cash=Decimal("10000"),
            initial_positions=(),
            corporate_actions=(),
            decision_source=BuyStock(),
            risk_engine=RiskEngine(()),
            rule_book=MarketRuleBook((_profile(),)),
        )
    )

    assert result.fills == ()
    assert result.orders[0].cancellation_reason is CancellationReason.LIMIT_UP


def test_strategy_uses_comparable_prices_while_fills_and_sizing_use_raw_prices() -> None:
    bars = _bars()
    bars.loc[pd.Timestamp("2026-07-20"), ["open", "high", "low", "close"]] = Decimal("10")
    bars.loc[pd.Timestamp("2026-07-21"), ["open", "high", "low", "close"]] = Decimal("5")
    bars["limit_up"] = [Decimal("11"), Decimal("5.5")]
    bars["limit_down"] = [Decimal("9"), Decimal("4.5")]
    bars["adjust_factor"] = [Decimal("0.5"), Decimal("1")]
    bars["adjust_flag"] = ["3", "3"]

    class ComparableSource:
        def __init__(self) -> None:
            self.closes: list[Decimal] = []

        def targets(self, context: StrategyContext) -> DecisionTarget:
            self.closes.append(Decimal(str(context.history(SYMBOL)["close"].iloc[-1])))
            weight = Decimal("1") if context.as_of == date(2026, 7, 20) else Decimal("0")
            return DecisionTarget(weights={SYMBOL: weight}, sleeve_weights={SYMBOL: {}})

    source = ComparableSource()
    request = BacktestRequest(
        run_id="dual-price",
        sessions=(date(2026, 7, 20), date(2026, 7, 21)),
        instruments={SYMBOL: INSTRUMENT},
        bars={SYMBOL: bars},
        initial_cash=Decimal("1000"),
        initial_positions=(),
        corporate_actions=(),
        decision_source=source,
        risk_engine=RiskEngine(()),
        rule_book=MarketRuleBook((_profile(),)),
    )

    result = BacktestEngine().run(request)

    assert source.closes == [Decimal("5.0"), Decimal("5")]
    assert result.orders[0].quantity == 100
    assert result.fills[0].price == Decimal("5.0000")
    assert not any("COMPARABLE_PRICE_FACTOR_MISSING" in warning for warning in result.warnings)


def test_explicit_adjusted_execution_bars_are_rejected() -> None:
    bars = _bars()
    bars["adjust_factor"] = [Decimal("1"), Decimal("1")]
    bars["adjust_flag"] = ["1", "1"]

    with pytest.raises(ValueError, match="raw.*adjust_flag"):
        BacktestRequest(
            run_id="adjusted",
            sessions=(date(2026, 7, 20), date(2026, 7, 21)),
            instruments={SYMBOL: INSTRUMENT},
            bars={SYMBOL: bars},
            initial_cash=Decimal("1000"),
            initial_positions=(),
            corporate_actions=(),
            decision_source=_AssertingSource({}),
            risk_engine=RiskEngine(()),
            rule_book=MarketRuleBook((_profile(),)),
        )


def test_string_suspension_flag_is_rejected_without_truth_coercion() -> None:
    bars = _bars()
    bars["suspended"] = pd.Series(["False", False], index=bars.index, dtype=object)
    request = BacktestRequest(
        run_id="bad-suspended",
        sessions=(date(2026, 7, 20), date(2026, 7, 21)),
        instruments={SYMBOL: INSTRUMENT},
        bars={SYMBOL: bars},
        initial_cash=Decimal("1000"),
        initial_positions=(),
        corporate_actions=(),
        decision_source=_AssertingSource({}),
        risk_engine=RiskEngine(()),
        rule_book=MarketRuleBook((_profile(),)),
    )

    with pytest.raises(TypeError, match="suspended.*bool"):
        BacktestEngine().run(request)


def test_fill_validates_price_gross_and_buy_stamp_duty() -> None:
    result = BacktestEngine().run(_request(_AssertingSource({date(2026, 7, 20): Decimal("1")})))
    fill = result.fills[0]

    with pytest.raises(ValueError, match="four decimal"):
        replace(fill, price=Decimal("4.10001"))
    with pytest.raises(ValueError, match="gross_amount"):
        replace(fill, gross_amount=Decimal("1.00"))
    with pytest.raises(ValueError, match="buy.*stamp duty"):
        replace(
            fill,
            stamp_duty=Decimal("1.00"),
            total_fee=round_money(fill.total_fee + Decimal("1")),
        )


def test_result_rejects_ghost_mismatched_impossible_and_unprovenanced_fills() -> None:
    result = BacktestEngine().run(_request(_AssertingSource({date(2026, 7, 20): Decimal("1")})))
    fill = result.fills[0]
    ghost = replace(fill, fill_id="ghost", order_id="ghost")
    with pytest.raises(ValueError, match="ghost"):
        replace(result, fills=(ghost,))
    with pytest.raises(ValueError, match="side"):
        replace(result, fills=(replace(fill, side=OrderSide.SELL),))
    with pytest.raises(ValueError, match="date"):
        replace(result, fills=(replace(fill, trading_day=date(2026, 7, 20)),))
    with pytest.raises(ValueError, match="profile"):
        replace(result, used_profile_ids=())


@pytest.mark.parametrize(
    ("policy", "expected_quantity"),
    [
        (OddLotSellPolicy.ALLOWED, 50),
        (OddLotSellPolicy.FORBIDDEN, None),
        (OddLotSellPolicy.POSITION_REMAINDER_ONLY, None),
    ],
)
def test_engine_directional_odd_lot_sell_honors_execution_policy(
    policy: OddLotSellPolicy, expected_quantity: int | None
) -> None:
    class HalfTarget:
        def targets(self, context: StrategyContext) -> DecisionTarget:
            return DecisionTarget(weights={SYMBOL: Decimal("0.50")}, sleeve_weights={SYMBOL: {}})

    profile = replace(_profile(), odd_lot_sell_policy=policy)
    bars = _bars()
    bars.loc[:, ["open", "high", "low", "close"]] = Decimal("10")
    bars["limit_up"] = Decimal("11")
    bars["limit_down"] = Decimal("9")
    request = BacktestRequest(
        run_id=f"odd-lot-{policy.value}",
        sessions=(date(2026, 7, 20), date(2026, 7, 21)),
        instruments={SYMBOL: INSTRUMENT},
        bars={SYMBOL: bars},
        initial_cash=Decimal("500"),
        initial_positions=(InitialPosition(SYMBOL, 150, 150, Decimal("10"), Decimal("10")),),
        corporate_actions=(),
        decision_source=HalfTarget(),
        risk_engine=RiskEngine(()),
        rule_book=MarketRuleBook((profile,)),
    )

    result = BacktestEngine().run(request)

    if expected_quantity is None:
        assert result.orders == ()
        assert result.fills == ()
    else:
        assert result.orders[0].quantity == expected_quantity
        assert result.fills[0].quantity == expected_quantity


def test_result_requires_order_risk_codes_to_match_same_decision_trace() -> None:
    source = _AssertingSource({date(2026, 7, 20): Decimal("0.80")})
    result = BacktestEngine().run(
        _request(
            source,
            risk_engine=RiskEngine((SingleStockCapRule(maximum_weight=Decimal("0.40")),)),
        )
    )
    assert result.orders[0].risk_codes == ("SINGLE_STOCK_CAP",)

    unsupported = replace(result.orders[0], risk_codes=("UNSUPPORTED_RISK",))
    with pytest.raises(ValueError, match="risk code.*trace"):
        replace(result, orders=(unsupported, *result.orders[1:]))


def _two_adjustment_result() -> BacktestResult:
    source = _AssertingSource({date(2026, 7, 20): Decimal("0.80")})
    return BacktestEngine().run(
        _request(
            source,
            risk_engine=RiskEngine(
                (
                    SingleStockCapRule(maximum_weight=Decimal("0.50"), priority=10),
                    CashReserveRule(minimum_cash_weight=Decimal("0.60"), priority=20),
                )
            ),
        )
    )


def test_result_rejects_duplicate_risk_trace_keys() -> None:
    result = _two_adjustment_result()
    duplicate = result.risk_traces[0]

    with pytest.raises(ValueError, match="duplicate risk trace"):
        replace(result, risk_traces=(duplicate, *result.risk_traces))


@pytest.mark.parametrize(
    "codes",
    [
        ("CASH_RESERVE", "SINGLE_STOCK_CAP"),
        ("SINGLE_STOCK_CAP",),
        (),
    ],
)
def test_result_requires_exact_ordered_risk_code_tuple(codes: tuple[str, ...]) -> None:
    result = _two_adjustment_result()
    assert result.orders[0].risk_codes == ("SINGLE_STOCK_CAP", "CASH_RESERVE")
    changed = replace(result.orders[0], risk_codes=codes)

    with pytest.raises(ValueError, match="risk codes.*exactly match"):
        replace(result, orders=(changed, *result.orders[1:]))


def test_exact_engine_risk_codes_and_external_no_trace_empty_codes_are_valid() -> None:
    exact = _two_adjustment_result()
    assert exact.orders[0].risk_codes == tuple(
        adjustment.code for adjustment in exact.risk_traces[0].adjustments
    )

    no_risk = BacktestEngine().run(_request(_AssertingSource({date(2026, 7, 20): Decimal("1")})))
    external = replace(no_risk, risk_traces=())
    assert external.orders[0].risk_codes == ()
