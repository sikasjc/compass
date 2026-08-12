from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import inspect
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]
import pytest

from compass.backtest.market_rules import (
    MarketRuleBook,
    MarketRuleProfile,
    OddLotSellPolicy,
    PriceLimitMode,
    SettlementMode,
)
from compass.domain.market import AssetType, Instrument, InstrumentId
from compass.domain.trading import AccountSnapshot, Position, TargetIntent
from compass.portfolio.trace import AllocationPolicy, AllocationStage
from compass.risk.base import RiskAdjustment, RiskSeverity, RiskStage
from compass.risk.engine import RiskEngine
from compass.risk.rules import (
    PriceConstraintRule,
    SingleEtfCapRule,
    StrategyBudgetRule,
    TradabilityRule,
    TurnoverCapRule,
)
from compass.services.decision_service import (
    CloseDecisionRequest,
    DecisionDataError,
    DecisionResult,
    DecisionService,
    DecisionSide,
    EstimatedCosts,
    InstrumentRiskMetadata,
    StrategyDecisionTrace,
)
from compass.storage.account_repository import AccountRepository
from compass.storage.database import Database
from compass.strategies.base import (
    StrategyContext,
    StrategyDecision,
    StrategyDecisionStatus,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
DAY = date(2026, 7, 22)
DECISION_AT = datetime(2026, 7, 22, 15, 5, tzinfo=SHANGHAI)
SOURCE_AT = datetime(2026, 7, 22, 15, 0, tzinfo=SHANGHAI)


def _instrument(
    value: str = "SSE.510300",
    *,
    asset_type: AssetType = AssetType.ETF,
    lot: int = 100,
    same_day_sell: bool = False,
) -> Instrument:
    return Instrument(InstrumentId.parse(value), asset_type, lot, same_day_sell)


def _bars(close: str = "10", *, future: bool = False, adjust_factor: str | None = None):
    days = ["2026-07-21", "2026-07-22"]
    values = [Decimal("9"), Decimal(close)]
    if future:
        days.append("2026-07-23")
        values.append(Decimal("999"))
    data: dict[str, object] = {
        "open": values,
        "high": values,
        "low": values,
        "close": values,
        "volume": [100000] * len(days),
        "amount": [Decimal("900000"), Decimal("1000000"), Decimal("99900000")][
            : len(days)
        ],
        "suspended": [False] * len(days),
        "limit_up": [value * Decimal("1.20") for value in values],
        "limit_down": [value * Decimal("0.80") for value in values],
        "price_limit_rate": [Decimal("0.10")] * len(days),
        "risk_warning": [False] * len(days),
        "listing_regime_known": [True] * len(days),
    }
    if adjust_factor is not None:
        data["adjust_factor"] = [Decimal(adjust_factor)] * len(days)
    return pd.DataFrame(data, index=pd.to_datetime(days))


def _profile(
    instrument: Instrument,
    *,
    odd_lot: OddLotSellPolicy = OddLotSellPolicy.POSITION_REMAINDER_ONLY,
    confirmed: bool = True,
) -> MarketRuleProfile:
    return MarketRuleProfile(
        profile_id=f"fees-{instrument.instrument_id.code}-{odd_lot.value}",
        exchange=instrument.instrument_id.exchange,
        asset_type=instrument.asset_type,
        effective_from=date(2026, 1, 1),
        effective_to=None,
        buy_lot_size=instrument.lot_size,
        odd_lot_sell_policy=odd_lot,
        settlement_mode=(
            SettlementMode.T_PLUS_ZERO if instrument.same_day_sell else SettlementMode.T_PLUS_ONE
        ),
        same_day_sell_eligible=instrument.same_day_sell,
        price_limit_mode=PriceLimitMode.PERCENTAGE,
        price_limit_rate=Decimal("0.10"),
        risk_warning_price_limit_rate=Decimal("0.05"),
        commission_rate=Decimal("0.0003"),
        minimum_commission=Decimal("5.00"),
        sell_stamp_duty_rate=Decimal("0.0005"),
        transfer_fee_rate=Decimal("0.00001"),
        slippage_bps=Decimal("10"),
        maximum_volume_participation=Decimal("0.10"),
        fee_profile_confirmed=confirmed,
    )


@dataclass(frozen=True)
class StaticStrategy:
    strategy_id: str
    decision: StrategyDecision

    def generate_targets(self, context: StrategyContext) -> StrategyDecision:
        assert context.as_of == DAY
        return self.decision


def _intent(strategy_id: str, instrument: Instrument, weight: str) -> TargetIntent:
    return TargetIntent(
        strategy_id=strategy_id,
        instrument=instrument.instrument_id,
        target_weight=Decimal(weight),
        score=1.25,
        confidence=0.8,
        reason_code="TEST_SIGNAL",
        valid_until=DAY + timedelta(days=1),
    )


def _repository(
    tmp_path: Path,
    snapshot: AccountSnapshot,
    *,
    account_id: str = "main",
) -> AccountRepository:
    database = Database.sqlite_at(tmp_path / f"{account_id}.db")
    database.create_schema()
    repository = AccountRepository(database, account_id, lambda: DECISION_AT - timedelta(minutes=10))
    repository.save(snapshot)
    return repository


def _request(
    instrument_map: dict[InstrumentId, Instrument],
    bars: dict[InstrumentId, pd.DataFrame],
    strategies: tuple[StaticStrategy, ...],
    policy: AllocationPolicy,
    rule_book: MarketRuleBook,
    *,
    account_id: str = "main",
    risk_engine: RiskEngine | None = None,
    accepted: bool = True,
    close_complete: bool = True,
    source_at: datetime = SOURCE_AT,
    risk_metadata: dict[InstrumentId, InstrumentRiskMetadata] | None = None,
    strategy_pools: dict[str, tuple[InstrumentId, ...]] | None = None,
) -> CloseDecisionRequest:
    return CloseDecisionRequest(
        account_id=account_id,
        decision_at=DECISION_AT,
        valid_until=DAY + timedelta(days=1),
        data_accepted=accepted,
        daily_close_complete=close_complete,
        market_data_source_at=source_at,
        instruments=instrument_map,
        bars=bars,
        strategies=strategies,
        allocation_policy=policy,
        risk_engine=risk_engine or RiskEngine(()),
        rule_book=rule_book,
        risk_metadata=risk_metadata or {},
        strategy_pools=strategy_pools,
    )


def test_decision_preserves_raw_allocation_risk_cost_and_identity_chain(tmp_path: Path) -> None:
    instrument = _instrument()
    symbol = instrument.instrument_id
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("100000.00"), ()))
    strategy = StaticStrategy(
        "rotation-main",
        StrategyDecision(
            (_intent("rotation-main", instrument, "0.30"),),
            StrategyDecisionStatus.GENERATED,
            "ROTATION_SELECTED",
            {"rank": 1},
        ),
    )
    policy = AllocationPolicy(
        strategy_budgets={"rotation-main": Decimal("0.80")},
        asset_class_budgets={AssetType.ETF: Decimal("1")},
        asset_types={symbol: AssetType.ETF},
        cash_reserve=Decimal("0"),
    )
    request = _request(
        {symbol: instrument},
        {symbol: _bars()},
        (strategy,),
        policy,
        MarketRuleBook((_profile(instrument),)),
        risk_engine=RiskEngine((SingleEtfCapRule(maximum_weight=Decimal("0.20")),)),
    )

    result = DecisionService(repository).generate_close_decision(request)
    recommendation = result.recommendations[0]

    assert recommendation.raw_intents == (strategy.decision.intents[0],)
    assert recommendation.strategy_decisions[0].reason_code == "ROTATION_SELECTED"
    assert recommendation.allocated_weight == Decimal("0.240000000000")
    assert any(trace.reason_code == "STRATEGY_BUDGET_APPLIED" for trace in recommendation.allocation_trace)
    assert recommendation.pre_risk_weight == Decimal("0.240000000000")
    assert recommendation.current_weight == Decimal("0")
    assert recommendation.final_weight == Decimal("0.20")
    assert recommendation.risk_adjustments[0].code == "SINGLE_ETF_CAP"
    assert recommendation.blocked is False
    assert recommendation.target_quantity == 2000
    assert recommendation.quantity_delta == 2000
    assert recommendation.quantity_delta % 100 == 0
    assert recommendation.side is DecisionSide.BUY
    assert recommendation.reference_price == Decimal("10")
    assert recommendation.estimated_execution_price == Decimal("10.0100")
    assert recommendation.gross_amount == Decimal("20020.00")
    assert recommendation.costs.commission == Decimal("6.01")
    assert recommendation.costs.stamp_duty == Decimal("0.00")
    assert recommendation.costs.transfer_fee == Decimal("0.20")
    assert recommendation.costs.total == Decimal("6.21")
    assert recommendation.profile_id.startswith("fees-")
    assert recommendation.market_data_source_at == SOURCE_AT
    assert recommendation.account_snapshot_row_id == repository.latest().row_id
    assert recommendation.account_snapshot_hash == repository.latest().content_hash
    assert result.decision_at == DECISION_AT
    assert result.decision_date == DAY
    assert result.valid_until == DAY + timedelta(days=1)
    assert result.remaining_cash == Decimal("79973.79")


def test_formal_decision_fails_closed_when_effective_limit_state_is_unknown(
    tmp_path: Path,
) -> None:
    instrument = _instrument()
    symbol = instrument.instrument_id
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("100000.00"), ()))
    strategy = StaticStrategy(
        "unknown-rules",
        StrategyDecision.generated((_intent("unknown-rules", instrument, "0.30"),)),
    )
    policy = AllocationPolicy(
        {"unknown-rules": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF},
        Decimal("0"),
    )
    frame = _bars().drop(columns=["limit_up", "limit_down", "price_limit_rate"])

    with pytest.raises(
        DecisionDataError, match=f"PRICE_LIMIT_STATE_UNKNOWN:{symbol}"
    ):
        DecisionService(repository).generate_close_decision(
            _request(
                {symbol: instrument},
                {symbol: frame},
                (strategy,),
                policy,
                MarketRuleBook((_profile(instrument),)),
            )
        )


def test_skipped_close_decision_preserves_manual_holding(tmp_path: Path) -> None:
    """Treating an empty SKIPPED result as a cash target would emit a full sell."""

    instrument = _instrument()
    symbol = instrument.instrument_id
    snapshot = AccountSnapshot(
        DAY,
        Decimal("90000.00"),
        (Position(symbol, 1000, 1000, Decimal("9"), Decimal("10")),),
    )
    repository = _repository(tmp_path, snapshot)
    strategy = StaticStrategy(
        "weekly-rotation",
        StrategyDecision.empty(
            StrategyDecisionStatus.SKIPPED,
            "NOT_REBALANCE_SESSION",
        ),
    )
    policy = AllocationPolicy(
        strategy_budgets={"weekly-rotation": Decimal("1")},
        asset_class_budgets={AssetType.ETF: Decimal("1")},
        asset_types={symbol: AssetType.ETF},
        cash_reserve=Decimal("0"),
    )

    result = DecisionService(repository).generate_close_decision(
        _request(
            {symbol: instrument},
            {symbol: _bars()},
            (strategy,),
            policy,
            MarketRuleBook((_profile(instrument),)),
        )
    )

    recommendation = result.recommendations[0]
    assert recommendation.strategy_decisions[0].status is StrategyDecisionStatus.SKIPPED
    assert recommendation.current_weight == Decimal("0.10")
    assert recommendation.final_weight == Decimal("0.10")
    assert recommendation.quantity_delta == 0
    assert recommendation.side is DecisionSide.NONE


def test_mixed_generated_strategy_does_not_liquidate_a_separate_skipped_sleeve(
    tmp_path: Path,
) -> None:
    weekly = _instrument("SSE.510300")
    tactical = _instrument("SZSE.159915")
    instruments = {
        weekly.instrument_id: weekly,
        tactical.instrument_id: tactical,
    }
    repository = _repository(
        tmp_path,
        AccountSnapshot(
            DAY,
            Decimal("5000.00"),
            (
                Position(
                    weekly.instrument_id,
                    500,
                    500,
                    Decimal("10"),
                    Decimal("10"),
                ),
            ),
        ),
    )
    skipped = StaticStrategy(
        "weekly",
        StrategyDecision.empty(
            StrategyDecisionStatus.SKIPPED,
            "NOT_REBALANCE_DAY",
        ),
    )
    generated = StaticStrategy(
        "tactical",
        StrategyDecision.generated(
            (_intent("tactical", tactical, "1"),)
        ),
    )
    policy = AllocationPolicy(
        {"tactical": Decimal("0.20"), "weekly": Decimal("0.80")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF for symbol in instruments},
        Decimal("0"),
    )

    result = DecisionService(repository).generate_close_decision(
        _request(
            instruments,
            {
                weekly.instrument_id: _bars("10"),
                tactical.instrument_id: _bars("10"),
            },
            (generated, skipped),
            policy,
            MarketRuleBook(tuple(_profile(item) for item in instruments.values())),
            strategy_pools={
                "tactical": (tactical.instrument_id,),
                "weekly": (weekly.instrument_id,),
            },
        )
    )
    by_symbol = {item.instrument: item for item in result.recommendations}

    assert by_symbol[weekly.instrument_id].side is DecisionSide.NONE
    assert by_symbol[weekly.instrument_id].target_quantity == 500
    assert by_symbol[tactical.instrument_id].side is DecisionSide.BUY


def test_formal_strategy_context_preserves_uninterrupted_manual_holding_date(
    tmp_path: Path,
) -> None:
    """Using only the latest snapshot would lose deterministic manual holding age."""

    instrument = _instrument()
    symbol = instrument.instrument_id
    database = Database.sqlite_at(tmp_path / "holding-age.db")
    database.create_schema()
    repository = AccountRepository(database, "main", lambda: DECISION_AT - timedelta(minutes=10))
    repository.save(
        AccountSnapshot(
            DAY - timedelta(days=10),
            Decimal("90000.00"),
            (Position(symbol, 1000, 1000, Decimal("9"), Decimal("10")),),
        )
    )
    repository.save(
        AccountSnapshot(
            DAY,
            Decimal("90000.00"),
            (Position(symbol, 1000, 1000, Decimal("9"), Decimal("10")),),
        )
    )

    @dataclass(frozen=True)
    class HoldingStrategy:
        strategy_id: str = "holding-age"

        def generate_targets(self, context: StrategyContext) -> StrategyDecision:
            assert context.holding(symbol) is not None
            assert context.holding(symbol).holding_since == DAY - timedelta(days=10)
            return StrategyDecision.empty(
                StrategyDecisionStatus.SKIPPED,
                "NOT_REBALANCE_SESSION",
            )

    policy = AllocationPolicy(
        strategy_budgets={"holding-age": Decimal("1")},
        asset_class_budgets={AssetType.ETF: Decimal("1")},
        asset_types={symbol: AssetType.ETF},
        cash_reserve=Decimal("0"),
    )

    DecisionService(repository).generate_close_decision(
        _request(
            {symbol: instrument},
            {symbol: _bars()},
            (HoldingStrategy(),),  # type: ignore[arg-type]
            policy,
            MarketRuleBook((_profile(instrument),)),
        )
    )


def test_held_instrument_omitted_by_strategy_gets_explained_available_limited_sell(
    tmp_path: Path,
) -> None:
    instrument = _instrument("SSE.600000", asset_type=AssetType.STOCK)
    symbol = instrument.instrument_id
    snapshot = AccountSnapshot(
        DAY,
        Decimal("5000.00"),
        (Position(symbol, 1000, 500, Decimal("9"), Decimal("10")),),
    )
    repository = _repository(tmp_path, snapshot)
    strategy = StaticStrategy(
        "cash",
        StrategyDecision.empty(StrategyDecisionStatus.CASH, "NO_SIGNAL"),
    )
    policy = AllocationPolicy(
        strategy_budgets={"cash": Decimal("1")},
        asset_class_budgets={AssetType.STOCK: Decimal("1")},
        asset_types={symbol: AssetType.STOCK},
        cash_reserve=Decimal("0"),
    )
    result = DecisionService(repository).generate_close_decision(
        _request(
            {symbol: instrument},
            {symbol: _bars()},
            (strategy,),
            policy,
            MarketRuleBook((_profile(instrument),)),
        )
    )

    recommendation = result.recommendations[0]
    assert recommendation.raw_intents == ()
    assert recommendation.allocated_weight == Decimal("0")
    assert recommendation.final_weight == Decimal("0")
    assert recommendation.target_quantity == 500
    assert recommendation.quantity_delta == -500
    assert recommendation.side is DecisionSide.SELL
    assert "AVAILABILITY_LIMITED" in recommendation.reason_codes
    assert result.remaining_cash > snapshot.cash


@pytest.mark.parametrize(
    ("policy", "expected_delta"),
    [
        (OddLotSellPolicy.ALLOWED, -150),
        (OddLotSellPolicy.POSITION_REMAINDER_ONLY, -150),
        (OddLotSellPolicy.FORBIDDEN, -100),
    ],
)
def test_sell_quantity_honors_all_odd_lot_policies(
    tmp_path: Path, policy: OddLotSellPolicy, expected_delta: int
) -> None:
    instrument = _instrument("SSE.600000", asset_type=AssetType.STOCK)
    symbol = instrument.instrument_id
    repository = _repository(
        tmp_path,
        AccountSnapshot(
            DAY,
            Decimal("1000.00"),
            (Position(symbol, 150, 150, Decimal("10"), Decimal("10")),),
        ),
        account_id=policy.value,
    )
    strategy = StaticStrategy(
        "cash", StrategyDecision.empty(StrategyDecisionStatus.CASH, "NO_SIGNAL")
    )
    allocation = AllocationPolicy(
        {"cash": Decimal("1")},
        {AssetType.STOCK: Decimal("1")},
        {symbol: AssetType.STOCK},
        Decimal("0"),
    )
    recommendation = DecisionService(repository).generate_close_decision(
        _request(
            {symbol: instrument},
            {symbol: _bars()},
            (strategy,),
            allocation,
            MarketRuleBook((_profile(instrument, odd_lot=policy),)),
            account_id=policy.value,
        )
    ).recommendations[0]
    assert recommendation.quantity_delta == expected_delta


def test_two_buys_are_cash_scaled_proportionally_and_input_order_independent(
    tmp_path: Path,
) -> None:
    first = _instrument("SSE.510300", lot=10)
    second = _instrument("SZSE.159915", lot=10)
    instruments = {first.instrument_id: first, second.instrument_id: second}
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("1005.00"), ()))
    intents = (_intent("both", first, "0.5"), _intent("both", second, "0.5"))
    strategy = StaticStrategy("both", StrategyDecision.generated(intents))
    allocation = AllocationPolicy(
        {"both": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF for symbol in instruments},
        Decimal("0"),
    )
    profiles = MarketRuleBook(tuple(_profile(item) for item in instruments.values()))

    normal = DecisionService(repository).generate_close_decision(
        _request(
            instruments,
            {symbol: _bars() for symbol in instruments},
            (strategy,),
            allocation,
            profiles,
        )
    )
    reversed_result = DecisionService(repository).generate_close_decision(
        _request(
            dict(reversed(tuple(instruments.items()))),
            {symbol: _bars() for symbol in reversed(tuple(instruments))},
            (StaticStrategy("both", StrategyDecision.generated(tuple(reversed(intents)))),),
            allocation,
            profiles,
        )
    )

    assert tuple((item.instrument, item.quantity_delta) for item in normal.recommendations) == tuple(
        (item.instrument, item.quantity_delta) for item in reversed_result.recommendations
    )
    assert all("CASH_SCALED" in item.reason_codes for item in normal.recommendations)
    assert normal.remaining_cash >= Decimal("0.00")
    assert sum(
        item.gross_amount + item.costs.total
        for item in normal.recommendations
        if item.side is DecisionSide.BUY
    ) <= Decimal("1005.00")


def test_cash_scaling_repeats_when_minimum_commissions_keep_first_pass_over_budget(
    tmp_path: Path,
) -> None:
    first = _instrument("SSE.510300", lot=10)
    second = _instrument("SZSE.159915", lot=10)
    instruments = {first.instrument_id: first, second.instrument_id: second}
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("20.00"), ()))
    strategy = StaticStrategy(
        "both",
        StrategyDecision.generated(
            (_intent("both", first, "0.5"), _intent("both", second, "0.5"))
        ),
    )
    allocation = AllocationPolicy(
        {"both": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF for symbol in instruments},
        Decimal("0"),
    )

    result = DecisionService(repository).generate_close_decision(
        _request(
            instruments,
            {symbol: _bars("0.01") for symbol in instruments},
            (strategy,),
            allocation,
            MarketRuleBook(tuple(_profile(item) for item in instruments.values())),
        )
    )

    spend = sum(
        (item.gross_amount + item.costs.total for item in result.recommendations),
        Decimal("0"),
    )
    assert spend <= Decimal("20.00")
    assert result.remaining_cash == Decimal("20.00") - spend
    assert result.remaining_cash >= Decimal("0.00")


@pytest.mark.parametrize(
    "bad_mode", ("rejected", "incomplete-close", "future-source", "missing-close")
)
def test_decision_fails_closed_for_unaccepted_future_or_missing_daily_data(
    tmp_path: Path, bad_mode: str
) -> None:
    instrument = _instrument()
    symbol = instrument.instrument_id
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("10000.00"), ()))
    strategy = StaticStrategy("s", StrategyDecision.generated((_intent("s", instrument, "0.2"),)))
    allocation = AllocationPolicy(
        {"s": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF},
        Decimal("0"),
    )
    bars = _bars()
    if bad_mode == "missing-close":
        bars = bars.iloc[:1]
    request = _request(
        {symbol: instrument},
        {symbol: bars},
        (strategy,),
        allocation,
        MarketRuleBook((_profile(instrument),)),
        accepted=bad_mode != "rejected",
        close_complete=bad_mode != "incomplete-close",
        source_at=(DECISION_AT + timedelta(seconds=1) if bad_mode == "future-source" else SOURCE_AT),
    )
    with pytest.raises(DecisionDataError):
        DecisionService(repository).generate_close_decision(request)


def test_strategy_receives_only_comparable_non_future_data_and_request_is_defensive(
    tmp_path: Path,
) -> None:
    instrument = _instrument()
    symbol = instrument.instrument_id
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("10000.00"), ()))
    caller_bars = _bars(future=True, adjust_factor="2")

    @dataclass(frozen=True)
    class InspectingStrategy:
        strategy_id: str = "inspect"

        def generate_targets(self, context: StrategyContext) -> StrategyDecision:
            history = context.history(symbol)
            assert history.index.max().date() == DAY
            assert history.loc[pd.Timestamp(DAY), "close"] == Decimal("20")
            history.loc[:, "close"] = Decimal("1")
            return StrategyDecision.generated((_intent(self.strategy_id, instrument, "0.2"),))

    allocation = AllocationPolicy(
        {"inspect": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF},
        Decimal("0"),
    )
    request = CloseDecisionRequest(
        account_id="main",
        decision_at=DECISION_AT,
        valid_until=DAY + timedelta(days=1),
        data_accepted=True,
        daily_close_complete=True,
        market_data_source_at=SOURCE_AT,
        instruments={symbol: instrument},
        bars={symbol: caller_bars},
        strategies=(InspectingStrategy(),),
        allocation_policy=allocation,
        risk_engine=RiskEngine(()),
        rule_book=MarketRuleBook((_profile(instrument),)),
    )
    caller_bars.loc[pd.Timestamp(DAY), "close"] = Decimal("1")
    recommendation = DecisionService(repository).generate_close_decision(request).recommendations[0]
    assert recommendation.reference_price == Decimal("10")
    assert request.bars[symbol].loc[pd.Timestamp(DAY), "close"] == Decimal("10")


def test_formal_close_rejects_non_string_raw_adjust_flag(tmp_path: Path) -> None:
    instrument = _instrument()
    symbol = instrument.instrument_id
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("10000.00"), ()))
    strategy = StaticStrategy(
        "s", StrategyDecision.generated((_intent("s", instrument, "0.2"),))
    )
    allocation = AllocationPolicy(
        {"s": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF},
        Decimal("0"),
    )
    bars = _bars()
    bars["adjust_flag"] = [3, 3]

    with pytest.raises(DecisionDataError, match="RAW_PRICE_REQUIRED"):
        DecisionService(repository).generate_close_decision(
            _request(
                {symbol: instrument},
                {symbol: bars},
                (strategy,),
                allocation,
                MarketRuleBook((_profile(instrument),)),
            )
        )


def test_invalid_intent_unconfirmed_fees_missing_account_and_risk_block_fail_closed(
    tmp_path: Path,
) -> None:
    instrument = _instrument()
    symbol = instrument.instrument_id
    database = Database.sqlite_at(tmp_path / "empty.db")
    database.create_schema()
    missing = AccountRepository(database, "main", lambda: DECISION_AT)
    strategy = StaticStrategy("s", StrategyDecision.generated((_intent("s", instrument, "0.2"),)))
    allocation = AllocationPolicy(
        {"s": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF},
        Decimal("0"),
    )
    with pytest.raises(LookupError, match="ACCOUNT_SNAPSHOT_MISSING"):
        DecisionService(missing).generate_close_decision(
            _request(
                {symbol: instrument},
                {symbol: _bars()},
                (strategy,),
                allocation,
                MarketRuleBook((_profile(instrument),)),
            )
        )

    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("10000.00"), ()))
    with pytest.raises(ValueError, match="FEE_PROFILE_UNCONFIRMED"):
        DecisionService(repository).generate_close_decision(
            _request(
                {symbol: instrument},
                {symbol: _bars()},
                (strategy,),
                allocation,
                MarketRuleBook((_profile(instrument, confirmed=False),)),
            )
        )

    wrong = StaticStrategy(
        "s", StrategyDecision.generated((_intent("different", instrument, "0.2"),))
    )
    with pytest.raises(ValueError, match="strategy id"):
        DecisionService(repository).generate_close_decision(
            _request(
                {symbol: instrument},
                {symbol: _bars()},
                (wrong,),
                allocation,
                MarketRuleBook((_profile(instrument),)),
            )
        )

    blocked = DecisionService(repository).generate_close_decision(
        _request(
            {symbol: instrument},
            {symbol: _bars()},
            (strategy,),
            allocation,
            MarketRuleBook((_profile(instrument),)),
            risk_engine=RiskEngine((TradabilityRule(),)),
            risk_metadata={symbol: InstrumentRiskMetadata(tradable=False)},
        )
    ).recommendations[0]
    assert blocked.blocked is True
    assert blocked.quantity_delta == 0
    assert blocked.side is DecisionSide.NONE
    assert blocked.risk_adjustments[0].code == "NON_TRADABLE"
    assert blocked.reason_codes[-2:] == ("RISK_BLOCKED", "NO_TRADE")


def test_decision_is_advisory_only_and_does_not_mutate_account(tmp_path: Path) -> None:
    instrument = _instrument()
    symbol = instrument.instrument_id
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("10000.00"), ()))
    before = repository.history()
    strategy = StaticStrategy("s", StrategyDecision.generated((_intent("s", instrument, "0.2"),)))
    allocation = AllocationPolicy(
        {"s": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF},
        Decimal("0"),
    )
    service = DecisionService(repository)
    service.generate_close_decision(
        _request(
            {symbol: instrument},
            {symbol: _bars()},
            (strategy,),
            allocation,
            MarketRuleBook((_profile(instrument),)),
        )
    )
    assert repository.history() == before
    forbidden = {"submit", "place_order", "execute"}
    assert forbidden.isdisjoint(name for name, _ in inspect.getmembers(service, callable))
    source = Path(inspect.getfile(DecisionService)).read_text(encoding="utf-8")
    imports = tuple(
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    )
    assert "compass.backtest.broker" not in imports
    assert "Broker" not in source


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
def test_close_decision_accumulates_portfolio_risk_state_in_canonical_order(
    tmp_path: Path,
    rules: tuple[object, ...],
    expected_codes: tuple[str, ...],
) -> None:
    first = _instrument("SSE.510300")
    second = _instrument("SZSE.159915")
    instruments = {second.instrument_id: second, first.instrument_id: first}
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("10000.00"), ()))
    strategy = StaticStrategy(
        "both",
        StrategyDecision.generated(
            (_intent("both", second, "0.40"), _intent("both", first, "0.40"))
        ),
    )
    policy = AllocationPolicy(
        {"both": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF for symbol in instruments},
        Decimal("0"),
    )

    result = DecisionService(repository).generate_close_decision(
        _request(
            instruments,
            {symbol: _bars() for symbol in instruments},
            (strategy,),
            policy,
            MarketRuleBook(tuple(_profile(item) for item in instruments.values())),
            risk_engine=RiskEngine(rules),  # type: ignore[arg-type]
        )
    )
    by_symbol = {item.instrument: item for item in result.recommendations}

    assert sum((item.final_weight for item in by_symbol.values()), Decimal("0")) <= Decimal(
        "0.50"
    )
    assert tuple(item.code for item in by_symbol[first.instrument_id].risk_adjustments) == ()
    assert tuple(item.code for item in by_symbol[second.instrument_id].risk_adjustments) == (
        expected_codes
    )
    assert by_symbol[second.instrument_id].final_weight == Decimal("0.10")


def test_blocked_sell_keeps_current_exposure_in_later_strategy_budget(
    tmp_path: Path,
) -> None:
    held = _instrument("SSE.510300")
    bought = _instrument("SZSE.159915")
    instruments = {bought.instrument_id: bought, held.instrument_id: held}
    repository = _repository(
        tmp_path,
        AccountSnapshot(
            DAY,
            Decimal("4000.00"),
            (Position(held.instrument_id, 600, 600, Decimal("10"), Decimal("10")),),
        ),
    )
    strategy = StaticStrategy(
        "rotate",
        StrategyDecision.generated((_intent("rotate", bought, "0.40"),)),
    )
    policy = AllocationPolicy(
        {"rotate": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF for symbol in instruments},
        Decimal("0"),
    )

    result = DecisionService(repository).generate_close_decision(
        _request(
            instruments,
            {symbol: _bars() for symbol in instruments},
            (strategy,),
            policy,
            MarketRuleBook(tuple(_profile(item) for item in instruments.values())),
            risk_engine=RiskEngine(
                (TradabilityRule(), StrategyBudgetRule(maximum_weight=Decimal("0.70")))
            ),
            risk_metadata={held.instrument_id: InstrumentRiskMetadata(tradable=False)},
        )
    )
    by_symbol = {item.instrument: item for item in result.recommendations}

    assert by_symbol[held.instrument_id].blocked is True
    assert by_symbol[held.instrument_id].target_quantity == 600
    assert by_symbol[bought.instrument_id].final_weight == Decimal("0.10")
    assert tuple(
        item.code for item in by_symbol[bought.instrument_id].risk_adjustments
    ) == ("STRATEGY_BUDGET",)


def test_blocked_fractional_cent_holding_consumes_rounded_strategy_exposure(
    tmp_path: Path,
) -> None:
    held = _instrument("SSE.510300", lot=1)
    bought = _instrument("SZSE.159915", lot=1)
    instruments = {bought.instrument_id: bought, held.instrument_id: held}
    repository = _repository(
        tmp_path,
        AccountSnapshot(
            DAY,
            Decimal("0.99"),
            (Position(held.instrument_id, 1, 1, Decimal("0.005"), Decimal("0.005")),),
        ),
    )
    strategy = StaticStrategy(
        "fractional",
        StrategyDecision.generated((_intent("fractional", bought, "0.01"),)),
    )
    policy = AllocationPolicy(
        {"fractional": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF for symbol in instruments},
        Decimal("0"),
    )
    profiles = tuple(
        replace(
            _profile(instrument, odd_lot=OddLotSellPolicy.ALLOWED),
            commission_rate=Decimal("0"),
            minimum_commission=Decimal("0"),
            sell_stamp_duty_rate=Decimal("0"),
            transfer_fee_rate=Decimal("0"),
            slippage_bps=Decimal("0"),
        )
        for instrument in instruments.values()
    )

    result = DecisionService(repository).generate_close_decision(
        _request(
            instruments,
            {
                held.instrument_id: _bars("0.005"),
                bought.instrument_id: _bars("0.0001"),
            },
            (strategy,),
            policy,
            MarketRuleBook(profiles),
            risk_engine=RiskEngine(
                (TradabilityRule(), StrategyBudgetRule(maximum_weight=Decimal("0.01")))
            ),
            risk_metadata={held.instrument_id: InstrumentRiskMetadata(tradable=False)},
        )
    )
    by_symbol = {item.instrument: item for item in result.recommendations}
    rounded_target_value = sum(
        (
            (item.reference_price * item.target_quantity).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            for item in result.recommendations
        ),
        Decimal("0"),
    )

    assert result.decision_equity == Decimal("1.00")
    assert by_symbol[held.instrument_id].current_weight == Decimal("0.01")
    assert by_symbol[held.instrument_id].blocked is True
    assert by_symbol[bought.instrument_id].final_weight == Decimal("0")
    assert by_symbol[bought.instrument_id].target_quantity == 0
    assert rounded_target_value / result.decision_equity <= Decimal("0.01")


def test_close_decision_uses_fresh_raw_close_for_every_valuation_stage(tmp_path: Path) -> None:
    instrument = _instrument("SSE.510300", lot=1)
    symbol = instrument.instrument_id
    repository = _repository(
        tmp_path,
        AccountSnapshot(
            DAY,
            Decimal("500.00"),
            (Position(symbol, 100, 100, Decimal("5"), Decimal("5")),),
        ),
    )

    @dataclass(frozen=True)
    class ValuationStrategy:
        strategy_id: str = "fresh"

        def generate_targets(self, context: StrategyContext) -> StrategyDecision:
            assert context.account_equity == Decimal("1500.00")
            assert context.holding(symbol).mark_price == Decimal("10")
            return StrategyDecision.generated((_intent(self.strategy_id, instrument, "0.50"),))

    policy = AllocationPolicy(
        {"fresh": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF},
        Decimal("0"),
    )
    result = DecisionService(repository).generate_close_decision(
        CloseDecisionRequest(
            account_id="main",
            decision_at=DECISION_AT,
            valid_until=DAY + timedelta(days=1),
            data_accepted=True,
            daily_close_complete=True,
            market_data_source_at=SOURCE_AT,
            instruments={symbol: instrument},
            bars={symbol: _bars()},
            strategies=(ValuationStrategy(),),
            allocation_policy=policy,
            risk_engine=RiskEngine((PriceConstraintRule(),)),
            rule_book=MarketRuleBook((_profile(instrument),)),
            risk_metadata={symbol: InstrumentRiskMetadata(price_constraint_ok=False)},
        )
    )
    recommendation = result.recommendations[0]

    assert result.decision_equity == Decimal("1500.00")
    assert recommendation.decision_equity == Decimal("1500.00")
    assert recommendation.current_quantity == 100
    assert recommendation.current_weight == Decimal("0.666666666666")
    assert recommendation.blocked is True
    assert recommendation.quantity_delta == 0
    assert recommendation.target_quantity == 100
    assert recommendation.risk_adjustments[0].code == "PRICE_CONSTRAINT"


def test_recommendation_keeps_bare_symbol_cash_reserve_allocation_trace(tmp_path: Path) -> None:
    instrument = _instrument()
    symbol = instrument.instrument_id
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("10000.00"), ()))
    strategy = StaticStrategy(
        "all", StrategyDecision.generated((_intent("all", instrument, "1.0"),))
    )
    policy = AllocationPolicy(
        {"all": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF},
        Decimal("0.50"),
    )

    recommendation = DecisionService(repository).generate_close_decision(
        _request(
            {symbol: instrument},
            {symbol: _bars()},
            (strategy,),
            policy,
            MarketRuleBook((_profile(instrument),)),
        )
    ).recommendations[0]

    assert recommendation.allocated_weight == Decimal("0.500000000000")
    assert any(
        item.stage is AllocationStage.CASH_RESERVE
        and item.reason_code == "CASH_RESERVE_FLOOR"
        for item in recommendation.allocation_trace
    )


def _audited_result(tmp_path: Path) -> DecisionResult:
    instrument = _instrument()
    symbol = instrument.instrument_id
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("10000.00"), ()))
    strategy = StaticStrategy(
        "audit", StrategyDecision.generated((_intent("audit", instrument, "0.30"),))
    )
    policy = AllocationPolicy(
        {"audit": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF},
        Decimal("0"),
    )
    return DecisionService(repository).generate_close_decision(
        _request(
            {symbol: instrument},
            {symbol: _bars()},
            (strategy,),
            policy,
            MarketRuleBook((_profile(instrument),)),
            risk_engine=RiskEngine((SingleEtfCapRule(maximum_weight=Decimal("0.20")),)),
        )
    )


def test_strategy_decision_trace_recursively_freezes_nested_details() -> None:
    nested = {"items": [1, {"value": [2]}]}
    trace = StrategyDecisionTrace(
        "audit", StrategyDecisionStatus.GENERATED, "GENERATED", {"nested": nested}
    )
    nested["items"][1]["value"].append(3)

    assert trace.details["nested"]["items"][1]["value"] == (2,)


def test_recommendation_rejects_cross_field_audit_contradictions(tmp_path: Path) -> None:
    recommendation = _audited_result(tmp_path).recommendations[0]
    nonzero_costs = EstimatedCosts(
        Decimal("1.00"), Decimal("0.00"), Decimal("0.00"), Decimal("1.00")
    )
    discontinuous = RiskAdjustment(
        "SINGLE_ETF_CAP",
        RiskStage.PORTFOLIO,
        RiskSeverity.ADJUST,
        Decimal("0.25"),
        Decimal("0.20"),
        Decimal("0"),
        "Discontinuous trace.",
    )
    wrong_reference = replace(
        recommendation.risk_adjustments[0], reference_weight=Decimal("0.01")
    )
    non_generated_trace = replace(
        recommendation.strategy_decisions[0], status=StrategyDecisionStatus.CASH
    )
    invalid = (
        {"gross_amount": Decimal("0.00")},
        {"target_quantity": recommendation.target_quantity + 1},
        {
            "quantity_delta": 0,
            "target_quantity": recommendation.current_quantity,
            "side": DecisionSide.NONE,
            "estimated_execution_price": Decimal("10.0000"),
            "gross_amount": Decimal("1.00"),
            "costs": nonzero_costs,
        },
        {"blocked": True},
        {"risk_adjustments": (discontinuous,)},
        {"risk_adjustments": (wrong_reference,)},
        {"strategy_decisions": (non_generated_trace,)},
        {"reason_codes": recommendation.reason_codes + recommendation.reason_codes},
        {
            "raw_intents": (
                replace(
                    recommendation.raw_intents[0],
                    instrument=InstrumentId.parse("SZSE.159915"),
                ),
            )
        },
    )
    for changes in invalid:
        with pytest.raises((TypeError, ValueError)):
            replace(recommendation, **changes)


def test_decision_result_rejects_duplicate_bad_order_and_mismatched_identity(
    tmp_path: Path,
) -> None:
    result = _audited_result(tmp_path)
    recommendation = result.recommendations[0]
    with pytest.raises(ValueError, match="unique"):
        replace(result, recommendations=(recommendation, recommendation))

    other = replace(
        recommendation,
        instrument=InstrumentId.parse("SZSE.159915"),
        raw_intents=(),
    )
    with pytest.raises(ValueError, match="sorted"):
        replace(result, recommendations=(other, recommendation))

    mismatched = replace(
        recommendation,
        account_snapshot_hash="f" * 64,
    )
    with pytest.raises(ValueError, match="identity"):
        replace(result, recommendations=(mismatched,))

    for changes in (
        {"market_data_source_at": SOURCE_AT - timedelta(minutes=1)},
        {"decision_at": DECISION_AT + timedelta(minutes=1)},
        {"valid_until": result.valid_until + timedelta(days=1)},
    ):
        changed = replace(recommendation, **changes)
        with pytest.raises(ValueError, match="audit identity"):
            replace(result, recommendations=(changed,))

    with pytest.raises(ValueError, match="warnings"):
        replace(result, warnings=())
    with pytest.raises(ValueError, match="non-negative"):
        replace(result, remaining_cash=Decimal("-0.01"))

    orphaned = replace(recommendation, raw_intents=())
    with pytest.raises(ValueError, match="GENERATED"):
        replace(result, recommendations=(orphaned,))


def test_decision_result_accepts_generated_with_skipped_and_cash_strategies(
    tmp_path: Path,
) -> None:
    instrument = _instrument()
    symbol = instrument.instrument_id
    repository = _repository(tmp_path, AccountSnapshot(DAY, Decimal("10000.00"), ()))
    generated = StaticStrategy(
        "generated",
        StrategyDecision.generated((_intent("generated", instrument, "0.30"),)),
    )
    skipped = StaticStrategy(
        "skipped", StrategyDecision.empty(StrategyDecisionStatus.SKIPPED, "NOT_READY")
    )
    cash = StaticStrategy(
        "cash", StrategyDecision.empty(StrategyDecisionStatus.CASH, "NO_SIGNAL")
    )
    policy = AllocationPolicy(
        {
            "generated": Decimal("0.80"),
            "skipped": Decimal("0.10"),
            "cash": Decimal("0.10"),
        },
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF},
        Decimal("0"),
    )

    result = DecisionService(repository).generate_close_decision(
        _request(
            {symbol: instrument},
            {symbol: _bars()},
            (cash, generated, skipped),
            policy,
            MarketRuleBook((_profile(instrument),)),
        )
    )

    assert tuple(
        (trace.strategy_id, trace.status) for trace in result.strategy_decisions
    ) == (
        ("cash", StrategyDecisionStatus.CASH),
        ("generated", StrategyDecisionStatus.GENERATED),
        ("skipped", StrategyDecisionStatus.SKIPPED),
    )
    assert tuple(intent.strategy_id for intent in result.recommendations[0].raw_intents) == (
        "generated",
    )


def _cash_strategy_policy(
    instruments: dict[InstrumentId, Instrument],
) -> tuple[StaticStrategy, AllocationPolicy]:
    strategy = StaticStrategy(
        "cash", StrategyDecision.empty(StrategyDecisionStatus.CASH, "NO_SIGNAL")
    )
    policy = AllocationPolicy(
        {"cash": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF for symbol in instruments},
        Decimal("0"),
    )
    return strategy, policy


def test_negative_net_sell_becomes_explicit_post_risk_no_trade(tmp_path: Path) -> None:
    instrument = _instrument(lot=1)
    symbol = instrument.instrument_id
    repository = _repository(
        tmp_path,
        AccountSnapshot(
            DAY,
            Decimal("0.00"),
            (Position(symbol, 1, 1, Decimal("1"), Decimal("1")),),
        ),
    )
    strategy, policy = _cash_strategy_policy({symbol: instrument})
    profile = replace(
        _profile(instrument, odd_lot=OddLotSellPolicy.ALLOWED),
        slippage_bps=Decimal("0"),
    )

    result = DecisionService(repository).generate_close_decision(
        _request(
            {symbol: instrument},
            {symbol: _bars("1")},
            (strategy,),
            policy,
            MarketRuleBook((profile,)),
        )
    )
    recommendation = result.recommendations[0]

    assert recommendation.final_weight == Decimal("0")
    assert recommendation.side is DecisionSide.NONE
    assert recommendation.current_quantity == recommendation.target_quantity == 1
    assert recommendation.quantity_delta == 0
    assert recommendation.estimated_execution_price is None
    assert recommendation.gross_amount == Decimal("0.00")
    assert recommendation.costs == EstimatedCosts(
        Decimal("0.00"), Decimal("0.00"), Decimal("0.00"), Decimal("0.00")
    )
    assert recommendation.reason_codes[-2:] == (
        "SELL_FEES_EXCEED_AVAILABLE_CASH",
        "NO_TRADE",
    )
    assert result.remaining_cash == Decimal("0.00")


def test_negative_net_sell_can_use_existing_cash_and_preserves_sell_buy_order(
    tmp_path: Path,
) -> None:
    first = _instrument("SSE.510300", lot=1)
    second = _instrument("SSE.510500", lot=1)
    buy = _instrument("SZSE.159915", lot=1)
    instruments = {
        buy.instrument_id: buy,
        second.instrument_id: second,
        first.instrument_id: first,
    }
    repository = _repository(
        tmp_path,
        AccountSnapshot(
            DAY,
            Decimal("30.00"),
            (
                Position(second.instrument_id, 1, 1, Decimal("1"), Decimal("1")),
                Position(first.instrument_id, 1, 1, Decimal("1"), Decimal("1")),
            ),
        ),
    )
    strategy = StaticStrategy(
        "mixed", StrategyDecision.generated((_intent("mixed", buy, "0.50"),))
    )
    policy = AllocationPolicy(
        {"mixed": Decimal("1")},
        {AssetType.ETF: Decimal("1")},
        {symbol: AssetType.ETF for symbol in instruments},
        Decimal("0"),
    )
    sse = replace(
        _profile(first, odd_lot=OddLotSellPolicy.ALLOWED), slippage_bps=Decimal("0")
    )
    szse = replace(
        _profile(buy, odd_lot=OddLotSellPolicy.ALLOWED), slippage_bps=Decimal("0")
    )

    result = DecisionService(repository).generate_close_decision(
        _request(
            instruments,
            {symbol: _bars("1") for symbol in instruments},
            (strategy,),
            policy,
            MarketRuleBook((sse, szse)),
        )
    )

    assert tuple(item.side for item in result.recommendations) == (
        DecisionSide.SELL,
        DecisionSide.SELL,
        DecisionSide.BUY,
    )
    assert tuple(item.instrument for item in result.recommendations[:2]) == (
        first.instrument_id,
        second.instrument_id,
    )
    assert result.remaining_cash == Decimal("1.00")
