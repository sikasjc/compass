from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import AccountSnapshot, Position
from compass.services.instrument_classification import classify_instrument
from compass.services.local_signal_center import (
    SignalAccountValuationPoint,
    SignalInstrumentChoice,
    SignalShadowExecution,
    SignalShadowPoint,
    SignalShadowSimulation,
)
from compass.storage.account_repository import StoredAccountSnapshot
from compass.storage.signal_account_repository import SignalAccountProfile
from compass.storage.signal_account_repository import (
    ShadowExecutionTiming,
    SignalShadowSimulationSetting,
)
from compass.ui.pages.account_overview import (
    AccountOverviewPageModel,
    _fund_chart_options,
    _latest_marked_snapshot,
    _shadow_chart_options,
    _shadow_progress,
    _shadow_return,
)


NOW = datetime(2026, 8, 12, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
ETF = InstrumentId.parse("SSE.510300")


def _stored(row_id: int, cash: str, price: str) -> StoredAccountSnapshot:
    snapshot = AccountSnapshot(
        date(2026, 8, 11),
        Decimal(cash),
        (Position(ETF, 100, 100, Decimal("4"), Decimal(price)),),
    )
    return StoredAccountSnapshot(row_id, "main", NOW, f"{row_id:064x}", snapshot)


class Gateway:
    def __init__(self) -> None:
        self.profile = SignalAccountProfile("main", "默认账户", holdings_account_id="main")
        self.snapshots = (_stored(1, "10000.00", "4.1"), _stored(2, "9000.00", "4.2"))

    def account_profiles(self):  # type: ignore[no-untyped-def]
        return (self.profile,)

    def active_account_profile(self):  # type: ignore[no-untyped-def]
        return self.profile

    def select_account(self, account_id):  # type: ignore[no-untyped-def]
        assert account_id == "main"
        return self.profile

    def create_account(self, name, holdings_account_id=None):  # type: ignore[no-untyped-def]
        self.created = (name, holdings_account_id)
        return self.profile

    def delete_account(self, account_id):  # type: ignore[no-untyped-def]
        self.deleted = account_id
        return self.profile

    def instruments(self):  # type: ignore[no-untyped-def]
        return (SignalInstrumentChoice(ETF, "沪深300ETF", AssetType.ETF, date(2026, 8, 11), Decimal("4.2")),)

    def latest_account(self):  # type: ignore[no-untyped-def]
        return self.snapshots[-1]

    def account_history(self):  # type: ignore[no-untyped-def]
        return self.snapshots

    def account_valuation_history(self):  # type: ignore[no-untyped-def]
        return tuple(
            SignalAccountValuationPoint(
                item.snapshot.as_of,
                item.snapshot.cash,
                item.snapshot.equity - item.snapshot.cash,
                item.snapshot.equity,
                item.row_id,
                {
                    position.instrument: position.market_value
                    for position in item.snapshot.positions
                },
            )
            for item in self.snapshots
        )

    def enable_shadow_simulation(
        self,
        execution_timing,
        *,
        commission_rate,
        minimum_commission,
        slippage_bps,
    ):  # type: ignore[no-untyped-def]
        setting = SignalShadowSimulationSetting(
            self.snapshots[-1].row_id,
            execution_timing,
            commission_rate,
            minimum_commission,
            slippage_bps,
        )
        self.profile = SignalAccountProfile(
            self.profile.account_id,
            self.profile.name,
            holdings_account_id=self.profile.holdings_account_id,
            shadow_simulation=setting,
        )
        return self.profile

    def disable_shadow_simulation(self):  # type: ignore[no-untyped-def]
        self.profile = SignalAccountProfile(
            self.profile.account_id,
            self.profile.name,
            holdings_account_id=self.profile.holdings_account_id,
        )
        return self.profile

    def shadow_simulation(self):  # type: ignore[no-untyped-def]
        return None

    def compact_account_history(self):  # type: ignore[no-untyped-def]
        self.compacted = True
        return 1

    def save_account(self, cash, positions):  # type: ignore[no-untyped-def]
        self.saved = (cash, tuple(positions))
        return self.snapshots[-1]

    def decision_history(self):  # type: ignore[no-untyped-def]
        return ()

    def execution(self, decision_id):  # type: ignore[no-untyped-def]
        return None

    def decision_freshness(self, record):  # type: ignore[no-untyped-def]
        raise AssertionError


def test_account_overview_exposes_shared_holdings_history() -> None:
    state = AccountOverviewPageModel(Gateway(), today=lambda: date(2026, 8, 12)).state()

    assert state.active_profile.account_id == "main"
    assert state.latest == state.history[-1]
    assert len(state.history) == 2
    assert len(state.valuations) == 2
    assert state.decisions == ()


def test_account_fund_chart_combines_cash_market_value_and_equity() -> None:
    gateway = Gateway()
    options = _fund_chart_options(
        gateway.account_valuation_history(),
        gateway.snapshots,
        (),
        {ETF: classify_instrument(ETF, "沪深300ETF")},
    )

    assert options["legend"] == {
        "data": ["账户净值", "现金", "持仓·宽基"],
        "top": 8,
    }
    series = options["series"]
    assert isinstance(series, list)
    assert [item["name"] for item in series] == ["账户净值", "现金", "持仓·宽基"]


def test_account_current_position_is_marked_with_latest_market_close() -> None:
    gateway = Gateway()

    snapshot, missing = _latest_marked_snapshot(
        _stored(3, "9000.00", "4.1"), gateway.instruments()
    )

    assert snapshot.as_of == date(2026, 8, 11)
    assert snapshot.positions[0].mark_price == Decimal("4.2")
    assert missing == ()


def test_account_overview_saves_position_configuration_through_shared_gateway() -> None:
    gateway = Gateway()
    model = AccountOverviewPageModel(gateway)

    saved = model.save_account(
        "12345.00",
        (("SSE.510300", 200, 100, "4.10"),),
    )

    assert saved == gateway.snapshots[-1]
    cash, positions = gateway.saved
    assert cash == "12345.00"
    assert positions[0].instrument == "SSE.510300"
    assert positions[0].quantity == 200


def test_account_overview_manages_accounts_and_snapshot_history() -> None:
    gateway = Gateway()
    model = AccountOverviewPageModel(gateway)

    assert model.create_account("  长期账户  ", "main") is gateway.profile
    assert gateway.created == ("长期账户", "main")
    assert model.delete_account("main") is gateway.profile
    assert gateway.deleted == "main"
    assert model.compact_account_history() == 1
    assert gateway.compacted is True


def test_account_overview_configures_shadow_simulation_from_current_snapshot() -> None:
    gateway = Gateway()
    model = AccountOverviewPageModel(gateway)

    profile = model.enable_shadow_simulation("next_open", "0.0003", "5", 2)

    assert profile.shadow_simulation == SignalShadowSimulationSetting(
        2,
        ShadowExecutionTiming.NEXT_OPEN,
        Decimal("0.0003"),
        Decimal("5"),
        2,
    )
    assert model.disable_shadow_simulation().shadow_simulation is None


def test_shadow_observation_presents_normalized_returns_and_progress() -> None:
    simulation = SignalShadowSimulation(
        SignalShadowSimulationSetting(2),
        date(2026, 8, 11),
        (
            SignalShadowPoint(
                date(2026, 8, 11),
                Decimal("10000.00"),
                Decimal("10000.00"),
                Decimal("10000.00"),
            ),
            SignalShadowPoint(
                date(2026, 8, 12),
                Decimal("10100.00"),
                Decimal("10200.00"),
                Decimal("10050.00"),
            ),
        ),
        (
            SignalShadowExecution(
                "decision-1",
                date(2026, 8, 11),
                None,
                "pending",
                0,
                Decimal("0.00"),
            ),
        ),
        (),
    )

    options = _shadow_chart_options(simulation)
    series = options["series"]
    assert isinstance(series, list)
    assert series[0]["data"] == [0.0, 1.0]
    assert series[1]["data"] == [0.0, 2.0]
    assert series[2]["data"] == [0.0, 0.5]
    assert _shadow_return(Decimal("10200"), Decimal("10000")) == Decimal("2.00")
    assert _shadow_progress(simulation) == (
        ("建立起点", True, False),
        ("生成信号", True, False),
        ("等待行情", False, True),
        ("模拟成交", False, False),
    )
