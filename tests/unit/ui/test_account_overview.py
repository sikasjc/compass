from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import AccountSnapshot, Position
from compass.services.local_signal_center import SignalInstrumentChoice
from compass.storage.account_repository import StoredAccountSnapshot
from compass.storage.signal_account_repository import SignalAccountProfile
from compass.ui.pages.account_overview import (
    AccountOverviewPageModel,
    _fund_chart_options,
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
    assert state.decisions == ()


def test_account_fund_chart_combines_cash_market_value_and_equity() -> None:
    snapshots = Gateway().snapshots
    options = _fund_chart_options(snapshots, ())

    assert options["legend"] == {"data": ["账户净值", "现金", "持仓市值"], "top": 8}
    series = options["series"]
    assert isinstance(series, list)
    assert [item["name"] for item in series] == ["账户净值", "现金", "持仓市值"]


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
