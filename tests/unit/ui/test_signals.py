from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import cast

import pytest

from compass.services.export_service import DecisionExportRecord
from compass.services.local_decision_gateway import SelectedDecisionStrategy
from compass.services.local_signal_center import SignalDecisionFreshness
from compass.ui.pages.signals import (
    SignalPageModel,
    _decision_notice,
    _decision_deletable,
    _strategy_reason_text,
)
from compass.storage.signal_account_repository import SignalAccountProfile
from compass.storage.signal_execution_repository import SignalExecutionRecord
from compass.storage.signal_execution_repository import SignalExecutionStatus


class SignalGatewayStub:
    def __init__(self) -> None:
        self.generated: tuple[
            tuple[SelectedDecisionStrategy, ...], Decimal, Decimal
        ] | None = None
        self.result = cast(DecisionExportRecord, object())
        self.profile = SignalAccountProfile("main", "默认账户")
        self.saved: tuple[tuple[SelectedDecisionStrategy, ...], Decimal, Decimal] | None = None
        self.deleted: list[str] = []

    def account_profiles(self):  # type: ignore[no-untyped-def]
        return (self.profile,)

    def active_account_profile(self):  # type: ignore[no-untyped-def]
        return self.profile

    def select_account(self, account_id):  # type: ignore[no-untyped-def]
        return self.profile

    def create_account(self, name, holdings_account_id=None):  # type: ignore[no-untyped-def]
        return self.profile

    def delete_account(self, account_id):  # type: ignore[no-untyped-def]
        return self.profile

    def save_strategy_configuration(
        self,
        selections: tuple[SelectedDecisionStrategy, ...],
        *,
        cash_reserve: Decimal,
        minimum_trade_amount: Decimal,
    ) -> SignalAccountProfile:
        self.saved = selections, cash_reserve, minimum_trade_amount
        return self.profile

    def instruments(self):  # type: ignore[no-untyped-def]
        return ()

    def strategies(self):  # type: ignore[no-untyped-def]
        return ()

    def latest_account(self):  # type: ignore[no-untyped-def]
        return None

    def readable_decisions(self):  # type: ignore[no-untyped-def]
        return (), 0

    def decision(self, decision_id):  # type: ignore[no-untyped-def]
        return None

    def execution(self, decision_id):  # type: ignore[no-untyped-def]
        return None

    def decision_freshness(self, record):  # type: ignore[no-untyped-def]
        return SignalDecisionFreshness(False, ())

    def delete_decision(self, decision_id):  # type: ignore[no-untyped-def]
        self.deleted.append(decision_id)
        return True

    def clear_decisions(self):  # type: ignore[no-untyped-def]
        return 2

    def clear_invalid_decisions(self):  # type: ignore[no-untyped-def]
        return 1

    def compare_decision(self, decision_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def record_execution(  # type: ignore[no-untyped-def]
        self, decision_id, status, fills, *, fees, recorded_at
    ) -> SignalExecutionRecord:
        raise NotImplementedError

    def generate(
        self,
        selections: tuple[SelectedDecisionStrategy, ...],
        *,
        cash_reserve: Decimal,
        minimum_trade_amount: Decimal,
    ) -> DecisionExportRecord:
        self.generated = selections, cash_reserve, minimum_trade_amount
        return self.result


def test_signal_model_converts_multiple_percent_allocations_exactly() -> None:
    gateway = SignalGatewayStub()
    model = SignalPageModel(gateway)

    result = model.generate(
        (("trend-v1", 30), ("rotation-v1", "40")),
        10,
        "5000",
    )

    assert result is gateway.result
    assert gateway.generated == (
        (
            SelectedDecisionStrategy("trend-v1", Decimal("0.3")),
            SelectedDecisionStrategy("rotation-v1", Decimal("0.4")),
        ),
        Decimal("0.1"),
        Decimal("5000"),
    )


def test_signal_model_requires_at_least_one_strategy() -> None:
    model = SignalPageModel(SignalGatewayStub())

    with pytest.raises(ValueError, match="SIGNAL_STRATEGY_REQUIRED"):
        model.generate((), 10, 5000)


def test_signal_model_saves_account_scoped_strategy_configuration() -> None:
    gateway = SignalGatewayStub()
    model = SignalPageModel(gateway)

    result = model.save_strategy_configuration((("trend-v1", 55),), 15, "2500")

    assert result is gateway.profile
    assert gateway.saved == (
        (SelectedDecisionStrategy("trend-v1", Decimal("0.55")),),
        Decimal("0.15"),
        Decimal("2500"),
    )


def test_decision_notice_combines_freshness_and_trading_day_status() -> None:
    notice, level = _decision_notice(
        SignalDecisionFreshness(True, ("STRATEGY_CONFIGURATION_CHANGED",)),
        date(2026, 8, 11),
        date(2026, 8, 12),
        date(2026, 8, 12),
    )

    assert notice == "该建议已失效：策略设置已变化。请重新生成后再执行。"
    assert level == "negative"
    assert "最近完整交易日" not in notice


def test_signal_reason_codes_have_plain_language_explanations() -> None:
    assert _strategy_reason_text("MA_BULL_CONFIRMED") == (
        "短期均线连续高于长期均线，趋势信号看多"
    )
    assert _strategy_reason_text("CUSTOM_REASON") == "CUSTOM_REASON"


def test_signal_model_delegates_history_cleanup() -> None:
    gateway = SignalGatewayStub()
    model = SignalPageModel(gateway)

    assert model.delete_decision("decision-1") is True
    assert gateway.deleted == ["decision-1"]
    assert model.clear_decisions() == 2
    assert model.clear_invalid_decisions() == 1


def test_only_unadopted_or_ignored_signals_are_deletable() -> None:
    assert _decision_deletable(None) is True
    assert _decision_deletable(SignalExecutionStatus.IGNORED) is True
    assert _decision_deletable(SignalExecutionStatus.EXECUTED) is False
    assert _decision_deletable(SignalExecutionStatus.PARTIAL) is False


@pytest.mark.parametrize("budget", (Decimal("0"), Decimal("1.01")))
def test_selected_strategy_rejects_out_of_range_budget(budget: Decimal) -> None:
    with pytest.raises(ValueError, match="strategy budget"):
        SelectedDecisionStrategy("trend-v1", budget)
