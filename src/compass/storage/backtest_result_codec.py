from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from math import isfinite

from compass.backtest.engine import BacktestResult, RiskTrace
from compass.backtest.orders import (
    CancellationReason,
    Fill,
    LedgerPosition,
    LedgerSnapshot,
    Order,
    OrderSide,
    OrderStatus,
)
from compass.domain.market import InstrumentId
from compass.risk.base import (
    RiskAdjustment,
    RiskResult,
    RiskSeverity,
    RiskStage,
)


def _decimal(value: Decimal) -> str:
    if type(value) is not Decimal or not value.is_finite():
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    return str(value)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    return value


def _list(value: object) -> list[object]:
    if type(value) is not list:
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    return value


def _str(value: object) -> str:
    if type(value) is not str:
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    return value


def _int(value: object) -> int:
    if type(value) is not int:
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    return value


def _bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    return value


def _day(value: object) -> date:
    raw = _str(value)
    parsed = date.fromisoformat(raw)
    if parsed.isoformat() != raw:
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    return parsed


def _optional_day(value: object) -> date | None:
    return None if value is None else _day(value)


def _parsed_decimal(value: object) -> Decimal:
    raw = _str(value)
    parsed = Decimal(raw)
    if not parsed.is_finite() or str(parsed) != raw:
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    return parsed


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(_str(item) for item in _list(value))


def encode_backtest_result(result: BacktestResult) -> dict[str, object]:
    if type(result) is not BacktestResult:
        raise TypeError("result must be an exact BacktestResult")
    result.verify_integrity()
    return {
        "fills": [
            {
                "commission": _decimal(item.commission),
                "fill_id": item.fill_id,
                "gross_amount": _decimal(item.gross_amount),
                "instrument": str(item.instrument),
                "order_id": item.order_id,
                "price": _decimal(item.price),
                "profile_id": item.profile_id,
                "quantity": item.quantity,
                "side": item.side.value,
                "stamp_duty": _decimal(item.stamp_duty),
                "total_fee": _decimal(item.total_fee),
                "trading_day": item.trading_day.isoformat(),
                "transfer_fee": _decimal(item.transfer_fee),
            }
            for item in result.fills
        ],
        "ledger": [
            {
                "cash": _decimal(item.cash),
                "positions": [
                    {
                        "available_quantity": position.available_quantity,
                        "average_cost": _decimal(position.average_cost),
                        "instrument": str(position.instrument),
                        "mark_price": _decimal(position.mark_price),
                        "quantity": position.quantity,
                        "unsettled_quantity": position.unsettled_quantity,
                    }
                    for position in item.positions
                ],
                "trading_day": item.trading_day.isoformat(),
                "withdrawable_cash": _decimal(item.withdrawable_cash),
            }
            for item in result.ledger
        ],
        "orders": [
            {
                "cancellation_reason": (
                    None if item.cancellation_reason is None else item.cancellation_reason.value
                ),
                "created_on": item.created_on.isoformat(),
                "filled_quantity": item.filled_quantity,
                "instrument": str(item.instrument),
                "order_id": item.order_id,
                "quantity": item.quantity,
                "risk_codes": list(item.risk_codes),
                "scheduled_for": (
                    None if item.scheduled_for is None else item.scheduled_for.isoformat()
                ),
                "side": item.side.value,
                "sleeve_weights": {
                    key: _decimal(value) for key, value in item.sleeve_weights.items()
                },
                "status": item.status.value,
            }
            for item in result.orders
        ],
        "risk_traces": [
            {
                "adjustments": [
                    {
                        "after_weight": _decimal(adjustment.after_weight),
                        "before_weight": _decimal(adjustment.before_weight),
                        "code": adjustment.code,
                        "message": adjustment.message,
                        "reference_weight": _decimal(adjustment.reference_weight),
                        "severity": adjustment.severity.value,
                        "stage": adjustment.stage.value,
                    }
                    for adjustment in item.adjustments
                ],
                "blocked": item.result.blocked,
                "decision_date": item.decision_date.isoformat(),
                "final_weight": _decimal(item.result.final_weight),
                "instrument": str(item.instrument),
                "requested_weight": _decimal(item.result.requested_weight),
            }
            for item in result.risk_traces
        ],
        "run_id": result.run_id,
        "used_profile_ids": list(result.used_profile_ids),
        "warnings": list(result.warnings),
    }


def _decode_order(value: object) -> Order:
    item = _mapping(value)
    if set(item) != {
        "cancellation_reason",
        "created_on",
        "filled_quantity",
        "instrument",
        "order_id",
        "quantity",
        "risk_codes",
        "scheduled_for",
        "side",
        "sleeve_weights",
        "status",
    }:
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    raw_weights = _mapping(item["sleeve_weights"])
    reason = item["cancellation_reason"]
    return Order(
        order_id=_str(item["order_id"]),
        instrument=InstrumentId.parse(_str(item["instrument"])),
        side=OrderSide(_str(item["side"])),
        quantity=_int(item["quantity"]),
        created_on=_day(item["created_on"]),
        scheduled_for=_optional_day(item["scheduled_for"]),
        sleeve_weights={key: _parsed_decimal(weight) for key, weight in raw_weights.items()},
        risk_codes=_string_tuple(item["risk_codes"]),
        status=OrderStatus(_str(item["status"])),
        filled_quantity=_int(item["filled_quantity"]),
        cancellation_reason=(None if reason is None else CancellationReason(_str(reason))),
    )


def _decode_fill(value: object) -> Fill:
    item = _mapping(value)
    if set(item) != {
        "commission",
        "fill_id",
        "gross_amount",
        "instrument",
        "order_id",
        "price",
        "profile_id",
        "quantity",
        "side",
        "stamp_duty",
        "total_fee",
        "trading_day",
        "transfer_fee",
    }:
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    return Fill(
        fill_id=_str(item["fill_id"]),
        order_id=_str(item["order_id"]),
        instrument=InstrumentId.parse(_str(item["instrument"])),
        side=OrderSide(_str(item["side"])),
        quantity=_int(item["quantity"]),
        trading_day=_day(item["trading_day"]),
        price=_parsed_decimal(item["price"]),
        gross_amount=_parsed_decimal(item["gross_amount"]),
        commission=_parsed_decimal(item["commission"]),
        stamp_duty=_parsed_decimal(item["stamp_duty"]),
        transfer_fee=_parsed_decimal(item["transfer_fee"]),
        total_fee=_parsed_decimal(item["total_fee"]),
        profile_id=_str(item["profile_id"]),
    )


def _decode_ledger(value: object) -> LedgerSnapshot:
    item = _mapping(value)
    if set(item) != {"cash", "positions", "trading_day", "withdrawable_cash"}:
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    positions = []
    for raw_position in _list(item["positions"]):
        position = _mapping(raw_position)
        if set(position) != {
            "available_quantity",
            "average_cost",
            "instrument",
            "mark_price",
            "quantity",
            "unsettled_quantity",
        }:
            raise ValueError("BACKTEST_RESULT_INTEGRITY")
        positions.append(
            LedgerPosition(
                instrument=InstrumentId.parse(_str(position["instrument"])),
                quantity=_int(position["quantity"]),
                available_quantity=_int(position["available_quantity"]),
                unsettled_quantity=_int(position["unsettled_quantity"]),
                average_cost=_parsed_decimal(position["average_cost"]),
                mark_price=_parsed_decimal(position["mark_price"]),
            )
        )
    return LedgerSnapshot(
        trading_day=_day(item["trading_day"]),
        cash=_parsed_decimal(item["cash"]),
        withdrawable_cash=_parsed_decimal(item["withdrawable_cash"]),
        positions=tuple(positions),
    )


def _decode_risk_trace(value: object) -> RiskTrace:
    item = _mapping(value)
    if set(item) != {
        "adjustments",
        "blocked",
        "decision_date",
        "final_weight",
        "instrument",
        "requested_weight",
    }:
        raise ValueError("BACKTEST_RESULT_INTEGRITY")
    adjustments = []
    for raw_adjustment in _list(item["adjustments"]):
        adjustment = _mapping(raw_adjustment)
        if set(adjustment) != {
            "after_weight",
            "before_weight",
            "code",
            "message",
            "reference_weight",
            "severity",
            "stage",
        }:
            raise ValueError("BACKTEST_RESULT_INTEGRITY")
        adjustments.append(
            RiskAdjustment(
                code=_str(adjustment["code"]),
                stage=RiskStage(_int(adjustment["stage"])),
                severity=RiskSeverity(_str(adjustment["severity"])),
                before_weight=_parsed_decimal(adjustment["before_weight"]),
                after_weight=_parsed_decimal(adjustment["after_weight"]),
                reference_weight=_parsed_decimal(adjustment["reference_weight"]),
                message=_str(adjustment["message"]),
            )
        )
    result = RiskResult(
        requested_weight=_parsed_decimal(item["requested_weight"]),
        final_weight=_parsed_decimal(item["final_weight"]),
        blocked=_bool(item["blocked"]),
        adjustments=tuple(adjustments),
    )
    return RiskTrace(
        decision_date=_day(item["decision_date"]),
        instrument=InstrumentId.parse(_str(item["instrument"])),
        result=result,
    )


def decode_backtest_result(value: object) -> BacktestResult:
    try:
        payload = _mapping(value)
        expected = {
            "fills",
            "ledger",
            "orders",
            "risk_traces",
            "run_id",
            "used_profile_ids",
            "warnings",
        }
        if set(payload) != expected:
            raise ValueError
        result = BacktestResult(
            run_id=_str(payload["run_id"]),
            orders=tuple(_decode_order(item) for item in _list(payload["orders"])),
            fills=tuple(_decode_fill(item) for item in _list(payload["fills"])),
            ledger=tuple(_decode_ledger(item) for item in _list(payload["ledger"])),
            risk_traces=tuple(_decode_risk_trace(item) for item in _list(payload["risk_traces"])),
            used_profile_ids=_string_tuple(payload["used_profile_ids"]),
            warnings=_string_tuple(payload["warnings"]),
        )
        result.verify_integrity()
        return result
    except (KeyError, TypeError, ValueError, ArithmeticError):
        raise ValueError("BACKTEST_RESULT_INTEGRITY") from None


def encode_finite_float(value: float) -> str:
    if type(value) is not float or not isfinite(value):
        raise ValueError("BACKTEST_REPORT_INTEGRITY")
    return repr(value)


def decode_finite_float(value: object) -> float:
    raw = _str(value)
    parsed = float(raw)
    if not isfinite(parsed) or repr(parsed) != raw:
        raise ValueError("BACKTEST_REPORT_INTEGRITY")
    return parsed
