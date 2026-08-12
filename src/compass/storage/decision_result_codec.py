from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from math import isfinite

from compass.domain.market import InstrumentId
from compass.domain.trading import TargetIntent
from compass.portfolio.trace import AllocationAdjustment, AllocationStage
from compass.risk.base import RiskAdjustment, RiskSeverity, RiskStage
from compass.services.decision_service import (
    DecisionResult,
    DecisionSide,
    EstimatedCosts,
    RebalanceRecommendation,
    StrategyDecisionTrace,
)
from compass.strategies.base import StrategyDecisionStatus


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return value


def _list(value: object) -> list[object]:
    if type(value) is not list:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return value


def _str(value: object) -> str:
    if type(value) is not str:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return value


def _int(value: object) -> int:
    if type(value) is not int:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return value


def _bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return value


def _decimal(value: Decimal) -> str:
    if type(value) is not Decimal or not value.is_finite():
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return str(value)


def _parsed_decimal(value: object) -> Decimal:
    raw = _str(value)
    parsed = Decimal(raw)
    if not parsed.is_finite() or str(parsed) != raw:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return parsed


def _finite_float(value: object) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return value


def _day(value: object) -> date:
    raw = _str(value)
    parsed = date.fromisoformat(raw)
    if parsed.isoformat() != raw:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return parsed


def _moment(value: object) -> datetime:
    raw = _str(value)
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.isoformat() != raw:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return parsed


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(_str(item) for item in _list(value))


def json_value(value: object) -> object:
    if value is None or type(value) in {str, bool, int, float}:
        if type(value) is float and not isfinite(value):
            raise ValueError("DECISION_RESULT_INTEGRITY")
        return value
    if type(value) is Decimal:
        return _decimal(value)
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, Mapping):
        return {
            _str(key): json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [json_value(item) for item in value]
    raise TypeError("decision value is not JSON-safe")


def _encode_strategy_trace(item: StrategyDecisionTrace) -> dict[str, object]:
    return {
        "details": json_value(item.details),
        "reason_code": item.reason_code,
        "status": item.status.value,
        "strategy_id": item.strategy_id,
    }


def _decode_strategy_trace(value: object) -> StrategyDecisionTrace:
    item = _mapping(value)
    if set(item) != {"details", "reason_code", "status", "strategy_id"}:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return StrategyDecisionTrace(
        strategy_id=_str(item["strategy_id"]),
        status=StrategyDecisionStatus(_str(item["status"])),
        reason_code=_str(item["reason_code"]),
        details=_mapping(item["details"]),
    )


def _encode_intent(item: TargetIntent) -> dict[str, object]:
    return {
        "confidence": item.confidence,
        "instrument": str(item.instrument),
        "reason_code": item.reason_code,
        "score": item.score,
        "strategy_id": item.strategy_id,
        "target_weight": _decimal(item.target_weight),
        "valid_until": item.valid_until.isoformat(),
    }


def _decode_intent(value: object) -> TargetIntent:
    item = _mapping(value)
    if set(item) != {
        "confidence",
        "instrument",
        "reason_code",
        "score",
        "strategy_id",
        "target_weight",
        "valid_until",
    }:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return TargetIntent(
        strategy_id=_str(item["strategy_id"]),
        instrument=InstrumentId.parse(_str(item["instrument"])),
        target_weight=_parsed_decimal(item["target_weight"]),
        score=_finite_float(item["score"]),
        confidence=_finite_float(item["confidence"]),
        reason_code=_str(item["reason_code"]),
        valid_until=_day(item["valid_until"]),
    )


def _encode_allocation(item: AllocationAdjustment) -> dict[str, object]:
    return {
        "after_units": item.after_units,
        "before_units": item.before_units,
        "group": item.group,
        "input_keys": list(item.input_keys),
        "limit_units": item.limit_units,
        "reason_code": item.reason_code,
        "residue_recipients": list(item.residue_recipients),
        "stage": item.stage.value,
    }


def _decode_allocation(value: object) -> AllocationAdjustment:
    item = _mapping(value)
    if set(item) != {
        "after_units",
        "before_units",
        "group",
        "input_keys",
        "limit_units",
        "reason_code",
        "residue_recipients",
        "stage",
    }:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return AllocationAdjustment(
        stage=AllocationStage(_str(item["stage"])),
        group=_str(item["group"]),
        input_keys=_string_tuple(item["input_keys"]),
        before_units=_int(item["before_units"]),
        limit_units=_int(item["limit_units"]),
        after_units=_int(item["after_units"]),
        residue_recipients=_string_tuple(item["residue_recipients"]),
        reason_code=_str(item["reason_code"]),
    )


def _encode_risk(item: RiskAdjustment) -> dict[str, object]:
    return {
        "after_weight": _decimal(item.after_weight),
        "before_weight": _decimal(item.before_weight),
        "code": item.code,
        "message": item.message,
        "reference_weight": _decimal(item.reference_weight),
        "severity": item.severity.value,
        "stage": item.stage.value,
    }


def _decode_risk(value: object) -> RiskAdjustment:
    item = _mapping(value)
    if set(item) != {
        "after_weight",
        "before_weight",
        "code",
        "message",
        "reference_weight",
        "severity",
        "stage",
    }:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    return RiskAdjustment(
        code=_str(item["code"]),
        stage=RiskStage(_int(item["stage"])),
        severity=RiskSeverity(_str(item["severity"])),
        before_weight=_parsed_decimal(item["before_weight"]),
        after_weight=_parsed_decimal(item["after_weight"]),
        reference_weight=_parsed_decimal(item["reference_weight"]),
        message=_str(item["message"]),
    )


def _encode_recommendation(item: RebalanceRecommendation) -> dict[str, object]:
    return {
        "allocated_weight": _decimal(item.allocated_weight),
        "allocation_trace": [
            _encode_allocation(adjustment) for adjustment in item.allocation_trace
        ],
        "blocked": item.blocked,
        "costs": {
            "commission": _decimal(item.costs.commission),
            "stamp_duty": _decimal(item.costs.stamp_duty),
            "total": _decimal(item.costs.total),
            "transfer_fee": _decimal(item.costs.transfer_fee),
        },
        "current_quantity": item.current_quantity,
        "current_weight": _decimal(item.current_weight),
        "estimated_execution_price": (
            None
            if item.estimated_execution_price is None
            else _decimal(item.estimated_execution_price)
        ),
        "final_weight": _decimal(item.final_weight),
        "gross_amount": _decimal(item.gross_amount),
        "instrument": str(item.instrument),
        "pre_risk_weight": _decimal(item.pre_risk_weight),
        "profile_id": item.profile_id,
        "quantity_delta": item.quantity_delta,
        "raw_intents": [_encode_intent(intent) for intent in item.raw_intents],
        "reason_codes": list(item.reason_codes),
        "risk_adjustments": [_encode_risk(adjustment) for adjustment in item.risk_adjustments],
        "side": item.side.value,
        "target_quantity": item.target_quantity,
        "reference_price": _decimal(item.reference_price),
    }


def _decode_recommendation(
    value: object,
    *,
    result_fields: Mapping[str, object],
    strategy_decisions: tuple[StrategyDecisionTrace, ...],
) -> RebalanceRecommendation:
    item = _mapping(value)
    if set(item) != {
        "allocated_weight",
        "allocation_trace",
        "blocked",
        "costs",
        "current_quantity",
        "current_weight",
        "estimated_execution_price",
        "final_weight",
        "gross_amount",
        "instrument",
        "pre_risk_weight",
        "profile_id",
        "quantity_delta",
        "raw_intents",
        "reason_codes",
        "reference_price",
        "risk_adjustments",
        "side",
        "target_quantity",
    }:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    costs = _mapping(item["costs"])
    if set(costs) != {"commission", "stamp_duty", "total", "transfer_fee"}:
        raise ValueError("DECISION_RESULT_INTEGRITY")
    raw_estimated = item["estimated_execution_price"]
    return RebalanceRecommendation(
        instrument=InstrumentId.parse(_str(item["instrument"])),
        raw_intents=tuple(_decode_intent(intent) for intent in _list(item["raw_intents"])),
        strategy_decisions=strategy_decisions,
        allocated_weight=_parsed_decimal(item["allocated_weight"]),
        allocation_trace=tuple(
            _decode_allocation(adjustment) for adjustment in _list(item["allocation_trace"])
        ),
        pre_risk_weight=_parsed_decimal(item["pre_risk_weight"]),
        current_weight=_parsed_decimal(item["current_weight"]),
        final_weight=_parsed_decimal(item["final_weight"]),
        risk_adjustments=tuple(
            _decode_risk(adjustment) for adjustment in _list(item["risk_adjustments"])
        ),
        blocked=_bool(item["blocked"]),
        current_quantity=_int(item["current_quantity"]),
        target_quantity=_int(item["target_quantity"]),
        quantity_delta=_int(item["quantity_delta"]),
        side=DecisionSide(_str(item["side"])),
        reference_price=_parsed_decimal(item["reference_price"]),
        estimated_execution_price=(
            None if raw_estimated is None else _parsed_decimal(raw_estimated)
        ),
        gross_amount=_parsed_decimal(item["gross_amount"]),
        costs=EstimatedCosts(
            commission=_parsed_decimal(costs["commission"]),
            stamp_duty=_parsed_decimal(costs["stamp_duty"]),
            transfer_fee=_parsed_decimal(costs["transfer_fee"]),
            total=_parsed_decimal(costs["total"]),
        ),
        profile_id=_str(item["profile_id"]),
        market_data_source_at=_moment(result_fields["market_data_source_at"]),
        account_snapshot_row_id=_int(result_fields["account_snapshot_row_id"]),
        account_snapshot_hash=_str(result_fields["account_snapshot_hash"]),
        decision_equity=_parsed_decimal(result_fields["decision_equity"]),
        decision_at=_moment(result_fields["decision_at"]),
        decision_date=_day(result_fields["decision_date"]),
        valid_until=_day(result_fields["valid_until"]),
        reason_codes=_string_tuple(item["reason_codes"]),
    )


def encode_decision_result(result: DecisionResult) -> dict[str, object]:
    if type(result) is not DecisionResult:
        raise TypeError("result must be an exact DecisionResult")
    result.__post_init__()
    return {
        "account_id": result.account_id,
        "account_snapshot_hash": result.account_snapshot_hash,
        "account_snapshot_row_id": result.account_snapshot_row_id,
        "decision_at": result.decision_at.isoformat(),
        "decision_date": result.decision_date.isoformat(),
        "decision_equity": _decimal(result.decision_equity),
        "market_data_source_at": result.market_data_source_at.isoformat(),
        "recommendations": [_encode_recommendation(item) for item in result.recommendations],
        "remaining_cash": _decimal(result.remaining_cash),
        "strategy_decisions": [_encode_strategy_trace(item) for item in result.strategy_decisions],
        "valid_until": result.valid_until.isoformat(),
        "warnings": list(result.warnings),
    }


def decode_decision_result(value: object) -> DecisionResult:
    try:
        payload = _mapping(value)
        expected = {
            "account_id",
            "account_snapshot_hash",
            "account_snapshot_row_id",
            "decision_at",
            "decision_date",
            "decision_equity",
            "market_data_source_at",
            "recommendations",
            "remaining_cash",
            "strategy_decisions",
            "valid_until",
            "warnings",
        }
        if set(payload) != expected:
            raise ValueError
        traces = tuple(
            _decode_strategy_trace(item) for item in _list(payload["strategy_decisions"])
        )
        result = DecisionResult(
            account_id=_str(payload["account_id"]),
            account_snapshot_row_id=_int(payload["account_snapshot_row_id"]),
            account_snapshot_hash=_str(payload["account_snapshot_hash"]),
            decision_equity=_parsed_decimal(payload["decision_equity"]),
            decision_at=_moment(payload["decision_at"]),
            decision_date=_day(payload["decision_date"]),
            valid_until=_day(payload["valid_until"]),
            market_data_source_at=_moment(payload["market_data_source_at"]),
            strategy_decisions=traces,
            recommendations=tuple(
                _decode_recommendation(
                    item,
                    result_fields=payload,
                    strategy_decisions=traces,
                )
                for item in _list(payload["recommendations"])
            ),
            remaining_cash=_parsed_decimal(payload["remaining_cash"]),
            warnings=_string_tuple(payload["warnings"]),
        )
        result.__post_init__()
        return result
    except (KeyError, TypeError, ValueError, ArithmeticError):
        raise ValueError("DECISION_RESULT_INTEGRITY") from None
