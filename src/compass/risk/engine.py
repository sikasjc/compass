from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
import re

from compass.risk.base import (
    RiskAdjustment,
    RiskContext,
    RiskResult,
    RiskRule,
    RiskSeverity,
    RiskStage,
    RiskTarget,
)


class RiskEngine:
    """Apply enabled rules in a stable order and retain an exact audit chain."""

    _MAX_ORDER_ADJUSTMENTS = 128

    def __init__(self, rules: Iterable[RiskRule]) -> None:
        checked: list[RiskRule] = []
        codes: set[str] = set()
        for rule in rules:
            code = getattr(rule, "code", None)
            stage = getattr(rule, "stage", None)
            priority = getattr(rule, "priority", None)
            enabled = getattr(rule, "enabled", None)
            if type(code) is not str or not re.fullmatch(r"[A-Z][A-Z0-9_]*", code):
                raise ValueError("risk rule code must be upper snake case")
            if code in codes:
                raise ValueError(f"duplicate risk rule code: {code}")
            if type(stage) is not RiskStage:
                raise TypeError("risk rule stage must be an exact RiskStage")
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise TypeError("risk rule priority must be an exact integer")
            if priority < 0:
                raise ValueError("risk rule priority must be non-negative")
            if type(enabled) is not bool:
                raise TypeError("risk rule enabled state must be an exact bool")
            evaluate = getattr(rule, "evaluate", None)
            if not callable(evaluate):
                raise TypeError("risk rules must provide evaluate")
            codes.add(code)
            checked.append(rule)
        self._rules = tuple(sorted(checked, key=lambda item: (item.stage, item.priority, item.code)))

    @property
    def rules(self) -> tuple[RiskRule, ...]:
        return self._rules

    @staticmethod
    def _evaluate_rule(
        rule: RiskRule,
        context: RiskContext,
        target: RiskTarget,
        current: Decimal,
    ) -> RiskAdjustment | None:
        adjustment = rule.evaluate(context, target, current)
        if adjustment is None:
            return None
        if type(adjustment) is not RiskAdjustment:
            raise TypeError(f"risk rule {rule.code} returned an invalid adjustment")
        if adjustment.code != rule.code or adjustment.stage is not rule.stage:
            raise ValueError(f"risk rule {rule.code} returned mismatched trace identity")
        if adjustment.before_weight != current:
            raise ValueError(f"risk rule {rule.code} returned a discontinuous trace")
        if adjustment.reference_weight != target.current_weight:
            raise ValueError(f"risk rule {rule.code} returned a mismatched holding reference")
        return adjustment

    @staticmethod
    def _result(
        target: RiskTarget,
        current: Decimal,
        adjustments: list[RiskAdjustment],
        *,
        blocked: bool,
    ) -> RiskResult:
        return RiskResult(
            requested_weight=target.requested_weight,
            final_weight=current,
            blocked=blocked,
            adjustments=tuple(adjustments),
        )

    def apply(self, context: RiskContext, target: RiskTarget) -> RiskResult:
        if type(context) is not RiskContext:
            raise TypeError("context must be an exact RiskContext")
        if type(target) is not RiskTarget:
            raise TypeError("target must be an exact RiskTarget")
        current = target.requested_weight
        adjustments: list[RiskAdjustment] = []
        non_order_rules = tuple(
            rule for rule in self._rules if rule.enabled and rule.stage is not RiskStage.ORDER
        )
        order_rules = tuple(
            rule for rule in self._rules if rule.enabled and rule.stage is RiskStage.ORDER
        )

        for rule in non_order_rules:
            adjustment = self._evaluate_rule(rule, context, target, current)
            if adjustment is None:
                continue
            adjustments.append(adjustment)
            current = adjustment.after_weight
            if adjustment.severity is RiskSeverity.BLOCK:
                return self._result(
                    target, Decimal("0"), adjustments, blocked=True
                )

        seen_order_weights = {current}
        emitted_info_codes: set[str] = set()
        order_adjustments = 0
        while True:
            restart_order_stage = False
            for rule in order_rules:
                adjustment = self._evaluate_rule(rule, context, target, current)
                if adjustment is None:
                    continue
                if adjustment.severity is RiskSeverity.INFO:
                    if adjustment.code not in emitted_info_codes:
                        adjustments.append(adjustment)
                        emitted_info_codes.add(adjustment.code)
                    continue
                adjustments.append(adjustment)
                current = adjustment.after_weight
                if adjustment.severity is RiskSeverity.BLOCK:
                    return self._result(
                        target, Decimal("0"), adjustments, blocked=True
                    )
                if adjustment.after_weight == adjustment.before_weight:
                    continue
                order_adjustments += 1
                if order_adjustments > self._MAX_ORDER_ADJUSTMENTS:
                    raise RuntimeError("order-stage risk rules did not converge")
                if current in seen_order_weights:
                    raise RuntimeError("order-stage risk rules entered a weight cycle")
                seen_order_weights.add(current)
                restart_order_stage = True
                break
            if not restart_order_stage:
                return self._result(target, current, adjustments, blocked=False)
