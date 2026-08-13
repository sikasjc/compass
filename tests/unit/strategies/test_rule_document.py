from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from compass.strategies.rule_document import (
    RuleExecution,
    RuleSide,
    StrategyRule,
    StrategyRuleDocument,
    default_rule_document,
    document_from_parameters,
    document_from_payload,
    document_required_fields,
)


def test_rule_document_compiles_multiple_rules_into_legacy_safe_parameters() -> None:
    base = default_rule_document()
    document = base.model_copy(
        update={
            "rules": (
                *base.rules,
                StrategyRule(
                    rule_id="oversold_entry",
                    name="超跌加仓",
                    side=RuleSide.BUY,
                    priority=90,
                    expression="rsi(close, 14) < 25",
                    target_weight=Decimal("1"),
                ),
            )
        }
    )

    parameters = StrategyRuleDocument.model_validate(document).compile_parameters()

    assert " or " in parameters.buy_expression
    assert parameters.sell_expression.startswith("(")
    assert parameters.target_weight == Decimal("1")
    assert document_required_fields(document) == ("close",)


def test_rule_document_canonical_payload_round_trips_and_hashes() -> None:
    document = default_rule_document()

    restored = document_from_payload(document.canonical_payload())

    assert restored == document
    assert restored.document_hash == document.document_hash
    migrated = document_from_parameters("迁移策略", document.compile_parameters())
    original = document.compile_parameters()
    compiled = migrated.compile_parameters()
    assert compiled.variables == original.variables
    assert compiled.target_weight == original.target_weight
    assert compiled.buy_expression == f"({original.buy_expression})"
    assert compiled.sell_expression == f"({original.sell_expression})"


def test_rule_document_preserves_execution_timing_when_compiled_and_migrated() -> None:
    document = StrategyRuleDocument.model_validate(
        default_rule_document().model_copy(update={"execute": RuleExecution.NEXT_CLOSE})
    )

    parameters = document.compile_parameters()
    migrated = document_from_parameters("迁移策略", parameters)

    assert parameters.execution == "next_close"
    assert migrated.execute is RuleExecution.NEXT_CLOSE


def test_rule_document_rejects_incompatible_buy_targets_and_arbitrary_python() -> None:
    base = default_rule_document()
    with pytest.raises(ValueError, match="shared buy target"):
        StrategyRuleDocument.model_validate(
            base.model_copy(
                update={
                    "rules": (
                        *base.rules,
                        StrategyRule(
                            rule_id="small_entry",
                            name="小仓买入",
                            side=RuleSide.BUY,
                            priority=80,
                            expression="close > open",
                            target_weight=Decimal("0.5"),
                        ),
                    )
                }
            )
        )
    with pytest.raises(ValueError, match="DSL_FUNCTION_NOT_ALLOWED"):
        StrategyRuleDocument.model_validate(
            base.model_copy(
                update={
                    "rules": (
                        base.rules[0].model_copy(update={"expression": "__import__('os')"}),
                        base.rules[1],
                    )
                }
            )
        )


def test_rule_document_reports_matched_rules_by_priority() -> None:
    document = default_rule_document()
    frame = pd.DataFrame(
        {
            "open": [3.0] * 61,
            "high": [3.1] * 61,
            "low": [2.9] * 61,
            "close": [3.0] * 60 + [4.0],
            "volume": [1000.0] * 61,
            "amount": [3000.0] * 61,
        },
        index=pd.date_range("2026-05-01", periods=61, freq="D"),
    )

    matched = document.matched_rules(frame)

    assert tuple(item.rule_id for item in matched) == ("trend_entry",)
