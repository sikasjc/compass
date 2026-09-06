from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from compass.domain.market import AssetType, InstrumentId
from compass.strategies.base import HoldingSummary, StrategyContext, StrategyDecisionStatus
from compass.strategies.rule_dsl import (
    DslAction,
    DslExecutableRule,
    DslVariable,
    RuleDslParameters,
    RuleDslState,
    RuleDslStrategy,
    compile_rule,
    dsl_action_target,
)


INSTRUMENT = InstrumentId.parse("SSE.510300")


def variable(
    name: str,
    value: str,
    minimum: str,
    maximum: str,
    step: str = "1",
) -> DslVariable:
    return DslVariable(
        name=name,
        value=Decimal(value),
        minimum=Decimal(minimum),
        maximum=Decimal(maximum),
        step=Decimal(step),
    )


def parameters() -> RuleDslParameters:
    return RuleDslParameters(
        buy_expression="cross_above(sma(close, fast_window), sma(close, slow_window))",
        sell_expression="cross_below(sma(close, fast_window), sma(close, slow_window))",
        variables=(
            variable("fast_window", "2", "1", "3"),
            variable("slow_window", "3", "2", "5"),
        ),
        target_weight=Decimal("1"),
    )


def bars() -> pd.DataFrame:
    index = pd.date_range("2026-08-03", periods=4, freq="D", name="date")
    close = [3.0, 2.0, 2.0, 3.0]
    return pd.DataFrame(
        {
            "open": close,
            "high": [item + 0.1 for item in close],
            "low": [item - 0.1 for item in close],
            "close": close,
            "volume": [1000.0] * 4,
            "amount": [3000.0, 2000.0, 2000.0, 3000.0],
        },
        index=index,
    )


def test_rule_dsl_rejects_arbitrary_python_and_unknown_names() -> None:
    with pytest.raises(ValueError, match="DSL_FUNCTION_NOT_ALLOWED"):
        compile_rule("__import__('os')", ())
    with pytest.raises(ValueError, match="DSL_NAME_NOT_ALLOWED"):
        compile_rule("close > secret", ())


def test_rule_dsl_exports_only_selected_optimization_variables() -> None:
    configured = parameters()
    fixed = variable("fixed_threshold", "1", "1", "1").model_copy(
        update={"optimize": False}
    )
    configured = configured.model_copy(
        update={"variables": (*configured.variables, fixed)}
    )

    assert tuple(item.name for item in configured.optimization_variables) == (
        "fast_window",
        "slow_window",
    )


def test_rule_dsl_strategy_generates_buy_target_without_eval() -> None:
    strategy = RuleDslStrategy(parameters(), strategy_id="custom-rule")
    context = StrategyContext(
        as_of=date(2026, 8, 6),
        bars={INSTRUMENT: bars()},
        instruments=(INSTRUMENT,),
        account_equity=Decimal("100000"),
        asset_types={INSTRUMENT: AssetType.ETF},
    )

    decision = strategy.generate_targets(context)

    assert decision.status is StrategyDecisionStatus.GENERATED
    assert len(decision) == 1
    assert decision[0].instrument == INSTRUMENT
    assert decision[0].target_weight == Decimal("1")
    assert decision[0].reason_code == "DSL_BUY"


def test_rule_dsl_exposes_position_state_without_allowing_variable_shadowing() -> None:
    program = compile_rule(
        "has_position and holding_days >= 3 and position_return < -0.05",
        (),
    )

    assert program.evaluate(
        bars(),
        {},
        RuleDslState(True, 4, -0.10, -0.12),
    )
    with pytest.raises(ValueError, match="non-reserved"):
        variable("holding_days", "2", "1", "3")


def test_rule_dsl_executes_highest_priority_action_and_records_conflicts() -> None:
    configured = RuleDslParameters(
        buy_expression="close > 0",
        sell_expression="position_return < -0.05",
        rules=(
            DslExecutableRule(
                rule_id="stop_loss",
                name="止损",
                priority=300,
                expression="position_return < -0.05",
                action=DslAction.SELL_ALL,
            ),
            DslExecutableRule(
                rule_id="trend_entry",
                name="趋势进入",
                priority=100,
                expression="close > 0",
                action=DslAction.TARGET_WEIGHT,
                value=Decimal("0.8"),
            ),
        ),
    )
    strategy = RuleDslStrategy(configured, strategy_id="position-actions")
    context = StrategyContext(
        as_of=date(2026, 8, 6),
        bars={INSTRUMENT: bars()},
        instruments=(INSTRUMENT,),
        account_equity=Decimal("100000"),
        holdings={
            INSTRUMENT: HoldingSummary(
                INSTRUMENT,
                100,
                100,
                Decimal("4"),
                Decimal("3"),
                date(2026, 8, 3),
            )
        },
        asset_types={INSTRUMENT: AssetType.ETF},
    )

    decision = strategy.generate_targets(context)

    assert decision[0].target_weight == Decimal("0")
    assert decision[0].reason_code == "DSL_SELL"
    trace = decision.details["rule_traces"][0]  # type: ignore[index]
    assert trace["selected_rule_id"] == "stop_loss"  # type: ignore[index]
    assert trace["overridden_rule_ids"] == ("trend_entry",)  # type: ignore[index]


def test_dsl_position_actions_are_bounded_and_directional() -> None:
    increase = DslExecutableRule(
        rule_id="increase",
        name="加仓",
        priority=100,
        expression="close > 0",
        action=DslAction.INCREASE_BY,
        value=Decimal("0.3"),
    )
    reduce = increase.model_copy(
        update={"rule_id": "reduce", "action": DslAction.REDUCE_TO, "value": Decimal("0.4")}
    )

    assert dsl_action_target(increase, Decimal("0.8")) == Decimal("1")
    assert dsl_action_target(reduce, Decimal("0.7")) == Decimal("0.4")
    assert dsl_action_target(reduce, Decimal("0.2")) == Decimal("0.2")
