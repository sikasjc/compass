from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from compass.domain.market import AssetType, InstrumentId
from compass.strategies.base import StrategyContext, StrategyDecisionStatus
from compass.strategies.rule_dsl import (
    DslVariable,
    RuleDslParameters,
    RuleDslStrategy,
    compile_rule,
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
