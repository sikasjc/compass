import pytest

from compass.ui.condition_builder import build_condition
from compass.strategies.rule_dsl import compile_rule


@pytest.mark.parametrize(
    "left,operator,right",
    [
        ("close", ">", "sma(close, 20)"),
        ("close", "cross_above", "sma(close, 60)"),
        ("position_return", "<=", "-0.05"),
        ("rsi(close, 14)", "<", "30"),
    ],
)
def test_generated_conditions_compile(left: str, operator: str, right: str) -> None:
    compile_rule(build_condition(left, operator, right), ())


@pytest.mark.parametrize("right", ["NaN", "Infinity", "open('secret')", "1e100"])
def test_condition_builder_rejects_invalid_numeric_input(right: str) -> None:
    with pytest.raises(ValueError):
        build_condition("close", ">", right)
