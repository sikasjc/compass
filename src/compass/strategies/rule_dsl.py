from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from decimal import Decimal
from math import isfinite
import re
from typing import Literal, cast

import pandas as pd  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator

from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import TargetIntent
from compass.domain.weights import weight_to_units
from compass.strategies.base import (
    StrategyContext,
    StrategyDecision,
    StrategyDecisionStatus,
    StrategyFrequency,
    StrategyMetadata,
    StrategyParameters,
)
from compass.strategies.indicators import rsi, simple_moving_average
from compass.strategies.momentum import (
    _equal_weights,
    _normalize_strategy_id,
    _prepare_context,
)


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_FIELDS = frozenset({"open", "high", "low", "close", "volume", "amount"})
_FUNCTION_ARITY = {
    "cross_above": 2,
    "cross_below": 2,
    "highest": 2,
    "lowest": 2,
    "pct_change": 2,
    "rsi": 2,
    "sma": 2,
}
_RESERVED = _FIELDS | frozenset(_FUNCTION_ARITY)
_MAX_EXPRESSION_LENGTH = 2_048
_MAX_AST_NODES = 384
_MAX_WINDOW = 10_000


class DslVariable(StrategyParameters):
    name: str = Field(description="导出变量名称，供 DSL 和参数优化器引用。")
    value: Decimal = Field(strict=True, allow_inf_nan=False, description="当前回测使用的变量值。")
    minimum: Decimal = Field(strict=True, allow_inf_nan=False, description="优化搜索的最小值。")
    maximum: Decimal = Field(strict=True, allow_inf_nan=False, description="优化搜索的最大值。")
    step: Decimal = Field(
        strict=True, allow_inf_nan=False, gt=0, description="网格优化时的变量步长。"
    )
    optimize: bool = Field(default=True, strict=True, description="是否导出给参数优化器。")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value) or value in _RESERVED:
            raise ValueError("DSL variable name must be a non-reserved lower snake identifier")
        return value

    @model_validator(mode="after")
    def validate_range(self) -> DslVariable:
        if not self.minimum <= self.value <= self.maximum:
            raise ValueError("DSL variable value must be inside its optimization range")
        if any(
            abs(item) > Decimal("1000000")
            for item in (self.value, self.minimum, self.maximum, self.step)
        ):
            raise ValueError("DSL variable range is too large")
        return self


class RuleDslParameters(StrategyParameters):
    buy_expression: str = Field(
        min_length=1,
        max_length=_MAX_EXPRESSION_LENGTH,
        description="未持仓时触发买入的受限 DSL 布尔表达式。",
    )
    sell_expression: str = Field(
        min_length=1,
        max_length=_MAX_EXPRESSION_LENGTH,
        description="持仓时触发卖出的受限 DSL 布尔表达式。",
    )
    variables: tuple[DslVariable, ...] = Field(
        default=(),
        max_length=32,
        description="可修改、可导出给回测和优化器的变量定义。",
    )
    target_weight: Decimal = Field(
        default=Decimal("1"),
        strict=True,
        allow_inf_nan=False,
        gt=0,
        le=1,
        description="策略袖套内所有持仓标的的合计目标权重。",
    )
    execution: Literal["next_open", "next_close"] = Field(
        default="next_open",
        description="信号在收盘确认后计划采用的执行时点。",
    )

    @model_validator(mode="after")
    def validate_program(self) -> RuleDslParameters:
        names = tuple(item.name for item in self.variables)
        if len(set(names)) != len(names):
            raise ValueError("DSL variable names must be unique")
        compile_rule(self.buy_expression, names)
        compile_rule(self.sell_expression, names)
        required_history(
            (self.buy_expression, self.sell_expression),
            {item.name: item.value for item in self.variables},
        )
        weight_to_units(self.target_weight, label="target_weight")
        return self

    @property
    def variable_values(self) -> Mapping[str, Decimal]:
        return {item.name: item.value for item in self.variables}

    @property
    def optimization_variables(self) -> tuple[DslVariable, ...]:
        return tuple(item for item in self.variables if item.optimize)


class RuleDslProgram:
    def __init__(self, expression: str, variables: Sequence[str]) -> None:
        self.expression = expression.strip()
        try:
            parsed = ast.parse(self.expression, mode="eval")
        except SyntaxError:
            raise ValueError("DSL_SYNTAX_INVALID") from None
        if sum(1 for _ in ast.walk(parsed)) > _MAX_AST_NODES:
            raise ValueError("DSL_EXPRESSION_TOO_COMPLEX")
        self._variables = frozenset(variables)
        _validate_node(parsed.body, self._variables)
        self._root = parsed.body

    @property
    def referenced_fields(self) -> frozenset[str]:
        return frozenset(_referenced_fields(self._root))

    def evaluate(
        self,
        frame: pd.DataFrame,
        variables: Mapping[str, Decimal],
    ) -> bool:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return False
        missing = _referenced_fields(self._root) - set(frame.columns)
        if missing:
            raise ValueError("DSL_REQUIRED_FIELD_MISSING")
        values = {name: _finite_number(value) for name, value in variables.items()}
        if set(values) != self._variables:
            raise ValueError("DSL_VARIABLE_SET_MISMATCH")
        return _truth(_evaluate_node(self._root, frame, values))


def compile_rule(expression: str, variables: Sequence[str]) -> RuleDslProgram:
    if type(expression) is not str or not expression.strip():
        raise ValueError("DSL_EXPRESSION_EMPTY")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("DSL_EXPRESSION_TOO_LONG")
    checked_variables = tuple(variables)
    if any(
        type(item) is not str or not _IDENTIFIER.fullmatch(item) or item in _RESERVED
        for item in checked_variables
    ):
        raise ValueError("DSL_VARIABLE_NAME_INVALID")
    if len(set(checked_variables)) != len(checked_variables):
        raise ValueError("DSL_VARIABLE_NAMES_DUPLICATE")
    return RuleDslProgram(expression, checked_variables)


def required_history(expressions: Sequence[str], variables: Mapping[str, Decimal]) -> int:
    maximum = 2
    names = tuple(variables)
    for expression in expressions:
        program = compile_rule(expression, names)
        for node in ast.walk(program._root):
            if not isinstance(node, ast.Call):
                continue
            assert isinstance(node.func, ast.Name)
            if node.func.id not in {"sma", "rsi", "pct_change", "highest", "lowest"}:
                continue
            window_node = node.args[1]
            if isinstance(window_node, ast.Name):
                if window_node.id not in variables:
                    raise ValueError("DSL_WINDOW_MUST_BE_CONSTANT_OR_VARIABLE")
                raw_window: object = variables[window_node.id]
            elif isinstance(window_node, ast.Constant):
                raw_window = window_node.value
            else:
                raise ValueError("DSL_WINDOW_MUST_BE_CONSTANT_OR_VARIABLE")
            maximum = max(maximum, _window(raw_window) + 1)
    return maximum


def _validate_node(node: ast.AST, variables: frozenset[str]) -> None:
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)) or len(node.values) < 2:
            raise ValueError("DSL_BOOLEAN_OPERATOR_INVALID")
        for value in node.values:
            _validate_node(value, variables)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            raise ValueError("DSL_UNARY_OPERATOR_INVALID")
        _validate_node(node.operand, variables)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise ValueError("DSL_ARITHMETIC_OPERATOR_INVALID")
        _validate_node(node.left, variables)
        _validate_node(node.right, variables)
        return
    if isinstance(node, ast.Compare):
        if (
            len(node.ops) != 1
            or len(node.comparators) != 1
            or not isinstance(node.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq))
        ):
            raise ValueError("DSL_COMPARISON_INVALID")
        _validate_node(node.left, variables)
        _validate_node(node.comparators[0], variables)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTION_ARITY:
            raise ValueError("DSL_FUNCTION_NOT_ALLOWED")
        if node.keywords or len(node.args) != _FUNCTION_ARITY[node.func.id]:
            raise ValueError("DSL_FUNCTION_ARGUMENTS_INVALID")
        for argument in node.args:
            _validate_node(argument, variables)
        return
    if isinstance(node, ast.Name):
        if node.id not in _FIELDS and node.id not in variables:
            raise ValueError("DSL_NAME_NOT_ALLOWED")
        return
    if isinstance(node, ast.Constant):
        raw_constant = node.value
        if type(raw_constant) not in (int, float):
            raise ValueError("DSL_CONSTANT_INVALID")
        numeric_constant = float(cast(int | float, raw_constant))
        if not isfinite(numeric_constant):
            raise ValueError("DSL_CONSTANT_INVALID")
        if abs(numeric_constant) > 1_000_000:
            raise ValueError("DSL_CONSTANT_OUT_OF_RANGE")
        return
    raise ValueError("DSL_NODE_NOT_ALLOWED")


def _evaluate_node(
    node: ast.AST,
    frame: pd.DataFrame,
    variables: Mapping[str, float],
) -> object:
    if isinstance(node, ast.Name):
        return frame[node.id] if node.id in _FIELDS else variables[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BoolOp):
        values = [_truth(_evaluate_node(item, frame, variables)) for item in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand, frame, variables)
        if isinstance(node.op, ast.Not):
            return not _truth(value)
        return -value if isinstance(node.op, ast.USub) else value  # type: ignore[operator]
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, frame, variables)
        right = _evaluate_node(node.right, frame, variables)
        if isinstance(node.op, ast.Add):
            return left + right  # type: ignore[operator]
        if isinstance(node.op, ast.Sub):
            return left - right  # type: ignore[operator]
        if isinstance(node.op, ast.Mult):
            return left * right  # type: ignore[operator]
        return left / right  # type: ignore[operator]
    if isinstance(node, ast.Compare):
        left = _latest(_evaluate_node(node.left, frame, variables))
        right = _latest(_evaluate_node(node.comparators[0], frame, variables))
        if not _both_finite(left, right):
            return False
        left_number = float(cast(int | float | Decimal, left))
        right_number = float(cast(int | float | Decimal, right))
        operator = node.ops[0]
        if isinstance(operator, ast.Gt):
            return left_number > right_number
        if isinstance(operator, ast.GtE):
            return left_number >= right_number
        if isinstance(operator, ast.Lt):
            return left_number < right_number
        if isinstance(operator, ast.LtE):
            return left_number <= right_number
        if isinstance(operator, ast.Eq):
            return left_number == right_number
        return left_number != right_number
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        arguments = [_evaluate_node(item, frame, variables) for item in node.args]
        return _call(node.func.id, arguments)
    raise ValueError("DSL_NODE_NOT_ALLOWED")


def _call(name: str, arguments: Sequence[object]) -> object:
    first, second = arguments
    if name == "sma":
        return simple_moving_average(_series(first), _window(second))
    if name == "rsi":
        return rsi(_series(first), _window(second))
    if name == "pct_change":
        series = _series(first)
        window = _window(second)
        if len(series) <= window:
            return float("nan")
        return float(series.iloc[-1] / series.iloc[-1 - window] - 1)
    if name in {"highest", "lowest"}:
        values = _series(first).iloc[-_window(second) :]
        if values.empty:
            return float("nan")
        result = values.max() if name == "highest" else values.min()
        return float(result)
    left, right = _series(first), _series(second)
    if len(left) < 2 or len(right) < 2:
        return False
    current_left, previous_left = float(left.iloc[-1]), float(left.iloc[-2])
    current_right, previous_right = float(right.iloc[-1]), float(right.iloc[-2])
    if not all(map(isfinite, (current_left, previous_left, current_right, previous_right))):
        return False
    if name == "cross_above":
        return previous_left <= previous_right and current_left > current_right
    return previous_left >= previous_right and current_left < current_right


def _series(value: object) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise ValueError("DSL_SERIES_REQUIRED")
    return value


def _window(value: object) -> int:
    numeric = _finite_number(value)
    if not numeric.is_integer() or not 1 <= numeric <= _MAX_WINDOW:
        raise ValueError("DSL_WINDOW_INVALID")
    return int(numeric)


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("DSL_NUMBER_REQUIRED")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError("DSL_NUMBER_INVALID")
    return numeric


def _latest(value: object) -> object:
    if isinstance(value, pd.Series):
        return float(value.iloc[-1]) if len(value) else float("nan")
    return value


def _truth(value: object) -> bool:
    latest = _latest(value)
    if type(latest) is bool:
        return latest
    if isinstance(latest, (int, float, Decimal)):
        numeric = float(latest)
        return isfinite(numeric) and numeric != 0
    raise ValueError("DSL_BOOLEAN_REQUIRED")


def _both_finite(left: object, right: object) -> bool:
    return (
        isinstance(left, (int, float, Decimal))
        and not isinstance(left, bool)
        and isinstance(right, (int, float, Decimal))
        and not isinstance(right, bool)
        and isfinite(float(left))
        and isfinite(float(right))
    )


def _referenced_fields(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and item.id in _FIELDS}


class RuleDslStrategy:
    strategy_type = "rule_dsl"
    parameters_type = RuleDslParameters
    minimum_history = 2
    metadata = StrategyMetadata(
        strategy_type=strategy_type,
        version="1.0.0",
        display_name="自定义规则 DSL",
        description="使用受限 DSL 组合指标、买入条件、卖出条件和可优化变量。",
        supported_asset_types=frozenset({AssetType.ETF, AssetType.STOCK}),
        supported_frequencies=frozenset({StrategyFrequency.DAILY}),
        required_fields=frozenset({"close"}),
        minimum_history=minimum_history,
        default_required_history=60,
        parameters_type=parameters_type,
    )

    def __init__(
        self,
        parameters: RuleDslParameters | None = None,
        strategy_id: str = "rule_dsl",
    ) -> None:
        self.parameters = parameters or RuleDslParameters(
            buy_expression="cross_above(sma(close, fast_window), sma(close, slow_window))",
            sell_expression="cross_below(sma(close, fast_window), sma(close, slow_window))",
            variables=(
                DslVariable(
                    name="fast_window",
                    value=Decimal("20"),
                    minimum=Decimal("5"),
                    maximum=Decimal("40"),
                    step=Decimal("5"),
                ),
                DslVariable(
                    name="slow_window",
                    value=Decimal("60"),
                    minimum=Decimal("40"),
                    maximum=Decimal("200"),
                    step=Decimal("20"),
                ),
            ),
        )
        self.required_history = required_history(
            (self.parameters.buy_expression, self.parameters.sell_expression),
            self.parameters.variable_values,
        )
        names = tuple(self.parameters.variable_values)
        self._buy = compile_rule(self.parameters.buy_expression, names)
        self._sell = compile_rule(self.parameters.sell_expression, names)
        self.strategy_id = _normalize_strategy_id(strategy_id)

    def generate_targets(self, context: StrategyContext) -> StrategyDecision:
        prepared = _prepare_context(context, self.metadata.supported_asset_types)
        if isinstance(prepared, StrategyDecision):
            return prepared
        states: list[tuple[InstrumentId, bool, str]] = []
        skipped = 0
        for instrument in prepared.instruments:
            history = prepared.histories[instrument]
            if len(history) < self.required_history:
                skipped += 1
                continue
            buy = self._buy.evaluate(history, self.parameters.variable_values)
            sell = self._sell.evaluate(history, self.parameters.variable_values)
            holding = context.holding(instrument)
            held = holding is not None and holding.quantity > 0
            active = False if sell else buy or held
            reason = "DSL_SELL" if sell else "DSL_BUY" if buy else "DSL_HOLD"
            states.append((instrument, active, reason))
        if not states and skipped:
            return StrategyDecision.empty(
                StrategyDecisionStatus.SKIPPED,
                "INSUFFICIENT_HISTORY",
                details={"required_history": self.required_history},
            )
        active_count = sum(active for _, active, _ in states)
        if active_count == 0 and not any(
            context.holding(instrument) is not None for instrument, _, _ in states
        ):
            return StrategyDecision.empty(
                StrategyDecisionStatus.CASH,
                "DSL_NO_SIGNAL_CASH",
                details={"required_history": self.required_history},
            )
        weights = iter(_equal_weights(active_count, self.parameters.target_weight))
        intents = tuple(
            TargetIntent(
                strategy_id=self.strategy_id,
                instrument=instrument,
                target_weight=next(weights) if active else Decimal("0"),
                score=1.0 if active else -1.0,
                confidence=1.0,
                reason_code=reason,
                valid_until=context.as_of,
            )
            for instrument, active, reason in states
        )
        return StrategyDecision.generated(
            intents,
            details={
                "active_count": active_count,
                "required_history": self.required_history,
                "variables": {
                    name: str(value) for name, value in self.parameters.variable_values.items()
                },
            },
        )
