from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from functools import reduce
from operator import or_
import re

import pandas as pd  # type: ignore[import-untyped]
from pydantic import ConfigDict, Field, field_validator, model_validator

from compass.storage.canonical_json import canonical_json, content_hash
from compass.services.safe_display import safe_identifier
from compass.strategies.base import StrategyParameters
from compass.strategies.rule_dsl import (
    DslVariable,
    RuleDslParameters,
    RuleDslProgram,
    compile_rule,
    required_history,
)


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


class RuleSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class RuleExecution(StrEnum):
    NEXT_OPEN = "next_open"
    NEXT_CLOSE = "next_close"


class StrategyRule(StrategyParameters):
    rule_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=80)
    side: RuleSide
    priority: int = Field(strict=True, ge=1, le=10_000)
    expression: str = Field(min_length=1, max_length=2_048)
    target_weight: Decimal | None = Field(
        default=None,
        strict=True,
        allow_inf_nan=False,
        gt=0,
        le=1,
    )

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("rule id must be a lower snake identifier")
        return value

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        checked = value.strip()
        if not checked or any(ord(character) < 32 for character in checked):
            raise ValueError("rule name must be safe display text")
        return checked

    @model_validator(mode="after")
    def validate_action(self) -> StrategyRule:
        if self.side is RuleSide.BUY and self.target_weight is None:
            raise ValueError("buy rule requires target weight")
        if self.side is RuleSide.SELL and self.target_weight is not None:
            raise ValueError("sell rule cannot define target weight")
        return self


class StrategyRuleDocument(StrategyParameters):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, strict=True)
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    variables: tuple[DslVariable, ...] = Field(default=(), max_length=32)
    rules: tuple[StrategyRule, ...] = Field(min_length=2, max_length=32)
    signal_at: str = Field(default="close", pattern="^close$")
    execute: RuleExecution = RuleExecution.NEXT_OPEN

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        checked = value.strip()
        if not checked or any(ord(character) < 32 for character in checked):
            raise ValueError("strategy rule document name must be safe display text")
        return checked

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        checked = value.strip()
        if any(ord(character) < 32 and character not in "\n\t" for character in checked):
            raise ValueError("strategy description must be safe display text")
        return checked

    @model_validator(mode="after")
    def validate_document(self) -> StrategyRuleDocument:
        if self.schema_version != 1:
            raise ValueError("unsupported strategy rule document version")
        variable_names = tuple(variable.name for variable in self.variables)
        if len(set(variable_names)) != len(variable_names):
            raise ValueError("strategy variables must be unique")
        rule_ids = tuple(rule.rule_id for rule in self.rules)
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("strategy rule ids must be unique")
        if not any(rule.side is RuleSide.BUY for rule in self.rules):
            raise ValueError("strategy document requires a buy rule")
        if not any(rule.side is RuleSide.SELL for rule in self.rules):
            raise ValueError("strategy document requires a sell rule")
        targets = {rule.target_weight for rule in self.rules if rule.side is RuleSide.BUY}
        if len(targets) != 1:
            raise ValueError("first version requires one shared buy target weight")
        for rule in self.rules:
            compile_rule(rule.expression, variable_names)
        required_history(self.expressions, self.variable_values)
        return self

    @property
    def expressions(self) -> tuple[str, ...]:
        return tuple(rule.expression for rule in self.rules)

    @property
    def variable_values(self) -> Mapping[str, Decimal]:
        return {variable.name: variable.value for variable in self.variables}

    @property
    def target_weight(self) -> Decimal:
        value = next(rule.target_weight for rule in self.rules if rule.side is RuleSide.BUY)
        assert value is not None
        return value

    @property
    def minimum_history(self) -> int:
        return required_history(self.expressions, self.variable_values)

    @property
    def optimization_trial_count(self) -> int:
        count = 1
        for variable in self.variables:
            if not variable.optimize:
                continue
            count *= int((variable.maximum - variable.minimum) / variable.step) + 1
        return count

    @property
    def document_hash(self) -> str:
        return content_hash(canonical_json(self.canonical_payload()))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "description": self.description,
            "execute": self.execute.value,
            "name": self.name,
            "rules": [
                {
                    "expression": rule.expression,
                    "name": rule.name,
                    "priority": rule.priority,
                    "rule_id": rule.rule_id,
                    "side": rule.side.value,
                    "target_weight": (
                        None if rule.target_weight is None else str(rule.target_weight.normalize())
                    ),
                }
                for rule in self.rules
            ],
            "schema_version": self.schema_version,
            "signal_at": self.signal_at,
            "variables": [
                {
                    "maximum": str(variable.maximum.normalize()),
                    "minimum": str(variable.minimum.normalize()),
                    "name": variable.name,
                    "optimize": variable.optimize,
                    "step": str(variable.step.normalize()),
                    "value": str(variable.value.normalize()),
                }
                for variable in self.variables
            ],
        }

    def compile_parameters(self) -> RuleDslParameters:
        return RuleDslParameters(
            buy_expression=_combined_expression(self.rules, RuleSide.BUY),
            sell_expression=_combined_expression(self.rules, RuleSide.SELL),
            variables=self.variables,
            target_weight=self.target_weight,
            execution=self.execute.value,
        )

    def compiled_rules(self) -> tuple[tuple[StrategyRule, RuleDslProgram], ...]:
        names = tuple(self.variable_values)
        return tuple(
            (rule, compile_rule(rule.expression, names))
            for rule in sorted(
                self.rules,
                key=lambda item: (-item.priority, item.side is RuleSide.BUY, item.rule_id),
            )
        )

    def matched_rules(self, frame: pd.DataFrame) -> tuple[StrategyRule, ...]:
        return tuple(
            rule
            for rule, program in self.compiled_rules()
            if program.evaluate(frame, self.variable_values)
        )


def _combined_expression(rules: Sequence[StrategyRule], side: RuleSide) -> str:
    expressions = [f"({rule.expression})" for rule in rules if rule.side is side]
    if not expressions:
        raise ValueError("strategy side requires at least one rule")
    return " or ".join(expressions)


def default_rule_document(name: str = "ETF 趋势过滤") -> StrategyRuleDocument:
    return StrategyRuleDocument(
        name=name,
        description="短期均线上穿长期均线时进入，反向下穿时退出。",
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
        rules=(
            StrategyRule(
                rule_id="trend_entry",
                name="趋势买入",
                side=RuleSide.BUY,
                priority=100,
                expression=("cross_above(sma(close, fast_window), sma(close, slow_window))"),
                target_weight=Decimal("1"),
            ),
            StrategyRule(
                rule_id="trend_exit",
                name="趋势退出",
                side=RuleSide.SELL,
                priority=200,
                expression=("cross_below(sma(close, fast_window), sma(close, slow_window))"),
            ),
        ),
    )


def document_from_parameters(
    name: str,
    parameters: RuleDslParameters,
) -> StrategyRuleDocument:
    return StrategyRuleDocument(
        name=name,
        description="由原有买入/卖出表达式迁移。",
        variables=parameters.variables,
        execute=RuleExecution(parameters.execution),
        rules=(
            StrategyRule(
                rule_id="legacy_entry",
                name="买入条件",
                side=RuleSide.BUY,
                priority=100,
                expression=parameters.buy_expression,
                target_weight=parameters.target_weight,
            ),
            StrategyRule(
                rule_id="legacy_exit",
                name="卖出条件",
                side=RuleSide.SELL,
                priority=200,
                expression=parameters.sell_expression,
            ),
        ),
    )


def document_from_payload(payload: Mapping[str, object]) -> StrategyRuleDocument:
    try:
        raw_variables = payload["variables"]
        raw_rules = payload["rules"]
        if type(raw_variables) is not list or type(raw_rules) is not list:
            raise ValueError
        variables = tuple(
            DslVariable(
                name=str(item["name"]),
                value=Decimal(str(item["value"])),
                minimum=Decimal(str(item["minimum"])),
                maximum=Decimal(str(item["maximum"])),
                step=Decimal(str(item["step"])),
                optimize=item["optimize"],
            )
            for item in raw_variables
            if isinstance(item, Mapping)
        )
        rules = tuple(
            StrategyRule(
                rule_id=str(item["rule_id"]),
                name=str(item["name"]),
                side=RuleSide(str(item["side"])),
                priority=item["priority"],
                expression=str(item["expression"]),
                target_weight=(
                    None if item["target_weight"] is None else Decimal(str(item["target_weight"]))
                ),
            )
            for item in raw_rules
            if isinstance(item, Mapping)
        )
        if len(variables) != len(raw_variables) or len(rules) != len(raw_rules):
            raise ValueError
        return StrategyRuleDocument(
            schema_version=int(str(payload["schema_version"])),
            name=str(payload["name"]),
            description=str(payload["description"]),
            variables=variables,
            rules=rules,
            signal_at=str(payload["signal_at"]),
            execute=RuleExecution(str(payload["execute"])),
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("STRATEGY_RULE_DOCUMENT_INTEGRITY") from None


@dataclass(frozen=True, slots=True)
class RuleStrategyDraft:
    draft_id: str
    watchlist_id: str
    pool_snapshot_id: str
    document: StrategyRuleDocument
    updated_at: datetime
    source_instance_id: str | None = None

    def __post_init__(self) -> None:
        safe_identifier(self.draft_id, label="strategy draft id")
        safe_identifier(self.watchlist_id, label="strategy draft watchlist id")
        safe_identifier(self.pool_snapshot_id, label="strategy draft pool snapshot id")
        if not self.pool_snapshot_id.startswith(f"{self.watchlist_id}-"):
            raise ValueError("strategy draft pool snapshot must belong to watchlist")
        if type(self.document) is not StrategyRuleDocument:
            raise TypeError("strategy draft document must be exact")
        if (
            type(self.updated_at) is not datetime
            or self.updated_at.tzinfo is None
            or self.updated_at.utcoffset() is None
        ):
            raise ValueError("strategy draft updated_at must be timezone-aware")
        if self.source_instance_id is not None:
            safe_identifier(self.source_instance_id, label="strategy draft source instance id")


def document_required_fields(document: StrategyRuleDocument) -> tuple[str, ...]:
    fields: frozenset[str] = reduce(
        or_,
        (program.referenced_fields for _, program in document.compiled_rules()),
        frozenset(),
    )
    return tuple(sorted(fields))
