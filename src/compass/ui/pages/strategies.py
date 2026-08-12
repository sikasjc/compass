from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
import json
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from nicegui import ui
from pydantic import ValidationError

from compass.domain.market import AssetType, InstrumentId
from compass.services.safe_display import (
    frozen_errors,
    safe_display_text,
    safe_identifier,
    stable_code,
)
from compass.services.task_manager import Operation, TaskSnapshot, TaskStatus
from compass.strategies.base import StrategyFrequency, StrategyMetadata
from compass.strategies.registry import StrategyRegistry

if TYPE_CHECKING:
    from compass.services.strategy_optimizer import (
        OptimizationExperiment,
        OptimizationProgress,
        OptimizationRequest,
    )


T = TypeVar("T")
_MISSING = object()
_STRATEGY_GUIDES = {
    "buy_and_hold": (
        "按设定比例买入标的后长期持有，不根据短期价格变化主动择时。",
        "适合作为市场基准和主动策略的对照，也适合验证标的自身的长期表现。",
        "会完整承受市场下跌，结果对起止日期、标的选择和初始仓位较敏感。",
    ),
    "dual_ma": (
        "比较短期与长期均线；短均线持续高于长均线时持有，反之退出到现金。",
        "适合趋势持续时间较长的宽基指数或 ETF。",
        "震荡行情可能反复产生假信号，窗口越短通常交易越频繁。",
    ),
    "etf_rotation": (
        "综合多个周期的动量，并用长期趋势和波动率过滤，从候选 ETF 中选择得分靠前者。",
        "适合多个风格、行业或区域 ETF 之间存在阶段性强弱差异的市场。",
        "强弱切换过快时可能追涨杀跌，候选池和调仓频率会显著影响结果。",
    ),
    "cross_sectional_momentum": (
        "在同一时点比较多个标的的历史收益，持有排名靠前的若干标的。",
        "适合有足够多候选 ETF、趋势具有延续性的组合。",
        "市场风格突然反转时回撤可能集中，排名缓冲不足会提高换手率。",
    ),
    "mean_reversion": (
        "结合 RSI 与布林带识别短期超跌，在价格回归均值、触发止损或达到最长持有期时退出。",
        "适合震荡和短期过度反应较明显的标的。",
        "单边下跌中超跌可能持续，止损和最长持有期是关键风险控制参数。",
    ),
    "rule_dsl": (
        "用受限表达式组合价格、均线、RSI、上穿/下穿等条件，并分别定义买入与卖出逻辑。",
        "适合将明确的交易规则快速转换成可复现、可修改参数的策略实验。",
        "参数搜索容易过拟合；需要用样本外区间或滚动验证确认稳定性。",
    ),
}

_RULE_DSL_DEFAULT_PARAMETERS: Mapping[str, object] = {
    "buy_expression": "cross_above(sma(close, fast_window), sma(close, slow_window))",
    "sell_expression": "cross_below(sma(close, fast_window), sma(close, slow_window))",
    "variables": (
        {
            "name": "fast_window",
            "value": "20",
            "minimum": "5",
            "maximum": "40",
            "step": "5",
            "optimize": True,
        },
        {
            "name": "slow_window",
            "value": "60",
            "minimum": "40",
            "maximum": "200",
            "step": "20",
            "optimize": True,
        },
    ),
    "target_weight": "1",
}


class StrategyPageError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = stable_code(code, label="strategy page error code")
        super().__init__(self.code)


def _boundary_call(code: str, operation: Callable[[], T]) -> T:
    failed = False
    value: object = _MISSING
    try:
        value = operation()
    except Exception:
        failed = True
    if failed:
        raise StrategyPageError(code)
    return cast(T, value)


def _aware(value: object, *, label: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{label} must be an exact datetime")
    assert isinstance(value, datetime)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value


def _freeze(value: object, *, label: str) -> object:
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{label} must be finite")
        return value
    if type(value) is Decimal:
        assert isinstance(value, Decimal)
        if not value.is_finite():
            raise ValueError(f"{label} must be finite")
        return value
    if type(value) is str:
        safe_display_text(value, label=label)
        return value
    if isinstance(value, Enum):
        raise TypeError(f"{label} must use the registered primitive enum value")
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if type(key) is not str:
                raise TypeError(f"{label} keys must be exact strings")
            safe_display_text(key, label=f"{label} key")
            copied[key] = _freeze(item, label=f"{label}.{key}")
        return MappingProxyType(copied)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item, label=label) for item in value)
    raise TypeError(f"{label} contains an unsupported value")


def _display_value(value: object, *, label: str) -> object:
    if isinstance(value, Enum):
        return _display_value(value.value, label=label)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _display_value(item, label=f"{label}.{key}")
                for key, item in sorted(value.items())
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_display_value(item, label=label) for item in value)
    return _freeze(value, label=label)


def _inclusive_integer_minimum(schema: Mapping[str, object]) -> int | float | None:
    minimum = schema.get("minimum")
    if type(minimum) in {int, float}:
        return cast(int | float, minimum)
    exclusive = schema.get("exclusiveMinimum")
    if schema.get("type") == "integer" and type(exclusive) is int:
        return exclusive + 1
    return None


def _parameters(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("strategy parameters must be a mapping")
    frozen = _freeze(value, label="strategy parameters")
    assert isinstance(frozen, Mapping)
    return frozen


def _exact_parameter_tree_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        left_keys = tuple(left)
        right_keys = tuple(right)
        return left_keys == right_keys and all(
            _exact_parameter_tree_equal(left[key], right[key]) for key in left_keys
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)):
        assert isinstance(right, Sequence)
        return len(left) == len(right) and all(
            _exact_parameter_tree_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _json_value(value: object) -> object:
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if type(value) is Decimal:
        return str(value)
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise TypeError("strategy parameters contain a value that cannot be edited as JSON")


def _contains_enum(value: object) -> bool:
    if isinstance(value, Enum):
        return True
    if isinstance(value, Mapping):
        return any(_contains_enum(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_enum(item) for item in value)
    return False


def strategy_parameters_json(parameters: Mapping[str, object]) -> str:
    if not isinstance(parameters, Mapping):
        raise TypeError("strategy parameters must be a mapping")
    return json.dumps(
        _json_value(parameters),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def _canonical_instruments(value: Sequence[InstrumentId]) -> tuple[InstrumentId, ...]:
    instruments = tuple(value)
    if not instruments or any(type(item) is not InstrumentId for item in instruments):
        raise TypeError("strategy pool must contain exact InstrumentId values")
    if any(InstrumentId.parse(str(item)) != item for item in instruments):
        raise ValueError("strategy pool instruments must be canonical")
    if len(set(instruments)) != len(instruments):
        raise ValueError("strategy pool instruments must be unique")
    if instruments != tuple(sorted(instruments, key=str)):
        raise ValueError("strategy pool instruments must be sorted")
    return instruments


@dataclass(frozen=True, slots=True)
class StrategyParameterField:
    name: str
    description: str
    required: bool
    default: object
    minimum: int | float | None = None
    maximum: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    minimum_count: int | None = None
    maximum_count: int | None = None
    item_minimum: int | float | None = None
    item_maximum: int | float | None = None
    allowed_values: tuple[object, ...] = ()
    nullable: bool = False

    def __post_init__(self) -> None:
        safe_identifier(self.name, label="strategy parameter name")
        safe_display_text(self.description, label="strategy parameter description")
        if type(self.required) is not bool:
            raise TypeError("strategy parameter required must be an exact bool")
        object.__setattr__(self, "default", _display_value(self.default, label="strategy default"))
        for label in ("min_length", "max_length", "minimum_count", "maximum_count"):
            value = getattr(self, label)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{label} must be a non-negative exact integer or None")
        for label in ("minimum", "maximum", "item_minimum", "item_maximum"):
            value = getattr(self, label)
            if value is not None and (
                type(value) not in {int, float} or type(value) is float and not isfinite(value)
            ):
                raise ValueError(f"{label} must be a finite exact number or None")
        allowed = tuple(
            _freeze(item, label="strategy allowed value") for item in self.allowed_values
        )
        if allowed != tuple(sorted(set(allowed), key=str)):
            raise ValueError("strategy allowed values must be unique and sorted")
        if type(self.nullable) is not bool:
            raise TypeError("strategy parameter nullable must be an exact bool")
        object.__setattr__(self, "allowed_values", allowed)


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    strategy_type: str
    display_name: str
    version: str
    description: str
    supported_assets: tuple[AssetType, ...]
    supported_frequencies: tuple[StrategyFrequency, ...]
    required_fields: tuple[str, ...]
    minimum_history: int
    default_required_history: int
    parameter_fields: tuple[StrategyParameterField, ...]

    def __post_init__(self) -> None:
        safe_identifier(self.strategy_type, label="strategy type")
        safe_display_text(self.display_name, label="strategy display name")
        safe_identifier(self.version, label="strategy version")
        safe_display_text(self.description, label="strategy description")
        supported_assets = tuple(self.supported_assets)
        supported_frequencies = tuple(self.supported_frequencies)
        required_fields = tuple(self.required_fields)
        parameter_fields = tuple(self.parameter_fields)
        if any(type(item) is not AssetType for item in supported_assets):
            raise TypeError("supported assets must contain exact AssetType values")
        if supported_assets != tuple(sorted(set(supported_assets), key=lambda x: x.value)):
            raise ValueError("supported assets must be unique and sorted")
        if any(type(item) is not StrategyFrequency for item in supported_frequencies):
            raise TypeError("supported frequencies must contain exact values")
        if supported_frequencies != tuple(
            sorted(set(supported_frequencies), key=lambda x: x.value)
        ):
            raise ValueError("supported frequencies must be unique and sorted")
        if required_fields != tuple(sorted(set(required_fields))):
            raise ValueError("required fields must be unique and sorted")
        if type(self.minimum_history) is not int or self.minimum_history <= 0:
            raise ValueError("minimum history must be a positive exact integer")
        if (
            type(self.default_required_history) is not int
            or self.default_required_history < self.minimum_history
        ):
            raise ValueError("default history must not precede minimum history")
        if any(type(item) is not StrategyParameterField for item in parameter_fields):
            raise TypeError("parameter fields must contain exact values")
        names = tuple(item.name for item in parameter_fields)
        if len(set(names)) != len(names):
            raise ValueError("parameter field names must be unique")
        object.__setattr__(self, "supported_assets", supported_assets)
        object.__setattr__(self, "supported_frequencies", supported_frequencies)
        object.__setattr__(self, "required_fields", required_fields)
        object.__setattr__(self, "parameter_fields", parameter_fields)


@dataclass(frozen=True, slots=True)
class StrategyPool:
    watchlist_id: str
    snapshot_id: str
    instruments: tuple[InstrumentId, ...]
    asset_type: AssetType
    frequency: StrategyFrequency

    def __post_init__(self) -> None:
        safe_identifier(self.watchlist_id, label="watchlist id")
        safe_identifier(self.snapshot_id, label="pool snapshot id")
        if not self.snapshot_id.startswith(f"{self.watchlist_id}-"):
            raise ValueError("pool snapshot must belong to its watchlist")
        if type(self.asset_type) is not AssetType:
            raise TypeError("pool asset type must be an exact AssetType")
        if type(self.frequency) is not StrategyFrequency:
            raise TypeError("pool frequency must be an exact StrategyFrequency")
        object.__setattr__(self, "instruments", _canonical_instruments(self.instruments))


@dataclass(frozen=True, slots=True)
class StrategyPoolChoice:
    watchlist_id: str
    name: str
    instruments: tuple[InstrumentId, ...]
    asset_type: AssetType
    frequency: StrategyFrequency

    def __post_init__(self) -> None:
        safe_identifier(self.watchlist_id, label="watchlist id")
        safe_display_text(self.name, label="watchlist name")
        if type(self.asset_type) is not AssetType:
            raise TypeError("pool choice asset type must be an exact AssetType")
        if type(self.frequency) is not StrategyFrequency:
            raise TypeError("pool choice frequency must be an exact StrategyFrequency")
        object.__setattr__(self, "instruments", _canonical_instruments(self.instruments))


@dataclass(frozen=True, slots=True)
class StrategyFormModel:
    name: object
    strategy_type: object
    watchlist_id: object
    frequency: object
    parameters: object


@dataclass(frozen=True, slots=True)
class StrategyDraft:
    name: str
    strategy_type: str
    strategy_version: str
    watchlist_id: str
    pool_snapshot_id: str
    frequency: StrategyFrequency
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        safe_display_text(self.name, label="strategy name")
        safe_identifier(self.strategy_type, label="strategy type")
        safe_identifier(self.strategy_version, label="strategy version")
        safe_identifier(self.watchlist_id, label="watchlist id")
        safe_identifier(self.pool_snapshot_id, label="pool snapshot id")
        if not self.pool_snapshot_id.startswith(f"{self.watchlist_id}-"):
            raise ValueError("pool snapshot must belong to its watchlist")
        if type(self.frequency) is not StrategyFrequency:
            raise TypeError("strategy frequency must be exact")
        object.__setattr__(self, "parameters", _parameters(self.parameters))


@dataclass(frozen=True, slots=True)
class StrategyInstance:
    instance_id: str
    lineage_id: str
    version: int
    name: str
    strategy_type: str
    strategy_version: str
    watchlist_id: str
    pool_snapshot_id: str
    frequency: StrategyFrequency
    parameters: Mapping[str, object]
    enabled: bool
    created_at: datetime

    def __post_init__(self) -> None:
        for label, value in (
            ("instance id", self.instance_id),
            ("lineage id", self.lineage_id),
            ("strategy type", self.strategy_type),
            ("strategy version", self.strategy_version),
            ("watchlist id", self.watchlist_id),
            ("pool snapshot id", self.pool_snapshot_id),
        ):
            safe_identifier(value, label=label)
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("strategy instance version must be a positive exact integer")
        if not self.instance_id.endswith(f"-v{self.version}"):
            raise ValueError("instance id must identify its immutable version")
        if self.instance_id == self.lineage_id:
            raise ValueError("instance id must differ from lineage id")
        if not self.pool_snapshot_id.startswith(f"{self.watchlist_id}-"):
            raise ValueError("pool snapshot must belong to its watchlist")
        safe_display_text(self.name, label="strategy name")
        if type(self.frequency) is not StrategyFrequency:
            raise TypeError("strategy frequency must be exact")
        if type(self.enabled) is not bool:
            raise TypeError("strategy enabled must be an exact bool")
        _aware(self.created_at, label="strategy created_at")
        object.__setattr__(self, "parameters", _parameters(self.parameters))


@dataclass(frozen=True, slots=True)
class StrategyValidationResult:
    errors: Mapping[str, str]
    instance: StrategyInstance | None

    def __post_init__(self) -> None:
        errors = frozen_errors(self.errors)
        if self.instance is not None and type(self.instance) is not StrategyInstance:
            raise TypeError("instance must be an exact StrategyInstance or None")
        if self.instance is not None:
            self.instance.__post_init__()
        if (self.instance is None) == (not bool(errors)):
            raise ValueError("strategy validation must contain errors or an instance")
        object.__setattr__(self, "errors", errors)


@dataclass(frozen=True, slots=True)
class StrategyPageState:
    definitions: tuple[StrategyDefinition, ...]
    instances: tuple[StrategyInstance, ...]
    pools: tuple[StrategyPoolChoice, ...] = ()

    def __post_init__(self) -> None:
        definitions = tuple(self.definitions)
        instances = tuple(self.instances)
        pools = tuple(self.pools)
        if any(type(item) is not StrategyDefinition for item in definitions):
            raise TypeError("definitions must contain exact StrategyDefinition values")
        if any(type(item) is not StrategyInstance for item in instances):
            raise TypeError("instances must contain exact StrategyInstance values")
        if any(type(item) is not StrategyPoolChoice for item in pools):
            raise TypeError("pools must contain exact StrategyPoolChoice values")
        for instance in instances:
            instance.__post_init__()
        definition_ids = tuple(item.strategy_type for item in definitions)
        instance_ids = tuple(item.instance_id for item in instances)
        pool_ids = tuple(item.watchlist_id for item in pools)
        if definition_ids != tuple(sorted(set(definition_ids))):
            raise ValueError("strategy definitions must be unique and sorted")
        if instance_ids != tuple(sorted(set(instance_ids))):
            raise ValueError("strategy instances must be unique and sorted")
        if pool_ids != tuple(sorted(set(pool_ids))):
            raise ValueError("strategy pools must be unique and sorted")
        known = set(definition_ids)
        if any(item.strategy_type not in known for item in instances):
            raise ValueError("strategy instance type must be registered")
        lineages: dict[str, list[StrategyInstance]] = {}
        for instance in instances:
            lineages.setdefault(instance.lineage_id, []).append(instance)
        for lineage in lineages.values():
            ordered = sorted(lineage, key=lambda item: item.version)
            versions = tuple(item.version for item in ordered)
            if versions != tuple(range(1, len(ordered) + 1)):
                raise ValueError("strategy lineage versions must be gapless and unique")
            if any(item.enabled for item in ordered[:-1]):
                raise ValueError("only the latest strategy lineage version may be enabled")
        object.__setattr__(self, "definitions", definitions)
        object.__setattr__(self, "instances", instances)
        object.__setattr__(self, "pools", pools)


class StrategyGateway(Protocol):
    def list(self) -> Sequence[StrategyInstance]: ...
    def pools(self) -> Sequence[StrategyPoolChoice]: ...
    def pool(self, watchlist_id: str) -> StrategyPool: ...
    def create(self, draft: StrategyDraft) -> StrategyInstance: ...
    def copy(self, instance_id: str) -> StrategyInstance: ...
    def create_version(self, instance_id: str, draft: StrategyDraft) -> StrategyInstance: ...
    def disable(self, instance_id: str) -> None: ...
    def delete(self, instance_id: str) -> bool: ...
    def quick_backtest(self, instance_id: str) -> None: ...


class TaskGateway(Protocol):
    def submit(self, name: str, heavy: bool, operation: Operation) -> TaskSnapshot: ...
    def status(self, task_id: str) -> TaskSnapshot: ...


class StrategyOptimizerGateway(Protocol):
    def list(self) -> Sequence[OptimizationExperiment]: ...
    def default_range(self, instance_id: str) -> tuple[date, date]: ...
    def new_experiment_id(self) -> str: ...
    def run(self, experiment_id: str, request: OptimizationRequest) -> None: ...
    def progress(self, experiment_id: str) -> OptimizationProgress | None: ...
    def publish(self, experiment_id: str, rank: int = 1) -> StrategyInstance: ...


def _definition(metadata: StrategyMetadata) -> StrategyDefinition:
    schema = metadata.parameters_type.model_json_schema()
    raw_properties = schema.get("properties", {})
    if not isinstance(raw_properties, Mapping):
        raise TypeError("strategy parameter schema properties must be a mapping")
    fields: list[StrategyParameterField] = []
    for name, field in metadata.parameters_type.model_fields.items():
        raw = raw_properties.get(name, {})
        if not isinstance(raw, Mapping):
            raise TypeError("strategy parameter schema field must be a mapping")
        item_schema = raw.get("items", {})
        if not isinstance(item_schema, Mapping):
            raise TypeError("strategy parameter item schema must be a mapping")
        referenced = raw
        reference = raw.get("$ref")
        if reference is not None:
            if type(reference) is not str or not reference.startswith("#/$defs/"):
                raise ValueError("strategy parameter schema reference must be local")
            definitions = schema.get("$defs", {})
            if not isinstance(definitions, Mapping):
                raise TypeError("strategy parameter schema definitions must be a mapping")
            referenced_value = definitions.get(reference.removeprefix("#/$defs/"), {})
            if not isinstance(referenced_value, Mapping):
                raise TypeError("strategy parameter referenced schema must be a mapping")
            referenced = referenced_value
        nullable = False
        alternatives = raw.get("anyOf", ())
        if alternatives:
            if not isinstance(alternatives, Sequence):
                raise TypeError("strategy parameter alternatives must be a sequence")
            for alternative in alternatives:
                if not isinstance(alternative, Mapping):
                    raise TypeError("strategy parameter alternative must be a mapping")
                nullable = nullable or alternative.get("type") == "null"
        enum_values = referenced.get("enum", ())
        if enum_values and (not isinstance(enum_values, Sequence) or isinstance(enum_values, str)):
            raise TypeError("strategy parameter enum must be a sequence")
        allowed_values = tuple(sorted(enum_values, key=str)) if enum_values else ()
        default = None if field.is_required() else field.default
        fields.append(
            StrategyParameterField(
                name=name,
                description=cast(str, field.description),
                required=field.is_required(),
                default=default,
                minimum=cast(int | float | None, raw.get("minimum")),
                maximum=cast(int | float | None, raw.get("maximum")),
                min_length=cast(int | None, raw.get("minLength")),
                max_length=cast(int | None, raw.get("maxLength")),
                minimum_count=cast(int | None, raw.get("minItems")),
                maximum_count=cast(int | None, raw.get("maxItems")),
                item_minimum=_inclusive_integer_minimum(item_schema),
                item_maximum=cast(int | float | None, item_schema.get("maximum")),
                allowed_values=allowed_values,
                nullable=nullable,
            )
        )
    return StrategyDefinition(
        strategy_type=metadata.strategy_type,
        display_name=metadata.display_name,
        version=metadata.version,
        description=metadata.description,
        supported_assets=tuple(sorted(metadata.supported_asset_types, key=lambda x: x.value)),
        supported_frequencies=tuple(sorted(metadata.supported_frequencies, key=lambda x: x.value)),
        required_fields=tuple(sorted(metadata.required_fields)),
        minimum_history=metadata.minimum_history,
        default_required_history=metadata.default_required_history,
        parameter_fields=tuple(fields),
    )


class StrategyPageModel:
    def __init__(
        self,
        registry: StrategyRegistry,
        gateway: StrategyGateway,
        tasks: TaskGateway,
        optimizer: StrategyOptimizerGateway | None = None,
    ) -> None:
        self._registry = registry
        self._gateway = gateway
        self._tasks = tasks
        self._optimizer = optimizer
        self._optimization_task_id: str | None = None
        self._optimization_experiment_id: str | None = None

    def optimization_available(self) -> bool:
        return self._optimizer is not None

    def optimization_default_range(self, instance_id: str) -> tuple[date, date]:
        optimizer = self._optimizer
        if optimizer is None:
            raise StrategyPageError("STRATEGY_OPTIMIZER_UNAVAILABLE")
        return _boundary_call(
            "STRATEGY_OPTIMIZATION_RANGE_UNAVAILABLE",
            lambda: optimizer.default_range(instance_id),
        )

    def optimization_experiments(self) -> tuple[OptimizationExperiment, ...]:
        optimizer = self._optimizer
        if optimizer is None:
            return ()
        return _boundary_call(
            "STRATEGY_OPTIMIZATION_HISTORY_UNAVAILABLE",
            lambda: tuple(optimizer.list()),
        )

    def start_optimization(self, request: OptimizationRequest) -> TaskSnapshot:
        optimizer = self._optimizer
        if optimizer is None:
            raise StrategyPageError("STRATEGY_OPTIMIZER_UNAVAILABLE")
        experiment_id = optimizer.new_experiment_id()
        snapshot = _boundary_call(
            "STRATEGY_OPTIMIZATION_SUBMISSION_FAILED",
            lambda: self._tasks.submit(
                f"strategy-optimization:{experiment_id}",
                True,
                lambda: optimizer.run(experiment_id, request),
            ),
        )
        self._optimization_task_id = snapshot.task_id
        self._optimization_experiment_id = experiment_id
        return snapshot

    def optimization_task(self) -> TaskSnapshot | None:
        if self._optimization_task_id is None:
            return None
        return _boundary_call(
            "STRATEGY_OPTIMIZATION_TASK_UNAVAILABLE",
            lambda: self._tasks.status(self._optimization_task_id or ""),
        )

    def optimization_progress(self) -> OptimizationProgress | None:
        optimizer = self._optimizer
        experiment_id = self._optimization_experiment_id
        if optimizer is None or experiment_id is None:
            return None
        return _boundary_call(
            "STRATEGY_OPTIMIZATION_PROGRESS_UNAVAILABLE",
            lambda: optimizer.progress(experiment_id),
        )

    def publish_optimization(self, experiment_id: str) -> StrategyInstance:
        optimizer = self._optimizer
        if optimizer is None:
            raise StrategyPageError("STRATEGY_OPTIMIZER_UNAVAILABLE")
        return _boundary_call(
            "STRATEGY_OPTIMIZATION_PUBLISH_FAILED",
            lambda: optimizer.publish(experiment_id),
        )

    def state(self) -> StrategyPageState:
        definitions = _boundary_call(
            "STRATEGY_DEFINITIONS_UNAVAILABLE",
            lambda: tuple(_definition(item) for item in self._registry.list_metadata()),
        )
        instances = _boundary_call(
            "STRATEGY_STATE_UNAVAILABLE",
            lambda: tuple(
                self._validated_instance(item)
                for item in sorted(self._gateway.list(), key=lambda item: item.instance_id)
            ),
        )
        pools = _boundary_call(
            "STRATEGY_POOLS_UNAVAILABLE",
            self._available_pools,
        )
        return _boundary_call(
            "STRATEGY_STATE_UNAVAILABLE",
            lambda: StrategyPageState(definitions, instances, pools),
        )

    def _available_pools(self) -> tuple[StrategyPoolChoice, ...]:
        reader = getattr(self._gateway, "pools", None)
        if reader is None:
            return ()
        if not callable(reader):
            raise TypeError("strategy gateway pools reader must be callable")
        choices = tuple(reader())
        if any(type(item) is not StrategyPoolChoice for item in choices):
            raise TypeError("strategy gateway pools must contain exact choices")
        return tuple(sorted(choices, key=lambda item: item.watchlist_id))

    def create(self, form: StrategyFormModel) -> StrategyValidationResult:
        errors, draft = self._validate(form)
        if draft is None:
            return StrategyValidationResult(errors, None)
        instance = _boundary_call("STRATEGY_CREATE_FAILED", lambda: self._gateway.create(draft))
        checked = _boundary_call(
            "STRATEGY_CREATE_FAILED",
            lambda: self._created_instance(instance, draft),
        )
        return StrategyValidationResult({}, checked)

    def parameters_from_json(
        self, strategy_type: str, parameters_json: str
    ) -> Mapping[str, object]:
        checked_type = safe_identifier(strategy_type, label="strategy type")
        if type(parameters_json) is not str:
            raise TypeError("strategy parameters JSON must be exact text")

        def parse() -> Mapping[str, object]:
            metadata = self._registry.describe(checked_type)
            parsed = metadata.parameters_type.model_validate_json(parameters_json, strict=True)
            return _parameters(parsed.model_dump(mode="json"))

        return _boundary_call("STRATEGY_PARAMETERS_JSON_INVALID", parse)

    def copy(self, instance_id: str) -> StrategyInstance:
        source = self._instance(instance_id)
        copied = _boundary_call(
            "STRATEGY_COPY_FAILED", lambda: self._gateway.copy(source.instance_id)
        )

        def check() -> StrategyInstance:
            checked_copy = self._validated_instance(copied)
            if (
                checked_copy.instance_id == source.instance_id
                or checked_copy.lineage_id == source.lineage_id
                or checked_copy.version != 1
                or checked_copy.strategy_type != source.strategy_type
                or checked_copy.strategy_version != source.strategy_version
                or checked_copy.watchlist_id != source.watchlist_id
                or checked_copy.pool_snapshot_id != source.pool_snapshot_id
                or checked_copy.frequency is not source.frequency
                or not _exact_parameter_tree_equal(
                    checked_copy.parameters,
                    source.parameters,
                )
                or checked_copy.enabled is not True
                or checked_copy.created_at < source.created_at
            ):
                raise ValueError
            return checked_copy

        return _boundary_call("STRATEGY_COPY_FAILED", check)

    def create_version(self, instance_id: str, form: StrategyFormModel) -> StrategyValidationResult:
        source = self._instance(instance_id)
        errors, draft = self._validate(form)
        if draft is None:
            return StrategyValidationResult(errors, None)
        if draft.strategy_type != source.strategy_type:
            return StrategyValidationResult({"strategy_type": "编辑版本不能改变策略类型"}, None)
        revised = _boundary_call(
            "STRATEGY_VERSION_CREATE_FAILED",
            lambda: self._gateway.create_version(source.instance_id, draft),
        )

        def check() -> StrategyInstance:
            checked_revision = self._validated_instance(revised)
            if (
                checked_revision.instance_id == source.instance_id
                or checked_revision.lineage_id != source.lineage_id
                or checked_revision.version != source.version + 1
                or checked_revision.strategy_type != source.strategy_type
                or checked_revision.strategy_version != source.strategy_version
                or checked_revision.name != draft.name
                or checked_revision.watchlist_id != draft.watchlist_id
                or checked_revision.pool_snapshot_id != draft.pool_snapshot_id
                or checked_revision.frequency is not draft.frequency
                or not _exact_parameter_tree_equal(
                    checked_revision.parameters,
                    draft.parameters,
                )
                or checked_revision.enabled is not True
                or checked_revision.created_at < source.created_at
            ):
                raise ValueError
            return checked_revision

        checked = _boundary_call("STRATEGY_VERSION_CREATE_FAILED", check)
        return StrategyValidationResult({}, checked)

    def disable(self, instance_id: str) -> None:
        stable_id = safe_identifier(instance_id, label="strategy instance id")
        _boundary_call("STRATEGY_DISABLE_FAILED", lambda: self._gateway.disable(stable_id))

    def delete(self, instance_id: str) -> bool:
        source = self._instance(instance_id)
        return _boundary_call(
            "STRATEGY_DELETE_FAILED",
            lambda: self._gateway.delete(source.instance_id),
        )

    def start_quick_backtest(self, instance_id: str) -> TaskSnapshot:
        instance = self._instance(instance_id)
        expected_name = f"quick-backtest:{instance.instance_id}"

        def submit() -> TaskSnapshot:
            task = self._tasks.submit(
                expected_name,
                True,
                lambda: self._gateway.quick_backtest(instance.instance_id),
            )
            if type(task) is not TaskSnapshot:
                raise TypeError
            if task.name != expected_name or task.heavy is not True:
                raise ValueError
            return task

        return _boundary_call("STRATEGY_BACKTEST_SUBMISSION_FAILED", submit)

    def _instance(self, instance_id: str) -> StrategyInstance:
        stable_id = safe_identifier(instance_id, label="strategy instance id")

        def find() -> StrategyInstance:
            values = tuple(self._gateway.list())
            if any(type(item) is not StrategyInstance for item in values):
                raise TypeError
            matches = tuple(item for item in values if item.instance_id == stable_id)
            if len(matches) != 1:
                raise ValueError
            return self._validated_instance(matches[0])

        return _boundary_call("STRATEGY_INSTANCE_UNAVAILABLE", find)

    def _validated_instance(self, value: object) -> StrategyInstance:
        if type(value) is not StrategyInstance:
            raise TypeError("strategy gateway must return exact StrategyInstance values")
        assert isinstance(value, StrategyInstance)
        value.__post_init__()
        metadata = self._registry.describe(value.strategy_type)
        if value.strategy_version != metadata.version:
            raise ValueError("strategy instance version must match the registered definition")
        parsed = metadata.parameters_type.model_validate_json(
            strategy_parameters_json(value.parameters),
            strict=True,
        )
        canonical = _parameters(parsed.model_dump(mode="json"))
        if not _exact_parameter_tree_equal(value.parameters, canonical):
            raise ValueError("strategy instance parameters must be canonical primitives")
        return value

    def _validate(self, form: StrategyFormModel) -> tuple[Mapping[str, str], StrategyDraft | None]:
        if type(form) is not StrategyFormModel:
            raise TypeError("form must be an exact StrategyFormModel")
        errors: dict[str, str] = {}
        name = form.name.strip() if type(form.name) is str else ""
        if not name:
            errors["name"] = "策略名称不能为空"
        else:
            try:
                safe_display_text(name, label="strategy name")
            except (TypeError, ValueError):
                errors["name"] = "策略名称包含不安全文本"
        try:
            strategy_type = safe_identifier(form.strategy_type, label="strategy type")
            metadata = _boundary_call(
                "STRATEGY_DEFINITION_UNAVAILABLE",
                lambda: self._registry.describe(strategy_type),
            )
        except (TypeError, ValueError):
            strategy_type = ""
            metadata = None
            errors["strategy_type"] = "策略类型无效"
        if type(form.frequency) is not StrategyFrequency:
            errors["frequency"] = "策略频率无效"
        if type(form.watchlist_id) is not str:
            errors["watchlist_id"] = "标的池无效"
        else:
            try:
                safe_identifier(form.watchlist_id, label="watchlist id")
            except ValueError:
                errors["watchlist_id"] = "标的池无效"
        if metadata is None or errors:
            return frozen_errors(errors), None

        assert isinstance(form.watchlist_id, str)
        watchlist_id = form.watchlist_id
        pool = _boundary_call(
            "STRATEGY_POOL_UNAVAILABLE",
            lambda: self._gateway.pool(watchlist_id),
        )
        if type(pool) is not StrategyPool or pool.watchlist_id != watchlist_id:
            errors["watchlist_id"] = "标的池快照无效"
        elif (
            pool.asset_type not in metadata.supported_asset_types
            or pool.frequency not in metadata.supported_frequencies
            or pool.frequency is not form.frequency
        ):
            errors["watchlist_id"] = "标的池资产或频率不受此策略支持"

        if not isinstance(form.parameters, Mapping):
            errors["parameters"] = "策略参数必须是字段映射"
            return frozen_errors(errors), None
        try:
            parsed = (
                metadata.parameters_type.model_validate(dict(form.parameters), strict=True)
                if _contains_enum(form.parameters)
                else metadata.parameters_type.model_validate_json(
                    strategy_parameters_json(form.parameters),
                    strict=True,
                )
            )
        except ValidationError as error:
            for item in sorted(
                error.errors(include_url=False), key=lambda x: tuple(map(str, x["loc"]))
            ):
                location = ".".join(str(part) for part in item["loc"])
                field = f"parameters.{location}" if location else "parameters"
                if item["type"] == "extra_forbidden":
                    errors[field] = "不支持此参数"
                elif item["type"] == "missing":
                    errors[field] = "缺少必填参数"
                else:
                    errors[field] = "参数类型不正确"
            return frozen_errors(errors), None
        if errors:
            return frozen_errors(errors), None
        assert type(form.frequency) is StrategyFrequency
        draft = StrategyDraft(
            name=name,
            strategy_type=strategy_type,
            strategy_version=metadata.version,
            watchlist_id=watchlist_id,
            pool_snapshot_id=pool.snapshot_id,
            frequency=form.frequency,
            parameters=parsed.model_dump(mode="json"),
        )
        return MappingProxyType({}), draft

    def _created_instance(self, value: object, draft: StrategyDraft) -> StrategyInstance:
        checked = self._validated_instance(value)
        if (
            checked.version != 1
            or checked.strategy_type != draft.strategy_type
            or checked.strategy_version != draft.strategy_version
            or checked.name != draft.name
            or checked.watchlist_id != draft.watchlist_id
            or checked.pool_snapshot_id != draft.pool_snapshot_id
            or checked.frequency is not draft.frequency
            or not _exact_parameter_tree_equal(checked.parameters, draft.parameters)
            or checked.enabled is not True
        ):
            raise ValueError
        return checked


def render_strategies_page(model: StrategyPageModel | None) -> None:
    if model is None:
        ui.label("策略服务未配置；当前不会显示示例策略实例或虚构回测。")
        return
    try:
        state = model.state()
    except Exception:
        ui.label("策略状态读取失败，请查看本地脱敏日志。").classes("text-red-700")
        return
    if not state.definitions:
        ui.label("尚未注册策略定义。")
        return
    definitions = {item.strategy_type: item for item in state.definitions}
    first_definition = state.definitions[0]

    def default_parameters(definition: StrategyDefinition) -> Mapping[str, object]:
        if definition.strategy_type == "rule_dsl":
            return _RULE_DSL_DEFAULT_PARAMETERS
        return {field.name: field.default for field in definition.parameter_fields}

    ui.label("策略实验室只管理策略定义；运行、账户参数和结果统一放在“策略回测”页面。").classes(
        "text-sm text-slate-600"
    )
    ui.label("创建策略模板").classes("text-lg font-semibold")

    def update_definition(event: object) -> None:
        strategy_type = str(getattr(event, "value", definition_select.value))
        definition = definitions[strategy_type]
        frequency_select.options = {
            item.value: item.value for item in definition.supported_frequencies
        }
        frequency_select.value = definition.supported_frequencies[0].value
        parameters_input.value = strategy_parameters_json(default_parameters(definition))
        for control in (frequency_select, parameters_input):
            update = getattr(control, "update", None)
            if callable(update):
                update()
        custom_rule_builder.refresh()

    definition_select = ui.select(
        {item.strategy_type: item.display_name for item in state.definitions},
        label="策略定义",
        value=first_definition.strategy_type,
        on_change=update_definition,
    )
    name_input = ui.input("策略名称")
    pool_options = {
        item.watchlist_id: f"{item.name}（{'、'.join(map(str, item.instruments))}）"
        for item in state.pools
    }
    watchlist_select = ui.select(
        pool_options,
        label="标的池",
        value=None if not state.pools else state.pools[0].watchlist_id,
    )
    if not state.pools:
        ui.label("暂无已启用标的池，请先到“标的池”页面创建并启用标的。").classes(
            "text-sm text-amber-700"
        )
    frequency_select = ui.select(
        {item.value: item.value for item in first_definition.supported_frequencies},
        label="频率",
        value=first_definition.supported_frequencies[0].value,
    )
    parameters_input = ui.textarea(
        "参数 JSON", value=strategy_parameters_json(default_parameters(first_definition))
    ).classes("w-full font-mono")

    @ui.refreshable
    def custom_rule_builder() -> None:
        if str(definition_select.value) != "rule_dsl":
            return
        with ui.card().classes("w-full border border-indigo-200 bg-indigo-50 shadow-none"):
            ui.label("可视化条件构建器").classes("font-semibold")
            ui.label(
                "选择常见条件并设置导出变量；生成后仍可直接编辑下方参数 JSON 中的 DSL。"
            ).classes("text-sm text-slate-600")
            with ui.row().classes("w-full gap-4 items-end"):
                buy_template = ui.select(
                    {
                        "ma_cross": "短均线上穿长均线",
                        "price_ma": "收盘价上穿长均线",
                        "rsi": "RSI 低于买入阈值",
                    },
                    value="ma_cross",
                    label="买入条件",
                ).classes("min-w-64")
                sell_template = ui.select(
                    {
                        "ma_cross": "短均线下穿长均线",
                        "price_ma": "收盘价下穿长均线",
                        "rsi": "RSI 高于卖出阈值",
                    },
                    value="ma_cross",
                    label="卖出条件",
                ).classes("min-w-64")
                target_weight = ui.number(
                    "目标仓位（%）", value=100, min=1, max=100, step=5
                )

            variable_controls: dict[str, tuple[object, object, object, object, object]] = {}

            def variable_row(
                name: str,
                label: str,
                value: int,
                minimum: int,
                maximum: int,
                step: int,
            ) -> None:
                with ui.row().classes("w-full gap-3 items-end"):
                    ui.label(label).classes("w-28 text-sm")
                    current = ui.number("当前值", value=value, step=1)
                    lower = ui.number("最小值", value=minimum, step=1)
                    upper = ui.number("最大值", value=maximum, step=1)
                    stride = ui.number("步长", value=step, min=1, step=1)
                    optimize = ui.checkbox("参与优化", value=True)
                variable_controls[name] = (
                    current,
                    lower,
                    upper,
                    stride,
                    optimize,
                )

            variable_row("fast_window", "短均线窗口", 20, 5, 40, 5)
            variable_row("slow_window", "长均线窗口", 60, 40, 200, 20)
            variable_row("rsi_window", "RSI 窗口", 14, 6, 30, 2)
            variable_row("buy_rsi", "买入 RSI", 30, 15, 45, 5)
            variable_row("sell_rsi", "卖出 RSI", 70, 55, 85, 5)

            def generate_rule_parameters() -> None:
                buy_expressions = {
                    "ma_cross": "cross_above(sma(close, fast_window), sma(close, slow_window))",
                    "price_ma": "cross_above(close, sma(close, slow_window))",
                    "rsi": "rsi(close, rsi_window) < buy_rsi",
                }
                sell_expressions = {
                    "ma_cross": "cross_below(sma(close, fast_window), sma(close, slow_window))",
                    "price_ma": "cross_below(close, sma(close, slow_window))",
                    "rsi": "rsi(close, rsi_window) > sell_rsi",
                }
                buy_expression = buy_expressions[str(buy_template.value)]
                sell_expression = sell_expressions[str(sell_template.value)]
                referenced = f"{buy_expression} {sell_expression}"
                variables = []
                for name, controls in variable_controls.items():
                    current, lower, upper, stride, optimize = controls
                    variables.append(
                        {
                            "name": name,
                            "value": str(getattr(current, "value")),
                            "minimum": str(getattr(lower, "value")),
                            "maximum": str(getattr(upper, "value")),
                            "step": str(getattr(stride, "value")),
                            "optimize": bool(getattr(optimize, "value"))
                            and name in referenced,
                        }
                    )
                parameters_input.value = strategy_parameters_json(
                    {
                        "buy_expression": buy_expression,
                        "sell_expression": sell_expression,
                        "variables": tuple(variables),
                        "target_weight": str(
                            Decimal(str(target_weight.value)) / Decimal("100")
                        ),
                    }
                )
                parameters_input.update()
                ui.notify("DSL 与导出变量已生成。", type="positive")

            ui.button(
                "生成 DSL 与导出变量",
                on_click=generate_rule_parameters,
                icon="account_tree",
            )
            ui.label(
                "可用语法：sma、rsi、cross_above、cross_below、pct_change、"
                "highest、lowest，以及 and / or / not 和数值比较。"
            ).classes("text-xs text-slate-500")
            ui.label("没有被买卖表达式引用的变量不会交给优化器。").classes(
                "text-xs text-slate-500"
            )

    custom_rule_builder()
    ui.label("参数说明（来自已注册策略定义）").classes("font-semibold")
    for field in first_definition.parameter_fields:
        details = [f"默认={field.default!r}"]
        if field.minimum is not None or field.maximum is not None:
            details.append(f"范围={field.minimum}..{field.maximum}")
        if field.item_minimum is not None or field.item_maximum is not None:
            details.append(f"元素范围={field.item_minimum}..{field.item_maximum}")
        if field.allowed_values:
            details.append("可选=" + ",".join(map(str, field.allowed_values)))
        if field.nullable:
            details.append("可为空")
        ui.label(f"{field.name}：{field.description}；" + "；".join(details))
    create_feedback = ui.label("")

    def create_strategy() -> None:
        try:
            strategy_type = str(definition_select.value)
            parameters = model.parameters_from_json(strategy_type, str(parameters_input.value))
            result = model.create(
                StrategyFormModel(
                    name=name_input.value,
                    strategy_type=strategy_type,
                    watchlist_id=watchlist_select.value,
                    frequency=StrategyFrequency(str(frequency_select.value)),
                    parameters=parameters,
                )
            )
            if result.instance is None:
                create_feedback.set_text(
                    "创建失败："
                    + "；".join(f"{key}={value}" for key, value in result.errors.items())
                )
            else:
                ui.notify(f"创建成功：{result.instance.instance_id}", type="positive")
                ui.navigate.to("/strategies")
        except Exception as error:
            code = getattr(error, "code", "STRATEGY_FORM_INVALID")
            create_feedback.set_text(f"参数 JSON 或表单失败：{code}")

    ui.button("创建策略模板", on_click=create_strategy)
    ui.label("可用策略模板").classes("text-lg font-semibold mt-4")
    for definition in state.definitions:
        with ui.card().classes("w-full border border-slate-200 shadow-none"):
            ui.label(definition.display_name).classes("font-semibold")
            ui.label(f"{definition.strategy_type} / {definition.version}")
            ui.label(definition.description)
            ui.label(
                f"最少历史 {definition.minimum_history}；默认 {definition.default_required_history}"
            )
            guide = _STRATEGY_GUIDES.get(definition.strategy_type)
            if guide is not None:
                ui.label(f"基本原理：{guide[0]}").classes("text-sm text-slate-700")
                ui.label(f"适用场景：{guide[1]}").classes("text-sm text-slate-600")
                ui.label(f"主要风险：{guide[2]}").classes("text-sm text-amber-800")
    ui.label("已保存策略").classes("text-lg font-semibold mt-4")
    if not state.instances:
        ui.label("暂无已保存策略版本。")
    for instance in state.instances:
        with ui.card().classes("w-full border border-slate-200 shadow-none"):
            with ui.row().classes("w-full items-start justify-between gap-3"):
                with ui.column().classes("gap-1"):
                    ui.label(instance.name).classes("font-semibold")
                    ui.label(
                        f"{'启用' if instance.enabled else '已停用'} · "
                        f"{instance.strategy_type} · {instance.frequency.value}"
                    ).classes("text-sm text-slate-600")
                actions = ui.row().classes("items-center gap-1")
            ui.label(f"标的池快照：{instance.pool_snapshot_id}")
            ui.label(f"参数：{strategy_parameters_json(instance.parameters)}").classes(
                "text-xs text-slate-500 break-all"
            )
            feedback = ui.label("")

            def copy_instance(
                instance_id: str = instance.instance_id,
                feedback_element: Any = feedback,
            ) -> None:
                try:
                    copied = model.copy(instance_id)
                    ui.notify(f"复制成功：{copied.instance_id}", type="positive")
                    ui.navigate.to("/strategies")
                except Exception as error:
                    feedback_element.set_text(
                        f"复制失败：{getattr(error, 'code', 'STRATEGY_COPY_FAILED')}"
                    )

            def disable_instance(
                instance_id: str = instance.instance_id,
                feedback_element: Any = feedback,
            ) -> None:
                try:
                    model.disable(instance_id)
                    ui.notify(f"已停用：{instance_id}", type="positive")
                    ui.navigate.to("/strategies")
                except Exception as error:
                    feedback_element.set_text(
                        f"停用失败：{getattr(error, 'code', 'STRATEGY_DISABLE_FAILED')}"
                    )

            def delete_instance(
                instance_id: str = instance.instance_id,
                feedback_element: Any = feedback,
            ) -> None:
                try:
                    model.delete(instance_id)
                    ui.notify("策略已删除", type="positive")
                    ui.navigate.to("/strategies")
                except Exception as error:
                    feedback_element.set_text(
                        f"删除失败：{getattr(error, 'code', 'STRATEGY_DELETE_FAILED')}"
                    )

            with ui.dialog() as delete_dialog, ui.card():
                ui.label(f"删除“{instance.name}”？").classes("font-semibold")
                ui.label("将同时删除该策略的历史版本，此操作不可撤销。").classes(
                    "text-sm text-slate-600"
                )
                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("取消", on_click=delete_dialog.close).props("flat")
                    ui.button("删除", on_click=delete_instance).props(
                        "color=negative"
                    )

            with actions:
                ui.button(icon="content_copy", on_click=copy_instance).props(
                    "flat round aria-label=复制策略"
                ).tooltip("复制")
                disable_button = ui.button(
                    icon="pause_circle_outline",
                    on_click=disable_instance,
                ).props("flat round aria-label=停用策略")
                disable_button.tooltip("停用")
                if not instance.enabled:
                    disable_button.disable()
                ui.button(icon="delete_outline", on_click=delete_dialog.open).props(
                    "flat round color=negative aria-label=删除策略"
                ).tooltip("删除")

    _render_optimization_section(model, state)


def _render_optimization_section(
    model: StrategyPageModel,
    state: StrategyPageState,
) -> None:
    from compass.services.strategy_optimizer import (
        OptimizationRequest,
        OptimizationSearchSpace,
    )

    if not model.optimization_available():
        return
    candidates = tuple(
        item
        for item in state.instances
        if item.enabled and item.strategy_type == "dual_ma"
    )
    ui.separator().classes("my-6")
    ui.label("策略调优实验").classes("text-xl font-semibold")
    ui.label(
        "第一期对双均线的短周期、长周期和确认天数做网格搜索。数据按时间顺序切成 "
        "60% 训练、20% 验证、20% 冻结测试；只有验证集排名第一的候选会查看冻结测试。"
    ).classes("text-sm text-slate-600")
    if not candidates:
        ui.label("请先创建并启用一个双均线策略模板。需要至少约一年的本地行情。").classes(
            "text-amber-700"
        )
        return

    options = {
        item.instance_id: f"{item.name}（v{item.version}）"
        for item in candidates
    }
    source = ui.select(
        options,
        value=candidates[0].instance_id,
        label="作为调优起点的策略",
    ).classes("w-full max-w-xl")
    try:
        range_start, range_end = model.optimization_default_range(candidates[0].instance_id)
    except Exception:
        today = date.today()
        range_start, range_end = today.replace(year=today.year - 2), today
    with ui.row().classes("w-full gap-4 items-end"):
        start_input = ui.input("实验开始日期", value=range_start.isoformat()).props("type=date")
        end_input = ui.input("实验结束日期", value=range_end.isoformat()).props("type=date")

    def update_range() -> None:
        try:
            start, end = model.optimization_default_range(str(source.value))
            start_input.value = start.isoformat()
            end_input.value = end.isoformat()
            start_input.update()
            end_input.update()
        except Exception:
            ui.notify("无法读取该策略标的的共同数据区间。", type="negative")

    source.on_value_change(lambda _: update_range())
    with ui.row().classes("w-full gap-4 items-end"):
        short_input = ui.input("短均线候选", value="10,20,30")
        long_input = ui.input("长均线候选", value="40,60,90")
        confirmation_input = ui.input("确认天数候选", value="1,2,3")
    ui.label("候选值用逗号分隔，例如 10,20,30。短均线必须小于长均线。") \
        .classes("text-xs text-slate-500")
    feedback = ui.label("").classes("text-sm text-red-700")

    def parse_values(raw: object) -> tuple[int, ...]:
        if type(raw) is not str:
            raise ValueError
        return tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))

    def start_experiment() -> None:
        try:
            search_space = OptimizationSearchSpace(
                parse_values(short_input.value),
                parse_values(long_input.value),
                parse_values(confirmation_input.value),
            )
            model.start_optimization(
                OptimizationRequest(
                    str(source.value),
                    date.fromisoformat(str(start_input.value)),
                    date.fromisoformat(str(end_input.value)),
                    search_space,
                )
            )
            ui.notify(
                f"调优实验已启动，共 {search_space.trial_count} 组参数。",
                type="positive",
            )
            ui.navigate.to("/strategies")
        except Exception as error:
            feedback.set_text(
                "无法启动实验："
                + str(getattr(error, "code", error or "STRATEGY_OPTIMIZATION_INVALID"))
            )

    ui.button("开始调优实验", icon="tune", on_click=start_experiment).props("color=primary")
    ui.label("最多运行 50 组有效组合；实验回测不会进入普通回测任务历史。") \
        .classes("text-xs text-slate-500")

    task = model.optimization_task()
    active_statuses = {
        TaskStatus.QUEUED,
        TaskStatus.RUNNING,
        TaskStatus.CANCELLATION_REQUESTED,
    }
    if task is not None:
        labels = {
            TaskStatus.QUEUED: "等待运行",
            TaskStatus.RUNNING: "正在调优",
            TaskStatus.CANCELLATION_REQUESTED: "正在停止",
            TaskStatus.CANCELLED: "已停止",
            TaskStatus.SUCCEEDED: "已完成",
            TaskStatus.FAILED: "运行失败",
        }
        text = f"最近实验任务：{labels[task.status]}"
        if task.failure is not None:
            text += f"（{task.failure.code} / 错误 ID {task.failure.error_id}）"
        ui.label(text).classes(
            "text-sm "
            + ("text-red-700" if task.status is TaskStatus.FAILED else "text-blue-700")
        )
        progress = model.optimization_progress()
        phase_labels = {
            "preparing": "正在准备行情与策略",
            "training": "正在运行训练区间",
            "validation": "正在运行验证区间",
            "frozen_test": "正在检验最佳候选的冻结测试区间",
            "saving": "正在保存实验结果",
        }
        if progress is not None and task.status in active_statuses:
            detail = phase_labels[progress.phase]
            if progress.phase in {"training", "validation"}:
                detail += f" · 参数 {progress.current_trial}/{progress.trial_count}"
            if progress.short_window is not None:
                detail += (
                    f" · 短 {progress.short_window} / 长 {progress.long_window} / "
                    f"确认 {progress.confirmation_days} 天"
                )
            ui.label(detail).classes("text-sm font-medium text-slate-700")
            ui.linear_progress(value=progress.fraction).classes("w-full max-w-3xl")
            ui.label(
                "每组参数会先运行训练区间，再运行验证区间；因此同一个参数编号会显示两个阶段。"
            ).classes("text-xs text-slate-500")
        if task.status in active_statuses:
            def poll() -> None:
                current = model.optimization_task()
                if current is not None and current.status not in active_statuses:
                    poll_timer.deactivate()
                    ui.navigate.to("/strategies")

            poll_timer = ui.timer(1.0, poll)

    try:
        experiments = model.optimization_experiments()
    except Exception:
        ui.label("调优实验历史读取失败，请查看日志。").classes("text-sm text-red-700")
        return
    ui.label("实验结果").classes("text-lg font-semibold mt-4")
    if not experiments:
        ui.label("尚无调优实验。").classes("text-sm text-slate-500")
        return

    def percent(value: float | None) -> str:
        return "—" if value is None else f"{value:.2%}"

    for experiment in experiments:
        with ui.card().classes("w-full border border-slate-200 shadow-none"):
            with ui.row().classes("w-full justify-between items-start gap-3"):
                with ui.column().classes("gap-1"):
                    ui.label(experiment.source_name).classes("font-semibold")
                    ui.label(
                        f"{experiment.start} 至 {experiment.end} · "
                        f"训练截至 {experiment.training_end} · 验证截至 {experiment.validation_end}"
                    ).classes("text-sm text-slate-600")
                if experiment.published_instance_id is None:
                    def publish(experiment_id: str = experiment.experiment_id) -> None:
                        try:
                            created = model.publish_optimization(experiment_id)
                            ui.notify(f"已发布新策略版本：{created.name}", type="positive")
                            ui.navigate.to("/strategies")
                        except Exception as error:
                            ui.notify(
                                str(getattr(error, "code", "STRATEGY_OPTIMIZATION_PUBLISH_FAILED")),
                                type="negative",
                            )

                    button = ui.button(
                        "发布最佳候选",
                        icon="publish",
                        on_click=publish,
                    ).props("outline")
                    if (
                        not experiment.trials[0].eligible
                        or experiment.trials[0].frozen_test is None
                    ):
                        button.disable()
                else:
                    ui.label("已发布为新策略版本").classes("text-sm text-emerald-700")
            rows = []
            for trial in experiment.trials[:10]:
                rows.append(
                    {
                        "rank": trial.rank,
                        "parameters": (
                            f"{trial.short_window}/{trial.long_window}/"
                            f"{trial.confirmation_days}"
                        ),
                        "training": percent(trial.training.total_return),
                        "validation": percent(trial.validation.total_return),
                        "calmar": (
                            "—"
                            if trial.validation.calmar_ratio is None
                            else f"{trial.validation.calmar_ratio:.2f}"
                        ),
                        "drawdown": percent(trial.validation.maximum_drawdown),
                        "trades": trial.validation.trade_count,
                        "test": (
                            "—"
                            if trial.frozen_test is None
                            else percent(trial.frozen_test.total_return)
                        ),
                        "status": (
                            "可发布"
                            if trial.eligible
                            else (trial.rejection_reason or "未通过")
                        ),
                    }
                )
            result_table = ui.table(
                columns=[
                    {"name": "rank", "label": "排名", "field": "rank", "align": "left"},
                    {"name": "parameters", "label": "短/长/确认", "field": "parameters"},
                    {"name": "training", "label": "训练收益", "field": "training"},
                    {"name": "validation", "label": "验证收益", "field": "validation"},
                    {"name": "calmar", "label": "验证 Calmar", "field": "calmar"},
                    {"name": "drawdown", "label": "验证回撤", "field": "drawdown"},
                    {"name": "trades", "label": "验证成交", "field": "trades"},
                    {"name": "test", "label": "冻结测试收益", "field": "test"},
                    {"name": "status", "label": "状态", "field": "status"},
                ],
                rows=rows,
                row_key="rank",
                pagination={"rowsPerPage": 10},
            ).classes("w-full")
            result_table.add_slot(
                "header",
                """
                <q-tr :props="props">
                    <q-th key="rank" :props="props">排名</q-th>
                    <q-th key="parameters" :props="props">短/长/确认</q-th>
                    <q-th key="training" :props="props">
                        训练收益 <span class="q-ml-xs cursor-help fq-help-circle">?</span>
                        <q-tooltip max-width="320px">参数在前 60% 训练区间的累计收益。用于观察历史拟合效果，不能单独证明策略有效。</q-tooltip>
                    </q-th>
                    <q-th key="validation" :props="props">
                        验证收益 <span class="q-ml-xs cursor-help fq-help-circle">?</span>
                        <q-tooltip max-width="320px">同一参数在随后 20% 验证区间的累计收益。越高通常越好；若明显弱于训练收益，可能存在过拟合。</q-tooltip>
                    </q-th>
                    <q-th key="calmar" :props="props">
                        验证 Calmar <span class="q-ml-xs cursor-help fq-help-circle">?</span>
                        <q-tooltip max-width="320px">验证区间的年化收益与最大回撤之比，越高通常越好。成交很少或区间较短时需要谨慎解读。</q-tooltip>
                    </q-th>
                    <q-th key="drawdown" :props="props">验证回撤</q-th>
                    <q-th key="trades" :props="props">验证成交</q-th>
                    <q-th key="test" :props="props">
                        冻结测试收益 <span class="q-ml-xs cursor-help fq-help-circle">?</span>
                        <q-tooltip max-width="320px">最佳候选在最后 20% 未参与排名的数据上的累计收益，最接近面对未知行情的检验；只有第一名会计算。</q-tooltip>
                    </q-th>
                    <q-th key="status" :props="props">状态</q-th>
                </q-tr>
                """,
            )
            ui.add_css("""
                .fq-help-circle {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: 16px;
                    height: 16px;
                    border: 1px solid currentColor;
                    border-radius: 9999px;
                    color: #64748b;
                    font-size: 11px;
                    line-height: 1;
                }
            """)
