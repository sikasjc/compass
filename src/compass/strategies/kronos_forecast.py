from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from hashlib import sha256
import importlib
from math import isfinite
from threading import RLock
from typing import ClassVar, Protocol

import pandas as pd  # type: ignore[import-untyped]
from pydantic import Field, model_validator

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
from compass.strategies.indicators import simple_moving_average
from compass.strategies.momentum import (
    _equal_weights,
    _normalize_strategy_id,
    _prepare_context,
)


_MODEL_REPOSITORIES = {
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2_048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}


@dataclass(frozen=True, slots=True)
class KronosForecast:
    instrument: InstrumentId
    as_of: date
    horizon: int
    expected_return: float
    path_positive_ratio: float
    predicted_closes: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.instrument) is not InstrumentId:
            raise TypeError("forecast instrument must be exact")
        if type(self.as_of) is not date:
            raise TypeError("forecast as_of must be an exact date")
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ValueError("forecast horizon must be positive")
        for name in ("expected_return", "path_positive_ratio"):
            value = getattr(self, name)
            if type(value) is not float or not isfinite(value):
                raise ValueError(f"forecast {name} must be a finite exact float")
        if not 0 <= self.path_positive_ratio <= 1:
            raise ValueError("forecast path_positive_ratio must be between zero and one")
        if len(self.predicted_closes) != self.horizon or any(
            type(value) is not float or not isfinite(value) or value <= 0
            for value in self.predicted_closes
        ):
            raise ValueError("forecast closes must contain one positive value per horizon day")


@dataclass(frozen=True, slots=True)
class KronosForecastDiagnostic:
    instrument: InstrumentId
    action: str
    expected_return: float
    path_positive_ratio: float
    rank: int
    close: float
    trend_value: float
    trend_passed: bool
    target_weight: Decimal
    reason_code: str


class KronosForecaster(Protocol):
    def forecast(
        self,
        histories: Mapping[InstrumentId, pd.DataFrame],
        *,
        as_of: date,
        lookback: int,
        horizon: int,
        temperature: float,
        top_p: float,
        sample_count: int,
        seed: int,
    ) -> Sequence[KronosForecast]: ...


class KronosRuntimeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class KronosRuntimeStatus:
    installed: bool
    torch_version: str | None
    cuda_build: str | None
    cuda_available: bool
    device_name: str | None

    @property
    def display_text(self) -> str:
        if not self.installed:
            return "Kronos/PyTorch 尚未安装"
        if self.cuda_available:
            return (
                f"CUDA 可用 · {self.device_name or 'NVIDIA GPU'} · "
                f"PyTorch {self.torch_version} / CUDA {self.cuda_build}"
            )
        return f"当前仅 CPU · PyTorch {self.torch_version}"

    @property
    def action_text(self) -> str | None:
        if self.cuda_available:
            return None
        if not self.installed:
            return "请先运行 Kronos 安装脚本；NVIDIA 用户选择 CUDA 模式。"
        return "需要 GPU 时运行 scripts/install-kronos.ps1 -Mode CUDA；可用 -Proxy 指定代理。"


def kronos_runtime_status() -> KronosRuntimeStatus:
    try:
        importlib.import_module("model")
        torch = importlib.import_module("torch")
    except ImportError:
        return KronosRuntimeStatus(False, None, None, False, None)
    version = str(getattr(torch, "__version__", "unknown"))
    torch_version = getattr(torch, "version", None)
    cuda_build_value = None if torch_version is None else getattr(torch_version, "cuda", None)
    cuda_build = None if cuda_build_value is None else str(cuda_build_value)
    cuda = getattr(torch, "cuda", None)
    is_available = None if cuda is None else getattr(cuda, "is_available", None)
    cuda_available = bool(callable(is_available) and is_available())
    device_name: str | None = None
    get_device_name = None if cuda is None else getattr(cuda, "get_device_name", None)
    if cuda_available and callable(get_device_name):
        device_name = str(get_device_name(0))
    return KronosRuntimeStatus(True, version, cuda_build, cuda_available, device_name)


class KronosModelForecaster:
    """Lazy adapter around the optional Kronos model package and HF weights."""

    _inference_lock = RLock()
    _forecast_cache: ClassVar[
        OrderedDict[
            tuple[str, str, str, int, int, float, float, int, int, str],
            tuple[KronosForecast, ...],
        ]
    ] = OrderedDict()
    _forecast_cache_limit = 512

    def __init__(
        self,
        *,
        model_size: str = "mini",
        device: str = "auto",
        model_loader: Callable[[], object] | None = None,
    ) -> None:
        if model_size not in _MODEL_REPOSITORIES:
            raise ValueError("unsupported Kronos model size")
        if device not in {"auto", "cpu", "cuda", "mps"} and not device.startswith("cuda:"):
            raise ValueError("unsupported Kronos device")
        self.model_size = model_size
        if device == "auto":
            status = kronos_runtime_status()
            if status.cuda_available:
                self.device = "cuda"
            else:
                self.device = "mps" if _mps_available() else "cpu"
        else:
            self.device = device
        self._model_loader = model_loader or self._load_predictor
        self._cache_namespace = (
            "default" if model_loader is None else f"custom:{id(model_loader)}"
        )
        self._predictor: object | None = None

    @classmethod
    def clear_forecast_cache(cls) -> None:
        with cls._inference_lock:
            cls._forecast_cache.clear()

    @staticmethod
    def _history_digest(
        histories: Mapping[InstrumentId, pd.DataFrame],
        ordered: Sequence[InstrumentId],
        lookback: int,
    ) -> str:
        digest = sha256()
        columns = ("open", "high", "low", "close", "volume", "amount")
        for instrument in ordered:
            frame = histories[instrument].tail(lookback).loc[:, columns]
            digest.update(str(instrument).encode("utf-8"))
            digest.update(
                pd.util.hash_pandas_object(frame, index=True).values.tobytes()
            )
        return digest.hexdigest()

    def _load_predictor(self) -> object:
        try:
            module = importlib.import_module("model")
            torch = importlib.import_module("torch")
            kronos = getattr(module, "Kronos")
            tokenizer_type = getattr(module, "KronosTokenizer")
            predictor_type = getattr(module, "KronosPredictor")
        except (AttributeError, ImportError) as error:
            raise KronosRuntimeUnavailable(
                "KRONOS_OPTIONAL_DEPENDENCY_MISSING: run `uv sync --extra kronos` "
                "or `uv sync --extra kronos-cuda`"
            ) from error
        if self.device is not None and self.device.startswith("cuda"):
            cuda = getattr(torch, "cuda", None)
            is_available = None if cuda is None else getattr(cuda, "is_available", None)
            if not callable(is_available) or not is_available():
                raise KronosRuntimeUnavailable(
                    "KRONOS_CUDA_UNAVAILABLE: install a CUDA-enabled PyTorch build or choose CPU"
                )
        model_repository, tokenizer_repository, max_context = _MODEL_REPOSITORIES[self.model_size]
        tokenizer = tokenizer_type.from_pretrained(tokenizer_repository)
        model = kronos.from_pretrained(model_repository)
        return predictor_type(
            model,
            tokenizer,
            device=self.device,
            max_context=max_context,
        )

    @property
    def predictor(self) -> object:
        if self._predictor is None:
            self._predictor = self._model_loader()
        return self._predictor

    def forecast(
        self,
        histories: Mapping[InstrumentId, pd.DataFrame],
        *,
        as_of: date,
        lookback: int,
        horizon: int,
        temperature: float,
        top_p: float,
        sample_count: int,
        seed: int,
    ) -> tuple[KronosForecast, ...]:
        if not histories:
            return ()
        ordered = tuple(sorted(histories, key=str))
        cache_key = (
            self._cache_namespace,
            self.model_size,
            self.device,
            lookback,
            horizon,
            temperature,
            top_p,
            sample_count,
            seed,
            f"{as_of.isoformat()}:{self._history_digest(histories, ordered, lookback)}",
        )
        with self._inference_lock:
            cached = self._forecast_cache.get(cache_key)
            if cached is not None:
                self._forecast_cache.move_to_end(cache_key)
                return cached
        prepared: list[pd.DataFrame] = []
        x_timestamps: list[pd.Series] = []
        y_timestamps: list[pd.Series] = []
        latest_closes: list[float] = []
        for instrument in ordered:
            frame = histories[instrument].tail(lookback).copy(deep=True)
            if len(frame) != lookback:
                raise ValueError("KRONOS_HISTORY_LENGTH_MISMATCH")
            prepared.append(frame.loc[:, ["open", "high", "low", "close", "volume", "amount"]])
            x_timestamps.append(pd.Series(frame.index))
            future = pd.bdate_range(
                start=pd.Timestamp(as_of) + timedelta(days=1),
                periods=horizon,
            )
            y_timestamps.append(pd.Series(future))
            latest_closes.append(float(frame["close"].iloc[-1]))
        predict_batch = getattr(self.predictor, "predict_batch", None)
        if not callable(predict_batch):
            raise KronosRuntimeUnavailable("KRONOS_PREDICT_BATCH_UNAVAILABLE")
        try:
            torch = importlib.import_module("torch")
            manual_seed = getattr(torch, "manual_seed")
        except (AttributeError, ImportError) as error:
            raise KronosRuntimeUnavailable("KRONOS_TORCH_UNAVAILABLE") from error
        with self._inference_lock:
            cached = self._forecast_cache.get(cache_key)
            if cached is not None:
                self._forecast_cache.move_to_end(cache_key)
                return cached
            manual_seed(seed)
            cuda = getattr(torch, "cuda", None)
            cuda_seed = None if cuda is None else getattr(cuda, "manual_seed_all", None)
            if callable(cuda_seed):
                cuda_seed(seed)
            predicted = predict_batch(
                df_list=prepared,
                x_timestamp_list=x_timestamps,
                y_timestamp_list=y_timestamps,
                pred_len=horizon,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False,
            )
        if not isinstance(predicted, Sequence) or len(predicted) != len(ordered):
            raise ValueError("KRONOS_PREDICTION_COUNT_MISMATCH")
        result: list[KronosForecast] = []
        for instrument, latest_close, frame in zip(ordered, latest_closes, predicted, strict=True):
            if not isinstance(frame, pd.DataFrame) or "close" not in frame:
                raise ValueError("KRONOS_PREDICTION_FRAME_INVALID")
            closes = tuple(float(value) for value in frame["close"])
            result.append(
                KronosForecast(
                    instrument=instrument,
                    as_of=as_of,
                    horizon=horizon,
                    expected_return=closes[-1] / latest_close - 1,
                    path_positive_ratio=sum(value > latest_close for value in closes) / len(closes),
                    predicted_closes=closes,
                )
            )
        forecasts = tuple(result)
        with self._inference_lock:
            self._forecast_cache[cache_key] = forecasts
            self._forecast_cache.move_to_end(cache_key)
            while len(self._forecast_cache) > self._forecast_cache_limit:
                self._forecast_cache.popitem(last=False)
        return forecasts


def _mps_available() -> bool:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return False
    backends = getattr(torch, "backends", None)
    mps = None if backends is None else getattr(backends, "mps", None)
    is_available = None if mps is None else getattr(mps, "is_available", None)
    return bool(callable(is_available) and is_available())


class KronosForecastParameters(StrategyParameters):
    model_size: str = Field(
        default="mini",
        pattern="^(mini|small|base)$",
        description="Kronos 模型规模；mini 是首次接入和 CPU 环境的推荐起点。",
    )
    device: str = Field(
        default="auto",
        pattern="^(auto|cpu|cuda|cuda:[0-9]+|mps)$",
        description="推理设备；auto 会依次选择 CUDA、MPS 或 CPU。",
    )
    lookback: int = Field(
        default=256,
        strict=True,
        ge=64,
        le=2_048,
        description="每次预测使用的历史日线数量。",
    )
    horizon: int = Field(
        default=5,
        strict=True,
        ge=1,
        le=60,
        description="预测未来交易日数量。",
    )
    rebalance_interval: int = Field(
        default=5,
        strict=True,
        ge=1,
        le=60,
        description="两次模型预测之间至少间隔的交易日数量。",
    )
    entry_return: Decimal = Field(
        default=Decimal("0.02"),
        strict=True,
        allow_inf_nan=False,
        ge=0,
        le=Decimal("0.50"),
        description="预测收益达到该阈值才允许进入。",
    )
    exit_return: Decimal = Field(
        default=Decimal("-0.01"),
        strict=True,
        allow_inf_nan=False,
        ge=Decimal("-0.50"),
        le=0,
        description="预测收益低于该阈值时退出。",
    )
    minimum_path_positive_ratio: Decimal = Field(
        default=Decimal("0.60"),
        strict=True,
        allow_inf_nan=False,
        ge=0,
        le=1,
        description="预测路径中高于当前收盘价的最低比例。",
    )
    trend_window: int = Field(
        default=60,
        strict=True,
        ge=2,
        le=512,
        description="用于阻止逆长期趋势买入的移动平均窗口。",
    )
    top_n: int = Field(
        default=2,
        strict=True,
        ge=1,
        le=20,
        description="每次最多持有的预测最强标的数量。",
    )
    target_weight: Decimal = Field(
        default=Decimal("1"),
        strict=True,
        allow_inf_nan=False,
        gt=0,
        le=1,
        description="策略袖套内所有入选标的的目标总权重。",
    )
    temperature: float = Field(
        default=0.8,
        strict=True,
        gt=0,
        le=2,
        description="Kronos 采样温度。",
    )
    top_p: float = Field(
        default=0.9,
        strict=True,
        gt=0,
        le=1,
        description="Kronos nucleus sampling 概率。",
    )
    sample_count: int = Field(
        default=3,
        strict=True,
        ge=1,
        le=20,
        description="每个标的生成并平均的预测路径数量。",
    )
    seed: int = Field(
        default=42,
        strict=True,
        ge=0,
        le=2_147_483_647,
        description="保证相同数据与参数得到相同采样结果的随机种子。",
    )

    @model_validator(mode="after")
    def validate_model_limits(self) -> KronosForecastParameters:
        max_context = _MODEL_REPOSITORIES[self.model_size][2]
        if self.lookback > max_context:
            raise ValueError("lookback exceeds selected Kronos model context")
        if self.exit_return >= self.entry_return:
            raise ValueError("exit return must be below entry return")
        weight_to_units(self.target_weight, label="target_weight")
        return self


class KronosForecastStrategy:
    strategy_type = "kronos_forecast"
    parameters_type = KronosForecastParameters
    minimum_history = 64
    metadata = StrategyMetadata(
        strategy_type=strategy_type,
        version="1.0.0",
        display_name="Kronos K 线预测",
        description="用 Kronos 预测未来 K 线，再通过收益阈值、路径一致性和趋势过滤生成目标仓位。",
        supported_asset_types=frozenset({AssetType.ETF, AssetType.STOCK}),
        supported_frequencies=frozenset({StrategyFrequency.DAILY}),
        required_fields=frozenset({"open", "high", "low", "close", "volume", "amount"}),
        minimum_history=minimum_history,
        default_required_history=256,
        parameters_type=parameters_type,
    )

    def __init__(
        self,
        parameters: KronosForecastParameters | None = None,
        strategy_id: str = "kronos_forecast",
        forecaster: KronosForecaster | None = None,
    ) -> None:
        self.parameters = parameters or KronosForecastParameters()
        self.required_history = max(self.parameters.lookback, self.parameters.trend_window)
        self.strategy_id = _normalize_strategy_id(strategy_id)
        self._forecaster = forecaster or KronosModelForecaster(
            model_size=self.parameters.model_size,
            device=self.parameters.device,
        )
        self._last_forecast_positions: Mapping[InstrumentId, int] | None = None
        self._last_target_weights: Mapping[InstrumentId, Decimal] | None = None
        self._latest_diagnostics: tuple[KronosForecastDiagnostic, ...] = ()

    @property
    def latest_diagnostics(self) -> tuple[KronosForecastDiagnostic, ...]:
        return self._latest_diagnostics

    def generate_targets(self, context: StrategyContext) -> StrategyDecision:
        prepared = _prepare_context(context, self.metadata.supported_asset_types)
        if isinstance(prepared, StrategyDecision):
            return prepared
        eligible = {
            instrument: prepared.histories[instrument]
            for instrument in prepared.instruments
            if len(prepared.histories[instrument]) >= self.required_history
        }
        if not eligible:
            self._latest_diagnostics = ()
            return StrategyDecision.empty(
                StrategyDecisionStatus.SKIPPED,
                "INSUFFICIENT_HISTORY",
                details={"required_history": self.required_history},
            )
        positions = {instrument: len(frame) for instrument, frame in eligible.items()}
        if self._last_forecast_positions is not None and all(
            positions.get(instrument, 0) - self._last_forecast_positions.get(instrument, 0)
            < self.parameters.rebalance_interval
            for instrument in eligible
        ):
            self._latest_diagnostics = ()
            targets = self._last_target_weights or {}
            return StrategyDecision.generated(
                tuple(
                    TargetIntent(
                        strategy_id=self.strategy_id,
                        instrument=instrument,
                        target_weight=targets.get(instrument, Decimal("0")),
                        score=0.0,
                        confidence=1.0,
                        reason_code="KRONOS_REBALANCE_HOLD",
                        valid_until=context.as_of,
                    )
                    for instrument in sorted(eligible, key=str)
                ),
                details={
                    "rebalance_interval": self.parameters.rebalance_interval,
                    "reused_targets": True,
                },
            )
        forecasts = self._forecaster.forecast(
            eligible,
            as_of=context.as_of,
            lookback=self.parameters.lookback,
            horizon=self.parameters.horizon,
            temperature=self.parameters.temperature,
            top_p=self.parameters.top_p,
            sample_count=self.parameters.sample_count,
            seed=self.parameters.seed,
        )
        self._last_forecast_positions = positions
        by_instrument = {forecast.instrument: forecast for forecast in forecasts}
        entry_return = float(self.parameters.entry_return)
        exit_return = float(self.parameters.exit_return)
        minimum_positive = float(self.parameters.minimum_path_positive_ratio)
        previous_targets = self._last_target_weights or {}
        active = {
            instrument
            for instrument in eligible
            if previous_targets.get(instrument, Decimal("0")) > 0
            or (
                (holding := context.holding(instrument)) is not None
                and holding.quantity > 0
            )
        }
        kept: list[tuple[InstrumentId, float]] = []
        entry_candidates: list[tuple[InstrumentId, float]] = []
        market_state: dict[InstrumentId, tuple[float, float, bool]] = {}
        for instrument, frame in eligible.items():
            forecast = by_instrument.get(instrument)
            if forecast is None:
                continue
            close = float(frame["close"].iloc[-1])
            trend = simple_moving_average(frame["close"], self.parameters.trend_window)
            trend_value = float(trend.iloc[-1])
            trend_passed = isfinite(trend_value) and close >= trend_value
            market_state[instrument] = (close, trend_value, trend_passed)
            if instrument in active:
                if forecast.expected_return > exit_return:
                    kept.append((instrument, forecast.expected_return))
            elif (
                forecast.expected_return >= entry_return
                and forecast.path_positive_ratio >= minimum_positive
                and trend_passed
            ):
                entry_candidates.append((instrument, forecast.expected_return))
        entry_candidates.sort(key=lambda item: (-item[1], str(item[0])))
        selected = kept + entry_candidates[: max(0, self.parameters.top_n - len(kept))]
        selected.sort(key=lambda item: (-item[1], str(item[0])))
        selected_weights = dict(
            zip(
                (instrument for instrument, _ in selected),
                _equal_weights(len(selected), self.parameters.target_weight),
                strict=True,
            )
        )
        self._last_target_weights = {
            instrument: selected_weights.get(instrument, Decimal("0"))
            for instrument in eligible
        }
        ranks = {
            instrument: rank
            for rank, (instrument, _) in enumerate(
                sorted(
                    (
                        (instrument, forecast.expected_return)
                        for instrument, forecast in by_instrument.items()
                        if instrument in eligible
                    ),
                    key=lambda item: (-item[1], str(item[0])),
                ),
                start=1,
            )
        }
        intents = []
        diagnostics = []
        for instrument in sorted(eligible, key=str):
            forecast = by_instrument.get(instrument)
            if forecast is None:
                continue
            target = selected_weights.get(instrument, Decimal("0"))
            was_active = instrument in active
            if target > 0 and was_active:
                action = "HOLD"
                reason = "KRONOS_FORECAST_HOLD"
                threshold = max(abs(exit_return), 0.000001)
            elif target > 0:
                action = "BUY"
                reason = "KRONOS_FORECAST_ENTRY"
                threshold = entry_return
            elif was_active:
                action = "SELL"
                reason = "KRONOS_FORECAST_EXIT"
                threshold = abs(exit_return)
            else:
                action = "CASH"
                reason = "KRONOS_FORECAST_CASH"
                threshold = max(entry_return, abs(exit_return))
            intents.append(
                TargetIntent(
                    strategy_id=self.strategy_id,
                    instrument=instrument,
                    target_weight=target,
                    score=forecast.expected_return,
                    confidence=min(
                        1.0,
                        abs(forecast.expected_return) / max(threshold, 0.000001),
                    ),
                    reason_code=reason,
                    valid_until=context.as_of,
                )
            )
            close, trend_value, trend_passed = market_state[instrument]
            diagnostics.append(
                KronosForecastDiagnostic(
                    instrument=instrument,
                    action=action,
                    expected_return=forecast.expected_return,
                    path_positive_ratio=forecast.path_positive_ratio,
                    rank=ranks[instrument],
                    close=close,
                    trend_value=trend_value,
                    trend_passed=trend_passed,
                    target_weight=target,
                    reason_code=reason,
                )
            )
        self._latest_diagnostics = tuple(diagnostics)
        details = {
            "model_size": self.parameters.model_size,
            "forecast_count": len(forecasts),
            "selected_count": len(selected),
            "cash_count": len(intents) - len(selected),
            "horizon": self.parameters.horizon,
        }
        if not intents:
            return StrategyDecision.empty(
                StrategyDecisionStatus.SKIPPED,
                "KRONOS_FORECAST_MISSING",
                details=details,
            )
        return StrategyDecision.generated(intents, details=details)


def create_kronos_forecast_strategy(
    parameters: KronosForecastParameters | None = None,
    strategy_id: str = "kronos_forecast",
) -> KronosForecastStrategy:
    return KronosForecastStrategy(parameters, strategy_id)


create_kronos_forecast_strategy.metadata = KronosForecastStrategy.metadata  # type: ignore[attr-defined]
