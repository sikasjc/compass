from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import re
from typing import Protocol, TypeVar

from nicegui import ui

from compass.data.base import default_instrument_type
from compass.domain.market import AssetType, Exchange, InstrumentId
from compass.services.instrument_names import common_instrument_name
from compass.services.safe_display import (
    frozen_errors,
    safe_display_text,
    safe_identifier,
    stable_code,
)


InstrumentResolver = Callable[[InstrumentId], AssetType]
DataRangeLoader = Callable[[], Sequence["WatchlistDataRange"]]
InstrumentRetry = Callable[[InstrumentId], object]
_SEPARATOR = re.compile(r"[\s,]+")
_PLAIN_CODE = re.compile(r"\d{6}\Z")
T = TypeVar("T")
_MISSING = object()
QUICK_SELECTIONS = (
    ("大盘核心", ("SSE.000016", "SSE.000300")),
    ("全市场宽基", ("SSE.000300", "SSE.000905", "SSE.000852")),
    ("成长科技", ("SZSE.399006", "SZSE.399673", "SSE.000688")),
    ("红利风格", ("SSE.000015",)),
)


class WatchlistPageError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = stable_code(code, label="watchlist page error code")
        super().__init__(self.code)


def _gateway_call(code: str, operation: Callable[[], T]) -> T:
    result: object = _MISSING
    try:
        result = operation()
    except Exception:
        raise WatchlistPageError(code) from None
    return result


def _instruments(value: Sequence[InstrumentId]) -> tuple[InstrumentId, ...]:
    instruments = tuple(value)
    if not instruments:
        raise ValueError("watchlist instruments must not be empty")
    if any(type(item) is not InstrumentId for item in instruments):
        raise TypeError("watchlist instruments must contain exact InstrumentId values")
    if len(set(instruments)) != len(instruments):
        raise ValueError("watchlist instruments must be unique")
    if instruments != tuple(sorted(instruments, key=str)):
        raise ValueError("watchlist instruments must be sorted")
    return instruments


@dataclass(frozen=True, slots=True)
class WatchlistDraft:
    name: str
    instruments: tuple[InstrumentId, ...]

    def __post_init__(self) -> None:
        safe_display_text(self.name, label="watchlist name")
        object.__setattr__(self, "instruments", _instruments(self.instruments))


@dataclass(frozen=True, slots=True)
class WatchlistValidationResult:
    errors: Mapping[str, str]
    draft: WatchlistDraft | None

    def __post_init__(self) -> None:
        errors = frozen_errors(self.errors)
        if self.draft is not None and type(self.draft) is not WatchlistDraft:
            raise TypeError("draft must be an exact WatchlistDraft or None")
        if (self.draft is None) == (not bool(errors)):
            raise ValueError("validation must contain either errors or a draft")
        object.__setattr__(self, "errors", errors)


@dataclass(frozen=True, slots=True)
class WatchlistEntry:
    watchlist_id: str
    name: str
    instruments: tuple[InstrumentId, ...]
    enabled: bool

    def __post_init__(self) -> None:
        safe_identifier(self.watchlist_id, label="watchlist id")
        safe_display_text(self.name, label="watchlist name")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be an exact bool")
        object.__setattr__(self, "instruments", _instruments(self.instruments))


@dataclass(frozen=True, slots=True)
class WatchlistDataRange:
    instrument: InstrumentId
    first_day: date
    last_day: date

    def __post_init__(self) -> None:
        if type(self.instrument) is not InstrumentId:
            raise TypeError("watchlist data range instrument must be an exact InstrumentId")
        if type(self.first_day) is not date or type(self.last_day) is not date:
            raise TypeError("watchlist data range must contain exact dates")
        if self.first_day > self.last_day:
            raise ValueError("watchlist data range is reversed")


@dataclass(frozen=True, slots=True)
class WatchlistPageState:
    entry: WatchlistEntry | None
    data_ranges: tuple[WatchlistDataRange, ...] = ()

    def __post_init__(self) -> None:
        if self.entry is not None and type(self.entry) is not WatchlistEntry:
            raise TypeError("entry must be an exact WatchlistEntry or None")
        ranges = tuple(self.data_ranges)
        if any(type(item) is not WatchlistDataRange for item in ranges) or tuple(
            item.instrument for item in ranges
        ) != tuple(sorted({item.instrument for item in ranges}, key=str)):
            raise ValueError("watchlist data ranges must be unique and sorted")
        object.__setattr__(self, "data_ranges", ranges)


@dataclass(frozen=True, slots=True)
class WatchlistFormModel:
    name: object
    instruments_text: object
    resolver: InstrumentResolver = default_instrument_type

    def validate(self) -> WatchlistValidationResult:
        if not callable(self.resolver):
            raise TypeError("resolver must be callable")
        clean_name = self.name.strip() if type(self.name) is str else ""
        errors: dict[str, str] = {}
        if not clean_name:
            errors["name"] = "标的池名称不能为空"
        if type(self.instruments_text) is not str:
            errors["instruments"] = "标的代码必须使用文本批量输入"
            return WatchlistValidationResult(errors, None)
        values = tuple(value for value in _SEPARATOR.split(self.instruments_text.strip()) if value)
        if not values:
            errors["instruments"] = "至少输入一个标的代码"
            return WatchlistValidationResult(errors, None)
        if len(set(values)) != len(values):
            errors["instruments"] = "标的代码不能重复"
            return WatchlistValidationResult(errors, None)
        instruments: list[InstrumentId] = []
        for value in values:
            try:
                instrument = InstrumentId.parse(value)
                if str(instrument) != value:
                    raise ValueError
                if instrument.exchange not in {Exchange.SSE, Exchange.SZSE}:
                    raise ValueError
                asset_type = self.resolver(instrument)
                if type(asset_type) is not AssetType:
                    raise ValueError
            except Exception:
                errors["instruments"] = "当前只支持上证和深证的六位标的代码"
                break
            instruments.append(instrument)
        if errors:
            return WatchlistValidationResult(errors, None)
        return WatchlistValidationResult(
            {},
            WatchlistDraft(clean_name, tuple(sorted(instruments, key=str))),
        )


class WatchlistGateway(Protocol):
    def primary(self) -> WatchlistEntry | None: ...

    def save_primary(self, draft: WatchlistDraft) -> None: ...


class WatchlistPageModel:
    def __init__(
        self,
        gateway: WatchlistGateway,
        data_range_loader: DataRangeLoader | None = None,
        instrument_retry: InstrumentRetry | None = None,
    ) -> None:
        self._gateway = gateway
        self._data_range_loader = data_range_loader or (lambda: ())
        self._instrument_retry = instrument_retry

    def state(self) -> WatchlistPageState:
        entry = _gateway_call("WATCHLIST_STATE_UNAVAILABLE", self._gateway.primary)
        ranges = tuple(_gateway_call("WATCHLIST_DATA_RANGES_UNAVAILABLE", self._data_range_loader))
        return WatchlistPageState(entry, ranges)

    def save(self, form: WatchlistFormModel) -> WatchlistValidationResult:
        if type(form) is not WatchlistFormModel:
            raise TypeError("form must be a WatchlistFormModel")
        result = form.validate()
        draft = result.draft
        if draft is not None:
            _gateway_call(
                "WATCHLIST_SAVE_FAILED",
                lambda: self._gateway.save_primary(draft),
            )
        return result

    @property
    def supports_instrument_retry(self) -> bool:
        return self._instrument_retry is not None

    def retry_instrument(self, instrument: InstrumentId) -> None:
        if type(instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId")
        retry = self._instrument_retry
        if retry is None:
            raise WatchlistPageError("WATCHLIST_RETRY_UNAVAILABLE")
        _gateway_call(
            "WATCHLIST_RETRY_FAILED",
            lambda: retry(instrument),
        )


def _append_symbols(current: str, additions: Sequence[str]) -> str:
    values = [item for item in _SEPARATOR.split(current.strip()) if item]
    seen = set(values)
    for code in additions:
        if code not in seen:
            values.append(code)
            seen.add(code)
    return "\n".join(values)


def _market_symbols(market: object, codes_text: object) -> tuple[str, ...]:
    if market not in {Exchange.SSE.value, Exchange.SZSE.value}:
        raise ValueError("请选择上证或深证市场")
    if type(codes_text) is not str:
        raise ValueError("请输入六位标的代码")
    codes = tuple(item for item in _SEPARATOR.split(codes_text.strip()) if item)
    if not codes:
        raise ValueError("请输入六位标的代码")
    if any(_PLAIN_CODE.fullmatch(code) is None for code in codes):
        raise ValueError("代码应为六位数字，不需要填写 SSE 或 SZSE")
    return tuple(f"{market}.{code}" for code in codes)


def render_watchlists_page(model: WatchlistPageModel | None) -> None:
    if model is None:
        ui.label("标的池服务未配置。")
        return
    try:
        state = model.state()
    except Exception:
        ui.label("标的池读取失败，请查看本地日志。").classes("text-red-700")
        return
    selected_codes = [] if state.entry is None else list(map(str, state.entry.instruments))
    data_ranges = {item.instrument: item for item in state.data_ranges}
    with ui.card().classes("w-full border border-slate-200 shadow-none"):
        ui.label("关注标的").classes("text-lg font-semibold")
        ui.label("项目只维护这一个标的池。保存后，行情数据页会按这里的标的增量获取数据。").classes(
            "text-sm text-slate-600"
        )
        ui.label(
            "快捷预设优先追加对应指数，以获得更长历史；需要交易价格时仍可手工输入 ETF。"
        ).classes("text-xs text-slate-500")

        @ui.refreshable
        def instrument_groups() -> None:
            ui.label("当前标的（按市场）").classes("text-sm font-semibold text-slate-700")
            if not selected_codes:
                ui.label("暂无标的，请从下方添加或使用快捷预设。").classes("text-xs text-slate-400")
                return
            for market, market_name in (
                (Exchange.SSE.value, "上证"),
                (Exchange.SZSE.value, "深证"),
            ):
                market_codes = tuple(
                    code for code in selected_codes if code.startswith(f"{market}.")
                )
                if not market_codes:
                    continue
                with ui.column().classes(
                    "w-full gap-1 rounded-lg border border-slate-200 bg-slate-50 p-3"
                ):
                    ui.label(f"{market_name} · {len(market_codes)} 个").classes(
                        "text-sm font-medium text-slate-700"
                    )
                    for canonical_code in market_codes:
                        instrument = InstrumentId.parse(canonical_code)
                        name = common_instrument_name(instrument) or "名称待同步"
                        data_range = data_ranges.get(instrument)
                        range_label = (
                            "暂无行情数据"
                            if data_range is None
                            else (
                                f"数据：{data_range.first_day.isoformat()} 至 "
                                f"{data_range.last_day.isoformat()}"
                            )
                        )

                        def remove(code: str = canonical_code) -> None:
                            selected_codes.remove(code)
                            instrument_groups.refresh()

                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label(instrument.code).classes("text-sm font-mono text-slate-700")
                            ui.label(name).classes("text-sm text-slate-400")
                            ui.label(range_label).classes("text-xs text-slate-400")
                            ui.space()
                            if data_range is None and model.supports_instrument_retry:

                                def retry(item: InstrumentId = instrument) -> None:
                                    try:
                                        model.retry_instrument(item)
                                    except Exception:
                                        ui.notify(
                                            "该标的获取任务未启动，请检查当前任务状态。",
                                            type="negative",
                                        )
                                    else:
                                        ui.notify(
                                            "已启动该标的获取任务，可到行情数据页查看进度。",
                                            type="positive",
                                        )

                                ui.button("获取数据", on_click=retry).props(
                                    f"outline dense aria-label=获取{instrument.code}行情"
                                )
                            ui.button("移除", on_click=remove).props(
                                f"flat dense color=grey aria-label=移除{instrument.code}"
                            )

        instrument_groups()

        ui.separator().classes("my-1")
        ui.label("快捷追加").classes("text-sm font-semibold text-slate-700")
        with ui.row().classes("w-full gap-2 flex-wrap"):
            for label, additions in QUICK_SELECTIONS:

                def append_selected(codes: tuple[str, ...] = additions) -> None:
                    selected_codes[:] = _append_symbols(
                        "\n".join(selected_codes), codes
                    ).splitlines()
                    instrument_groups.refresh()

                ui.button(f"追加{label}", on_click=append_selected).props(
                    f"outline dense aria-label=追加{label}"
                )

        ui.label("手工添加").classes("text-sm font-semibold text-slate-700")
        with ui.row().classes("w-full gap-3 items-end flex-wrap"):
            market_select = (
                ui.select(
                    {
                        Exchange.SSE.value: "上证",
                        Exchange.SZSE.value: "深证",
                    },
                    label="股票市场",
                    value=Exchange.SSE.value,
                )
                .props("aria-label=股票市场")
                .classes("w-40")
            )
            code_input = (
                ui.input(
                    "标的代码",
                    placeholder="例如：000300",
                )
                .props("aria-label=标的代码")
                .classes("w-64")
            )

            def add_manual() -> None:
                try:
                    additions = _market_symbols(market_select.value, code_input.value)
                except ValueError as error:
                    ui.notify(str(error), type="warning")
                    return
                selected_codes[:] = _append_symbols(
                    "\n".join(selected_codes), additions
                ).splitlines()
                code_input.value = ""
                instrument_groups.refresh()

            ui.button("添加标的", on_click=add_manual, icon="add").props("aria-label=添加标的")
        ui.label("可一次输入多个六位代码，用空格、逗号或换行分隔。").classes(
            "text-xs text-slate-400"
        )
        feedback = ui.column().classes("w-full gap-1")

        def save() -> None:
            result = model.save(WatchlistFormModel("关注标的", "\n".join(selected_codes)))
            feedback.clear()
            with feedback:
                if result.errors:
                    for message in result.errors.values():
                        ui.label(message).classes("text-sm text-red-700")
                else:
                    ui.label("关注标的已保存。行情数据页将使用这个标的池。").classes(
                        "text-sm text-emerald-700"
                    )
            instrument_groups.refresh()

        ui.button("保存关注标的", on_click=save, icon="save").props("aria-label=保存关注标的")
