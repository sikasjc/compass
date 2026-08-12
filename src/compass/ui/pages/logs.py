from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar, cast

from nicegui import ui

from compass.services.diagnostic_log import DiagnosticLogEntry
from compass.ui.pages.settings import LOG_LEVELS, SettingsSnapshot


T = TypeVar("T")
_MISSING = object()


class LogsPageError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _boundary_call(code: str, operation: Callable[[], T]) -> T:
    failed = False
    result: object = _MISSING
    try:
        result = operation()
    except Exception:
        failed = True
    if failed:
        raise LogsPageError(code)
    return cast(T, result)


class LogsGateway(Protocol):
    def state(self) -> SettingsSnapshot: ...
    def set_log_level(self, level: str) -> None: ...
    def read_logs(
        self,
        limit: int,
        level: str | None,
        query: str,
    ) -> Sequence[DiagnosticLogEntry]: ...


class LogsPageModel:
    def __init__(self, gateway: LogsGateway) -> None:
        self._gateway = gateway

    def log_level(self) -> str:
        state = _boundary_call("LOG_SETTINGS_UNAVAILABLE", self._gateway.state)
        if type(state) is not SettingsSnapshot:
            raise TypeError("logs gateway must return an exact SettingsSnapshot")
        return state.log_level

    def set_log_level(self, level: str) -> None:
        if type(level) is not str or level not in LOG_LEVELS:
            raise ValueError("LOG_LEVEL_INVALID")
        _boundary_call("LOG_LEVEL_UPDATE_FAILED", lambda: self._gateway.set_log_level(level))

    def logs(
        self,
        *,
        limit: int = 300,
        level: str | None = None,
        query: str = "",
    ) -> tuple[DiagnosticLogEntry, ...]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("LOG_LIMIT_INVALID")
        if level is not None and level not in LOG_LEVELS:
            raise ValueError("LOG_LEVEL_INVALID")
        if type(query) is not str or len(query) > 100:
            raise ValueError("LOG_QUERY_INVALID")
        entries = tuple(
            _boundary_call(
                "LOG_READ_FAILED",
                lambda: self._gateway.read_logs(limit, level, query.strip()),
            )
        )
        if any(type(item) is not DiagnosticLogEntry for item in entries):
            raise TypeError("logs gateway returned invalid log entries")
        return entries


def render_logs_page(model: LogsPageModel | None) -> None:
    if model is None:
        ui.label("日志服务尚未配置。").classes("text-sm text-slate-600")
        return
    try:
        current_level = model.log_level()
    except Exception:
        ui.label("日志设置读取失败。").classes("text-red-700")
        return

    ui.label(
        "查看最近 300 条脱敏日志。响应正文仅保留最多 600 个字符的摘要；"
        "不会显示 Token、Cookie、代理密码或 Authorization。"
    ).classes("text-sm text-slate-600")
    feedback = ui.label("")
    with ui.card().classes("w-full border border-slate-200 shadow-none"):
        ui.label("日志记录设置").classes("font-semibold")
        with ui.row().classes("w-full gap-3 items-end flex-wrap"):
            log_level = ui.select(
                list(LOG_LEVELS),
                label="记录级别",
                value=current_level,
            ).classes("w-44")

            def apply_log_level() -> None:
                try:
                    model.set_log_level(str(log_level.value))
                    feedback.set_text(f"日志记录级别已应用：{log_level.value}")
                except Exception as error:
                    feedback.set_text(
                        "日志级别应用失败："
                        f"{getattr(error, 'code', 'LOG_LEVEL_UPDATE_FAILED')}"
                    )

            ui.button("应用记录级别", on_click=apply_log_level)

    with ui.card().classes("w-full border border-slate-200 shadow-none"):
        with ui.row().classes("w-full items-end gap-3 flex-wrap"):
            level_filter = (
                ui.select(
                    {"": "全部级别", **{item: item for item in LOG_LEVELS}},
                    label="筛选级别",
                    value="",
                )
                .props("aria-label=日志筛选级别")
                .classes("w-44")
            )
            query_filter = (
                ui.input("搜索", placeholder="如 510300、东方财富、authentication")
                .props("aria-label=日志搜索 maxlength=100 clearable")
                .classes("grow min-w-64")
            )

            refresh_button = ui.button(icon="refresh").props(
                "outline round aria-label=刷新日志"
            )

        @ui.refreshable
        def log_rows() -> None:
            selected_level = str(level_filter.value or "") or None
            query = str(query_filter.value or "")
            try:
                entries = model.logs(level=selected_level, query=query)
            except Exception as error:
                ui.label(
                    f"日志读取失败：{getattr(error, 'code', 'LOG_READ_FAILED')}"
                ).classes("text-sm text-red-700")
                return
            if not entries:
                ui.label(
                    "当前筛选条件下没有日志。发起一次连接测试或行情同步后再刷新。"
                ).classes("text-sm text-slate-500")
                return
            with ui.scroll_area().classes(
                "w-full h-[650px] rounded border border-slate-200 bg-slate-950 p-2"
            ):
                for entry in entries:
                    color = {
                        "DEBUG": "text-slate-400",
                        "INFO": "text-emerald-300",
                        "WARNING": "text-amber-300",
                        "ERROR": "text-red-300",
                        "CRITICAL": "text-red-400",
                    }[entry.level]
                    with ui.column().classes(
                        "w-full gap-0 border-b border-slate-800 py-2"
                    ):
                        ui.label(
                            f"{entry.occurred_at.isoformat(timespec='milliseconds')} "
                            f"{entry.level} · {entry.category}"
                        ).classes(f"text-xs font-mono {color}")
                        ui.label(entry.message).classes(
                            "text-xs font-mono text-slate-100 whitespace-pre-wrap break-all"
                        )

        refresh_button.on_click(log_rows.refresh)
        level_filter.on_value_change(lambda _: log_rows.refresh())
        query_filter.on("keydown.enter", lambda _: log_rows.refresh())
        log_rows()
