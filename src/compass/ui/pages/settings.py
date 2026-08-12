from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from ipaddress import ip_address
from pathlib import Path
from typing import Protocol, TypeVar, cast

from nicegui import ui

from compass.data.network_timeout import (
    DEFAULT_MARKET_TIMEOUT_SECONDS,
    MAXIMUM_MARKET_TIMEOUT_SECONDS,
    MINIMUM_MARKET_TIMEOUT_SECONDS,
    validate_market_timeout_seconds,
)
from compass.services.safe_display import safe_display_text, safe_identifier, stable_code


T = TypeVar("T")
_MISSING = object()
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
AUTOMATIC_SYNC_INTERVALS = (30, 60, 240, 720, 1440)


class ConnectionTestStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ConnectionTestResult:
    target: str
    display_name: str
    status: ConnectionTestStatus
    elapsed_ms: int | None = None
    detail_code: str | None = None

    def __post_init__(self) -> None:
        safe_identifier(self.target, label="connection test target")
        safe_display_text(self.display_name, label="connection test display name")
        if type(self.status) is not ConnectionTestStatus:
            raise TypeError("connection test status must be exact")
        if self.elapsed_ms is not None and (
            type(self.elapsed_ms) is not int or self.elapsed_ms < 0
        ):
            raise ValueError("connection test elapsed time is invalid")
        if self.detail_code is not None:
            stable_code(self.detail_code, label="connection test detail code")


class MarketProxyMode(str, Enum):
    NONE = "none"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class MarketProxySetting:
    mode: MarketProxyMode
    host: str | None = None
    port: int | None = None

    def __post_init__(self) -> None:
        if type(self.mode) is not MarketProxyMode:
            raise TypeError("proxy mode must be an exact MarketProxyMode")
        if self.mode is not MarketProxyMode.CUSTOM:
            if self.host is not None or self.port is not None:
                raise ValueError("non-custom proxy cannot contain an address")
            return
        if type(self.host) is not str:
            raise TypeError("custom proxy host must be an IP address")
        try:
            parsed_host = ip_address(self.host.strip())
        except ValueError:
            raise ValueError("PROXY_IP_INVALID") from None
        if parsed_host.is_unspecified or parsed_host.is_multicast:
            raise ValueError("PROXY_IP_INVALID")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("PROXY_PORT_INVALID")
        object.__setattr__(self, "host", str(parsed_host))


class SettingsPageError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = stable_code(code, label="settings page error code")
        super().__init__(self.code)


def _boundary_call(code: str, operation: Callable[[], T]) -> T:
    failed = False
    result: object = _MISSING
    try:
        result = operation()
    except Exception:
        failed = True
    if failed:
        raise SettingsPageError(code)
    return cast(T, result)


@dataclass(frozen=True, slots=True)
class ProviderSetting:
    provider: str
    display_name: str
    available: bool
    priority: int
    credential_present: bool | None

    def __post_init__(self) -> None:
        safe_identifier(self.provider, label="provider id")
        safe_display_text(self.display_name, label="provider display name")
        if type(self.available) is not bool:
            raise TypeError("provider availability must be an exact bool")
        if type(self.priority) is not int or self.priority < 0:
            raise ValueError("provider priority must be a non-negative exact integer")
        if self.credential_present is not None and type(self.credential_present) is not bool:
            raise TypeError("credential presence must be an exact bool or None")


@dataclass(frozen=True, slots=True)
class FeeProfileSetting:
    profile_id: str
    confirmed: bool


@dataclass(frozen=True, slots=True)
class RiskTemplateSetting:
    template_id: str
    display_name: str
    active: bool


@dataclass(frozen=True, slots=True)
class SettingsSnapshot:
    providers: tuple[ProviderSetting, ...]
    log_level: str
    fee_profile: FeeProfileSetting | None = None
    risk_templates: tuple[RiskTemplateSetting, ...] = ()
    market_request_timeout_seconds: int = DEFAULT_MARKET_TIMEOUT_SECONDS
    market_proxy: MarketProxySetting = MarketProxySetting(MarketProxyMode.SYSTEM)
    automatic_sync_on_startup: bool = False
    automatic_sync_interval_minutes: int | None = None
    automatic_sync_after_close: bool = False

    def __post_init__(self) -> None:
        providers = tuple(self.providers)
        if any(type(item) is not ProviderSetting for item in providers):
            raise TypeError("providers must contain exact ProviderSetting values")
        provider_ids = tuple(item.provider for item in providers)
        priorities = tuple(item.priority for item in providers)
        if len(set(provider_ids)) != len(provider_ids):
            raise ValueError("providers must be unique")
        if priorities != tuple(range(len(providers))):
            raise ValueError("providers must be ordered by contiguous priority")
        if type(self.log_level) is not str or self.log_level not in LOG_LEVELS:
            raise ValueError("LOG_LEVEL_INVALID")
        validate_market_timeout_seconds(self.market_request_timeout_seconds)
        if type(self.market_proxy) is not MarketProxySetting:
            raise TypeError("market proxy must be an exact MarketProxySetting")
        if type(self.automatic_sync_on_startup) is not bool:
            raise TypeError("automatic startup sync must be an exact bool")
        if type(self.automatic_sync_after_close) is not bool:
            raise TypeError("automatic close sync must be an exact bool")
        if (
            self.automatic_sync_interval_minutes is not None
            and self.automatic_sync_interval_minutes not in AUTOMATIC_SYNC_INTERVALS
        ):
            raise ValueError("AUTOMATIC_SYNC_INTERVAL_INVALID")
        templates = tuple(self.risk_templates)
        object.__setattr__(self, "providers", providers)
        object.__setattr__(self, "risk_templates", templates)


class SettingsGateway(Protocol):
    def state(self) -> SettingsSnapshot: ...
    def set_market_request_timeout(self, seconds: int) -> None: ...
    def set_market_proxy(self, setting: MarketProxySetting) -> None: ...
    def test_connections(self) -> Sequence[ConnectionTestResult]: ...
    def set_automatic_sync(
        self,
        on_startup: bool,
        interval_minutes: int | None,
        after_close: bool,
    ) -> None: ...


class SettingsPageModel:
    def __init__(
        self,
        gateway: SettingsGateway,
        runtime_data_dir: Path | None = None,
    ) -> None:
        self._gateway = gateway
        if runtime_data_dir is not None and not runtime_data_dir.is_absolute():
            raise ValueError("runtime data directory must be absolute")
        self._runtime_data_dir = runtime_data_dir

    @property
    def runtime_data_dir(self) -> Path | None:
        return self._runtime_data_dir

    def state(self) -> SettingsSnapshot:
        return _boundary_call("SETTINGS_STATE_UNAVAILABLE", self._state)

    def _state(self) -> SettingsSnapshot:
        state = self._gateway.state()
        if type(state) is not SettingsSnapshot:
            raise TypeError("settings gateway must return an exact SettingsSnapshot")
        return state

    def set_market_request_timeout(self, seconds: int) -> None:
        checked = validate_market_timeout_seconds(seconds)
        _boundary_call(
            "SETTINGS_MARKET_REQUEST_TIMEOUT_FAILED",
            lambda: self._gateway.set_market_request_timeout(checked),
        )

    def set_market_proxy(
        self,
        mode: MarketProxyMode,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        setting = MarketProxySetting(mode, host, port)
        _boundary_call(
            "SETTINGS_MARKET_PROXY_FAILED",
            lambda: self._gateway.set_market_proxy(setting),
        )

    def test_connections(self) -> tuple[ConnectionTestResult, ...]:
        results = tuple(
            _boundary_call(
                "SETTINGS_CONNECTION_TEST_FAILED",
                self._gateway.test_connections,
            )
        )
        if any(type(item) is not ConnectionTestResult for item in results):
            raise TypeError("connection test results are invalid")
        return results

    def set_automatic_sync(
        self,
        on_startup: bool,
        interval_minutes: int | None,
        after_close: bool,
    ) -> None:
        if type(on_startup) is not bool or type(after_close) is not bool:
            raise TypeError("automatic sync flags must be exact bools")
        if interval_minutes is not None and interval_minutes not in AUTOMATIC_SYNC_INTERVALS:
            raise ValueError("AUTOMATIC_SYNC_INTERVAL_INVALID")
        _boundary_call(
            "SETTINGS_AUTOMATIC_SYNC_FAILED",
            lambda: self._gateway.set_automatic_sync(
                on_startup,
                interval_minutes,
                after_close,
            ),
        )

def render_settings_page(model: SettingsPageModel | None) -> None:
    if model is None:
        ui.label("系统设置服务未配置；当前不会读取环境、密钥或本地私有路径。")
        return
    try:
        state = model.state()
    except Exception:
        ui.label("系统设置读取失败，请查看本地脱敏日志。").classes("text-red-700")
        return
    feedback = ui.label("")
    connection_results: tuple[ConnectionTestResult, ...] = ()
    if model.runtime_data_dir is not None:
        with ui.card().classes("w-full border border-slate-200 shadow-none bg-slate-50"):
            ui.label("本地运行数据目录").classes("font-semibold")
            ui.label(str(model.runtime_data_dir)).classes(
                "text-sm font-mono text-slate-700 break-all"
            )
            ui.label(
                "数据库、行情、账户、策略实验和日志保存在此处，不属于 Git 仓库。"
            ).classes("text-xs text-slate-500")
    ui.label("自动获取行情").classes("font-semibold")
    ui.label(
        "自动任务使用推荐主源并执行增量同步；若已有行情任务正在运行，本次触发会跳过。"
    ).classes("text-xs text-slate-500")
    startup_sync = ui.checkbox(
        "应用启动后自动创建行情同步任务",
        value=state.automatic_sync_on_startup,
    )
    after_close_sync = ui.checkbox(
        "每个交易日收盘后检查并同步一次（推荐）",
        value=state.automatic_sync_after_close,
    )
    ui.label(
        "收盘约 15 分钟后检查；应用启动时也会检查最近已完成交易日。仅当关注标的"
        "尚未覆盖该交易日时才创建任务，失败后会继续检查。"
    ).classes("text-xs text-slate-500")
    interval_sync = ui.select(
        {
            0: "不定期同步",
            30: "每 30 分钟",
            60: "每 1 小时",
            240: "每 4 小时",
            720: "每 12 小时",
            1440: "每 24 小时",
        },
        value=state.automatic_sync_interval_minutes or 0,
        label="定期同步间隔",
    ).props("outlined dense options-dense").classes("w-64")

    def save_automatic_sync() -> None:
        try:
            raw = interval_sync.value
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError("AUTOMATIC_SYNC_INTERVAL_INVALID")
            interval = int(raw)
            if interval != raw:
                raise ValueError("AUTOMATIC_SYNC_INTERVAL_INVALID")
            model.set_automatic_sync(
                bool(startup_sync.value),
                None if interval == 0 else interval,
                bool(after_close_sync.value),
            )
            feedback.set_text("自动获取行情设置已保存；定期设置将在下一次调度检查生效。")
        except Exception as error:
            feedback.set_text(
                "自动任务设置失败："
                f"{getattr(error, 'code', str(error) or 'AUTOMATIC_SYNC_INVALID')}"
            )

    ui.button("保存自动任务", icon="schedule", on_click=save_automatic_sync)
    ui.separator().classes("my-2")
    ui.label("行情数据源使用固定回退顺序：腾讯证券 → 东方财富 → BaoStock。").classes(
        "text-sm text-slate-600"
    )
    request_timeout = (
        ui.number(
            "行情请求超时时间（秒）",
            value=state.market_request_timeout_seconds,
            min=MINIMUM_MARKET_TIMEOUT_SECONDS,
            max=MAXIMUM_MARKET_TIMEOUT_SECONDS,
            step=1,
        )
        .props("aria-label=行情请求超时时间")
        .classes("w-64")
    )
    ui.label(
        "默认 5 秒；适合快速发现网络或代理异常。网络较慢、跨境代理不稳定时可调到 10～15 秒。"
    ).classes("text-xs text-slate-500")

    def apply_request_timeout() -> None:
        try:
            value = request_timeout.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("MARKET_REQUEST_TIMEOUT_INVALID")
            checked = int(value)
            if checked != value:
                raise ValueError("MARKET_REQUEST_TIMEOUT_INVALID")
            model.set_market_request_timeout(checked)
            feedback.set_text(f"行情请求超时时间已更新为 {checked} 秒")
        except Exception as error:
            feedback.set_text(
                "超时时间更新失败："
                f"{getattr(error, 'code', str(error) or 'MARKET_REQUEST_TIMEOUT_INVALID')}"
            )

    ui.button("应用超时时间", on_click=apply_request_timeout)
    ui.separator().classes("my-2")
    ui.label("行情网络代理").classes("font-semibold")
    proxy_mode = (
        ui.select(
            {
                MarketProxyMode.SYSTEM.value: "使用系统代理",
                MarketProxyMode.NONE.value: "不使用代理（直连）",
                MarketProxyMode.CUSTOM.value: "自定义代理",
            },
            label="代理模式",
            value=state.market_proxy.mode.value,
        )
        .props("aria-label=行情代理模式")
        .classes("w-64")
    )
    proxy_host = (
        ui.input(
            "代理 IP",
            value=state.market_proxy.host or "127.0.0.1",
        )
        .props("aria-label=行情代理IP")
        .classes("w-64")
    )
    proxy_port = (
        ui.number(
            "代理端口",
            value=state.market_proxy.port or 7897,
            min=1,
            max=65535,
            step=1,
        )
        .props("aria-label=行情代理端口")
        .classes("w-64")
    )

    def update_proxy_fields() -> None:
        enabled = proxy_mode.value == MarketProxyMode.CUSTOM.value
        if enabled:
            proxy_host.enable()
            proxy_port.enable()
        else:
            proxy_host.disable()
            proxy_port.disable()

    proxy_mode.on_value_change(lambda _: update_proxy_fields())
    update_proxy_fields()
    ui.label(
        "设置会立即应用于之后发起的东方财富、腾讯证券等 HTTP 行情请求；"
        "BaoStock 使用独立 TCP 连接，不经过 HTTP 代理。"
    ).classes("text-xs text-slate-500")

    def apply_market_proxy() -> None:
        try:
            mode = MarketProxyMode(str(proxy_mode.value))
            if mode is MarketProxyMode.CUSTOM:
                raw_port = proxy_port.value
                if isinstance(raw_port, bool) or not isinstance(raw_port, (int, float)):
                    raise ValueError("PROXY_PORT_INVALID")
                port = int(raw_port)
                if port != raw_port:
                    raise ValueError("PROXY_PORT_INVALID")
                model.set_market_proxy(mode, str(proxy_host.value).strip(), port)
                feedback.set_text(f"行情代理已应用：http://{str(proxy_host.value).strip()}:{port}")
            else:
                model.set_market_proxy(mode)
                label = "系统代理" if mode is MarketProxyMode.SYSTEM else "直连"
                feedback.set_text(f"行情网络已切换为：{label}")
        except Exception as error:
            feedback.set_text(
                f"代理设置失败：{getattr(error, 'code', str(error) or 'PROXY_INVALID')}"
            )

    ui.button("应用代理设置", on_click=apply_market_proxy)
    ui.separator().classes("my-2")
    ui.label("连接测试").classes("font-semibold")
    ui.label(
        "依次检查当前代理配置、东方财富、腾讯证券和 BaoStock 行情接口；"
        "测试不会写入行情数据。"
    ).classes("text-xs text-slate-500")

    @ui.refreshable
    def connection_result_rows() -> None:
        if not connection_results:
            ui.label("尚未测试。").classes("text-xs text-slate-400")
            return
        for result in connection_results:
            status_text = {
                ConnectionTestStatus.SUCCEEDED: "成功",
                ConnectionTestStatus.FAILED: "失败",
                ConnectionTestStatus.SKIPPED: "跳过",
            }[result.status]
            status_class = {
                ConnectionTestStatus.SUCCEEDED: "text-emerald-700",
                ConnectionTestStatus.FAILED: "text-red-700",
                ConnectionTestStatus.SKIPPED: "text-slate-500",
            }[result.status]
            elapsed = "" if result.elapsed_ms is None else f" · {result.elapsed_ms} ms"
            detail = "" if result.detail_code is None else f" · {result.detail_code}"
            ui.label(f"{result.display_name}：{status_text}{elapsed}{detail}").classes(
                f"text-sm {status_class}"
            )

    connection_result_rows()

    async def test_connections() -> None:
        nonlocal connection_results
        connection_button.disable()
        feedback.set_text("正在测试代理和行情数据源连接……")
        try:
            connection_results = await asyncio.to_thread(model.test_connections)
        except Exception as error:
            feedback.set_text(
                f"连接测试失败：{getattr(error, 'code', 'SETTINGS_CONNECTION_TEST_FAILED')}"
            )
        else:
            feedback.set_text("连接测试完成。")
            connection_result_rows.refresh()
        finally:
            connection_button.enable()

    connection_button = ui.button(
        "测试行情连接",
        on_click=test_connections,
        icon="network_check",
    ).props("aria-label=测试行情连接")
