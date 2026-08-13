from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import sys
from typing import Protocol, cast

from nicegui import app, ui

from compass.config import Settings
from compass.services.local_application import build_local_application
from compass.services.task_manager import TaskManager
from compass.ui.layout import NavigationItem, page_shell
from compass.ui.pages.account_overview import (
    AccountOverviewPageModel,
    render_account_overview_page,
)
from compass.ui.pages.data import DataPageModel, render_data_page
from compass.ui.pages.logs import LogsPageModel, render_logs_page
from compass.ui.pages.rule_editor import (
    render_rule_editor_page,
    render_rule_preview_page,
    render_rule_release_page,
    render_strategy_library_page,
    render_strategy_templates_page,
)
from compass.ui.pages.home import render_start_page
from compass.ui.pages.settings import SettingsPageModel, render_settings_page
from compass.ui.pages.signals import SignalPageModel, render_signals_page
from compass.ui.pages.strategy_lab import StrategyLabPageModel, render_strategy_lab_page
from compass.ui.pages.strategies import StrategyPageModel
from compass.ui.pages.watchlists import WatchlistPageModel, render_watchlists_page


LOCAL_HOST = "127.0.0.1"
NAV_ITEMS = (
    NavigationItem("开始", "/", "home"),
    NavigationItem("今日信号", "/signals", "recommend"),
    NavigationItem("账户", "/account", "account_balance_wallet"),
    NavigationItem("策略回测", "/backtests", "query_stats"),
    NavigationItem("策略实验室", "/strategies", "science"),
    NavigationItem("行情数据", "/data", "database"),
    NavigationItem("标的池", "/watchlists", "playlist_add_check"),
    NavigationItem("设置", "/settings", "settings"),
    NavigationItem("日志", "/logs", "article"),
)
ROUTES = tuple(item.route for item in NAV_ITEMS) + (
    "/strategies/editor",
    "/strategies/preview",
    "/strategies/release",
    "/strategies/templates",
)


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    host: str = LOCAL_HOST
    port: int = 8080
    show: bool = False
    reload: bool = False

    def __post_init__(self) -> None:
        if self.host != LOCAL_HOST:
            raise ValueError("the local UI may bind only to 127.0.0.1")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be an exact TCP port")
        if type(self.show) is not bool or type(self.reload) is not bool:
            raise TypeError("show and reload must be exact bool values")


def _cli_port(value: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("端口必须是 1 到 65535 之间的整数") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口必须是 1 到 65535 之间的整数")
    return port


def parse_run_configuration(arguments: Sequence[str]) -> RunConfiguration:
    if isinstance(arguments, (str, bytes)) or any(type(item) is not str for item in arguments):
        raise TypeError("application arguments must be exact strings")
    parser = argparse.ArgumentParser(
        prog="python -m compass.ui.app",
        allow_abbrev=False,
    )
    parser.add_argument("--port", type=_cli_port, default=8080)
    parsed = parser.parse_args(tuple(arguments))
    return RunConfiguration(port=parsed.port)


@dataclass(frozen=True, slots=True)
class AppViewModels:
    watchlists: WatchlistPageModel | None = None
    data: DataPageModel | None = None
    strategies: StrategyPageModel | None = None
    backtests: StrategyLabPageModel | None = None
    account: AccountOverviewPageModel | None = None
    signals: SignalPageModel | None = None
    settings: SettingsPageModel | None = None
    logs: LogsPageModel | None = None
    task_manager: TaskManager | None = None
    owns_task_manager: bool = False

    def __post_init__(self) -> None:
        if type(self.owns_task_manager) is not bool:
            raise TypeError("owns_task_manager must be an exact bool")
        if self.owns_task_manager and self.task_manager is None:
            raise ValueError("an owned task manager must be provided")


PageHandler = Callable[[], None]


class PageRegistrar(Protocol):
    def __call__(self, route: str) -> Callable[[PageHandler], PageHandler]: ...


def register_pages(
    models: AppViewModels,
    *,
    registrar: PageRegistrar | None = None,
) -> None:
    register = registrar or cast(PageRegistrar, ui.page)

    def home() -> None:
        with page_shell("开始", "快速进入常用功能，不需要记住页面路径", NAV_ITEMS):
            render_start_page()

    def watchlists() -> None:
        with page_shell("标的池", "选择并维护唯一的关注标的池", NAV_ITEMS):
            render_watchlists_page(models.watchlists)

    def data() -> None:
        with page_shell("行情数据", "查看来源、质量、缓存与后台同步状态", NAV_ITEMS):
            render_data_page(models.data)

    def strategy_lab() -> None:
        with page_shell("策略实验室", "创建、解释并管理可复用的策略定义", NAV_ITEMS):
            render_strategy_library_page(models.strategies)

    def rule_editor() -> None:
        with page_shell("规则编辑器", "使用规则与变量创建安全、可复现的策略草稿", NAV_ITEMS):
            render_rule_editor_page(models.strategies)

    def rule_preview() -> None:
        with page_shell("信号预览", "检查规则何时命中以及目标仓位如何变化", NAV_ITEMS):
            render_rule_preview_page(models.strategies)

    def rule_release() -> None:
        with page_shell("验证与发布", "检查执行语义并发布不可变策略版本", NAV_ITEMS):
            render_rule_release_page(models.strategies)

    def strategy_templates() -> None:
        with page_shell("内置模板与参数调优", "创建经典策略或运行参数实验", NAV_ITEMS):
            render_strategy_templates_page(models.strategies)

    def backtests() -> None:
        with page_shell("策略回测", "配置组合与账户参数，运行回测并查看结果", NAV_ITEMS):
            render_strategy_lab_page(models.backtests)

    def settings() -> None:
        with page_shell("设置", "管理网络与行情请求参数", NAV_ITEMS):
            render_settings_page(models.settings)

    def account() -> None:
        with page_shell("账户", "查看持仓、资金变化、建议采用情况与事后影响", NAV_ITEMS):
            render_account_overview_page(models.account)

    def signals() -> None:
        with page_shell(
            "今日信号",
            "连接账户方案、共享持仓、最新收盘信号与执行记录",
            NAV_ITEMS,
        ):
            render_signals_page(models.signals)

    def logs() -> None:
        with page_shell("日志", "查看应用与行情请求的本地脱敏日志", NAV_ITEMS):
            render_logs_page(models.logs)

    handlers = (
        home,
        signals,
        account,
        backtests,
        strategy_lab,
        data,
        watchlists,
        settings,
        logs,
        rule_editor,
        rule_preview,
        rule_release,
        strategy_templates,
    )
    for route, handler in zip(ROUTES, handlers, strict=True):
        register(route)(handler)


def create_app(
    settings: Settings,
    models: AppViewModels | None = None,
    *,
    registrar: PageRegistrar | None = None,
) -> None:
    if type(settings) is not Settings:
        raise TypeError("settings must be an exact Settings value")
    local_application = None
    if models is None:
        local_application = build_local_application(settings)
        view_models = local_application.models
    else:
        view_models = models
    register_pages(view_models, registrar=registrar)
    if local_application is not None:
        app.on_startup(local_application.start)
        app.on_shutdown(local_application.shutdown)
    elif view_models.owns_task_manager:
        assert view_models.task_manager is not None
        app.on_shutdown(view_models.task_manager.shutdown)


Runner = Callable[..., object]
AppFactory = Callable[[Settings], None]


def run(
    settings: Settings | None = None,
    *,
    configuration: RunConfiguration = RunConfiguration(),
    app_factory: AppFactory = create_app,
    runner: Runner = ui.run,
) -> None:
    active_settings = settings or Settings.from_env()
    app_factory(active_settings)
    runner(
        host=configuration.host,
        port=configuration.port,
        show=configuration.show,
        reload=configuration.reload,
    )


def main(arguments: Sequence[str] | None = None) -> None:
    selected = tuple(sys.argv[1:]) if arguments is None else arguments
    run(configuration=parse_run_configuration(selected))


if __name__ in {"__main__", "__mp_main__"}:
    main()
