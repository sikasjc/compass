from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from datetime import date

from nicegui import ui
from compass.ui.pages.watchlists import WatchlistPageModel, WatchlistPageState
from compass.ui.pages.signals import SignalPageModel
from compass.services.task_manager import TaskManager, TaskStatus
from compass.ui.task_status import task_status_label
from compass.ui.navigation import account_url


@dataclass(frozen=True, slots=True)
class StartPageEntry:
    title: str
    description: str
    route: str
    icon: str
    accent: str = "text-slate-700"


_PRIMARY_ENTRIES = (
    StartPageEntry(
        "今日信号",
        "根据账户持仓和已启用策略生成最新调仓建议。",
        "/signals",
        "recommend",
        "text-emerald-700",
    ),
    StartPageEntry(
        "账户",
        "维护账户、共享持仓，查看资金和持仓变化。",
        "/account",
        "account_balance_wallet",
        "text-blue-700",
    ),
    StartPageEntry(
        "策略回测",
        "组合多个策略和标的，运行回测并比较基准。",
        "/backtests",
        "query_stats",
        "text-indigo-700",
    ),
    StartPageEntry(
        "策略实验室",
        "创建策略模板，运行参数调优实验并发布新版本。",
        "/strategies",
        "science",
        "text-violet-700",
    ),
)

_DATA_ENTRIES = (
    StartPageEntry(
        "行情数据",
        "同步、检查和清理本地历史行情。",
        "/data",
        "database",
    ),
    StartPageEntry(
        "标的池",
        "维护需要关注、回测和生成信号的标的。",
        "/watchlists",
        "playlist_add_check",
    ),
    StartPageEntry(
        "设置",
        "配置数据源、代理、超时和自动同步任务。",
        "/settings",
        "settings",
    ),
    StartPageEntry(
        "日志",
        "排查行情请求、任务和应用运行问题。",
        "/logs",
        "article",
    ),
)


def _market_readiness(
    watchlist: WatchlistPageState,
    expected: date | None,
) -> tuple[date | None, bool]:
    """Return the oldest current endpoint and whether the whole pool is current."""

    pool = watchlist.entry
    needed = set(pool.instruments) if pool else set()
    days_by_instrument = {
        item.instrument: item.last_day for item in watchlist.data_ranges
    }
    available_days = tuple(
        days_by_instrument[instrument]
        for instrument in needed
        if instrument in days_by_instrument
    )
    data_day = min(available_days, default=None)
    ready = bool(needed) and len(available_days) == len(needed) and (
        expected is not None and data_day is not None and data_day >= expected
    )
    return data_day, ready


def _action_link(label: str, route: str, *, primary: bool = False) -> None:
    classes = (
        "no-underline inline-flex items-center rounded px-4 py-2 "
        + (
            "bg-emerald-700 text-white hover:bg-emerald-800"
            if primary
            else "text-emerald-700 hover:bg-emerald-50"
        )
    )
    ui.link(label, target=route).classes(classes)


def _entry_card(entry: StartPageEntry) -> None:
    with ui.link(target=entry.route).classes("no-underline text-inherit w-full"):
        with ui.card().classes(
            "w-full h-full border border-slate-200 shadow-none "
            "hover:border-emerald-500 hover:shadow-sm transition-all cursor-pointer"
        ):
            with ui.row().classes("items-start gap-3 flex-nowrap"):
                ui.icon(entry.icon).classes(f"text-2xl {entry.accent}")
                with ui.column().classes("gap-1"):
                    ui.label(entry.title).classes("font-semibold text-slate-900")
                    ui.label(entry.description).classes("text-sm text-slate-600")


def render_start_page(
    watchlists: WatchlistPageModel | None = None,
    signals: SignalPageModel | None = None,
    tasks: TaskManager | None = None,
    latest_session: Callable[[], date] | None = None,
) -> None:
    if watchlists is not None and signals is not None:
        render_workbench(watchlists, signals, tasks, latest_session)
        return
    with ui.card().classes(
        "w-full border-0 shadow-none bg-gradient-to-r from-emerald-50 to-slate-50"
    ):
        ui.label("从这里开始").classes("text-xl font-semibold text-slate-900")
        ui.label(
            "日常使用建议先同步行情，再查看今日信号；需要研究策略时进入策略实验室或策略回测。"
        ).classes("text-sm text-slate-600")

        with ui.row().classes("gap-3"):
            ui.button(
                "查看今日信号",
                icon="recommend",
                on_click=lambda: ui.navigate.to("/signals"),
            )
            ui.button(
                "同步行情数据",
                icon="sync",
                on_click=lambda: ui.navigate.to("/data"),
            ).props("outline")

    ui.label("常用功能").classes("text-lg font-semibold mt-2")
    with ui.grid(columns=2).classes("w-full gap-4 max-md:grid-cols-1"):
        for entry in _PRIMARY_ENTRIES:
            _entry_card(entry)

    ui.label("数据与系统").classes("text-lg font-semibold mt-2")
    with ui.grid(columns=4).classes("w-full gap-4 max-lg:grid-cols-2 max-md:grid-cols-1"):
        for entry in _DATA_ENTRIES:
            _entry_card(entry)

    with ui.card().classes("w-full border border-slate-200 shadow-none"):
        ui.label("推荐流程").classes("font-semibold")
        ui.label(
            "① 标的池维护关注标的  →  ② 行情数据完成同步  →  "
            "③ 策略实验室创建或调优策略  →  ④ 策略回测验证  →  "
            "⑤ 账户维护持仓  →  ⑥ 今日信号生成建议"
        ).classes("text-sm text-slate-600")


def render_workbench(
    watchlists: WatchlistPageModel,
    signals: SignalPageModel,
    tasks: TaskManager | None,
    latest_session: Callable[[], date] | None,
) -> None:
    @ui.refreshable
    def status() -> None:
        try:
            watchlist_state = watchlists.state()
            pool = watchlist_state.entry
            state = signals.state()
        except Exception:
            ui.label("工作台状态暂不可用，请到日志查看原因。").classes("text-red-700")
            ui.button("查看日志", on_click=lambda: ui.navigate.to("/logs"))
            return
        try:
            expected = latest_session() if latest_session else None
        except Exception:
            expected = None
        data_day, data_ready = _market_readiness(watchlist_state, expected)
        data_step_label = "同步完整行情"
        if not data_ready and data_day is not None and expected is not None:
            data_step_label += f"（当前 {data_day}，目标 {expected}）"
        account_id = state.active_account_profile.account_id
        steps = (
            ("添加关注标的", bool(pool and pool.enabled), "/watchlists"),
            (data_step_label, data_ready, "/data"),
            ("创建并启用策略", bool(state.strategies), "/strategies"),
            ("保存账户持仓", state.account is not None, account_url("/account", account_id)),
        )
        next_step = next((item for item in steps if not item[1]), (
            "查看并生成今日建议", False, account_url("/signals", account_id)
        ))
        with ui.card().classes("w-full bg-emerald-50 shadow-none border border-emerald-100"):
            ui.label(f"{state.active_account_profile.name} · 工作台").classes("text-xl font-semibold")
            ui.label(
                "准备已完成，可以生成今日建议。" if all(item[1] for item in steps)
                else f"下一步：{next_step[0]}"
            ).classes("text-sm text-slate-600")
            next_action = "前往同步行情" if next_step[2] == "/data" else next_step[0]
            _action_link(next_action, next_step[2], primary=True)
        with ui.grid(columns=3).classes("w-full gap-4 max-md:grid-cols-1"):
            pending = sum(
                signals.execution(item.decision_id) is None
                and not signals.decision_freshness(item).stale
                and item.result.valid_until >= date.today()
                for item in state.decision_history
            )
            for label, value, detail in (
                ("本地行情", data_day.isoformat() if data_day else "尚未同步",
                 f"完整交易日：{expected}" if expected else "交易日历尚不可用，请先同步行情"),
                ("账户资产（持仓快照）",
                 f"¥{state.account.snapshot.equity:,.2f}" if state.account else "尚未配置",
                 f"快照日期：{state.account.snapshot.as_of}" if state.account else "可先保存纯现金账户"),
                ("待处理建议", str(pending), "仅统计仍有效且未记录执行的建议"),
            ):
                with ui.card().classes("w-full border border-slate-200 shadow-none"):
                    ui.label(label).classes("text-sm text-slate-500")
                    ui.label(value).classes("text-xl font-semibold")
                    ui.label(detail).classes("text-xs text-slate-500")
        with ui.card().classes("w-full shadow-none border border-slate-200"):
            ui.label("使用准备").classes("font-semibold")
            for label, ready, route in steps:
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("check_circle" if ready else "radio_button_unchecked",
                                color="positive" if ready else "grey")
                        ui.label(label)
                    action_label = (
                        "查看"
                        if ready
                        else "前往同步"
                        if route == "/data"
                        else "去完成"
                    )
                    _action_link(action_label, route)
        if tasks is not None:
            with ui.card().classes("w-full shadow-none border border-slate-200"):
                ui.label("最近任务").classes("font-semibold")
                recent = sorted(tasks.snapshots(), key=lambda item: item.submitted_at, reverse=True)[:5]
                if not recent:
                    ui.label("本次启动还没有后台任务。").classes("text-sm text-slate-500")
                for task in recent:
                    ui.label(f"{task.name} · {task_status_label(task.status)}").classes(
                        "text-red-700" if task.status is TaskStatus.FAILED else "text-sm"
                    )
                ui.link("查看行情同步历史", "/data")
    status()
    ui.timer(10.0, status.refresh)
