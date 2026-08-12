from __future__ import annotations

from dataclasses import dataclass

from nicegui import ui


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


def render_start_page() -> None:
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
