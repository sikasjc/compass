from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from nicegui import ui
from compass.ui.navigation import account_url


@dataclass(frozen=True, slots=True)
class NavigationItem:
    label: str
    route: str
    icon: str


@contextmanager
def page_shell(
    title: str,
    subtitle: str,
    navigation: tuple[NavigationItem, ...],
) -> Iterator[None]:
    ui.colors(primary="#275D52", secondary="#B7791F", accent="#315B78")
    ui.page_title(f"{title} · Compass")
    request = ui.context.client.request
    current_path = request.url.path
    account_id = request.query_params.get("account_id")
    drawer = ui.left_drawer(value=None).props("width=232 breakpoint=900").classes(
        "bg-slate-900 text-slate-100 p-3"
    )
    with ui.header().classes("bg-slate-950 text-white items-center h-14"):
        ui.button(icon="menu", on_click=lambda: drawer.toggle()).props(
            "flat round color=white aria-label=展开或收起导航"
        )
        ui.label("Compass").classes("text-lg font-semibold tracking-wide")
        ui.label("策略研究与账户工作台").classes("text-xs text-slate-300")
    with drawer:
        for item in navigation:
            group = {"/": "日常操作", "/backtests": "策略研究", "/data": "数据与系统"}
            if item.route in group:
                ui.label(group[item.route]).classes("text-xs text-slate-400 mt-4 mb-2 px-3")
            active = current_path == item.route or (
                item.route == "/strategies" and current_path.startswith("/strategies/")
            )
            route = (
                account_url(item.route, account_id)
                if account_id and item.route in {"/account", "/signals"}
                else item.route
            )
            with ui.link(target=route).classes(
                "w-full no-underline text-slate-100 rounded px-3 py-2 "
                + ("bg-emerald-800 font-semibold" if active else "hover:bg-slate-800")
            ) as link:
                if active:
                    link.props('aria-current=page')
                with ui.row().classes("items-center gap-3"):
                    ui.icon(item.icon).classes("text-emerald-200" if active else "text-slate-400")
                    ui.label(item.label).classes("text-sm")
    with ui.column().classes("w-full max-w-screen-2xl mx-auto gap-4 p-4 md:p-6"):
        ui.label(title).classes("text-2xl font-semibold text-slate-900")
        ui.label(subtitle).classes("text-sm text-slate-600 -mt-3")
        yield
