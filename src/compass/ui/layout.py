from __future__ import annotations

from collections.abc import Iterator, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
from compass.config import runtime_label

from nicegui import ui
from compass.ui.navigation import account_url, account_storage_key


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
    *, runtime_root: Path | None = None,
    account_label: Callable[[str], str] | None = None,
) -> Iterator[None]:
    ui.colors(primary="#275D52", secondary="#B7791F", accent="#315B78")
    ui.page_title(f"{title} · Compass")
    request = ui.context.client.request
    current_path = request.url.path
    account_id = request.query_params.get("account_id")
    ui.add_body_html("""<script>(() => {
        const key = """ + json.dumps(account_storage_key(runtime_root)).replace("<", "\\u003c") + """;
        const current = new URL(location.href);
        try {
            const explicit = current.searchParams.get('account_id');
            if (explicit) sessionStorage.setItem(key, explicit);
            else if (!['/account', '/signals'].includes(current.pathname)) {
                const selected = sessionStorage.getItem(key);
                if (selected) {
                    current.searchParams.set('account_id', selected);
                    location.replace(current.href);
                }
            }
        } catch (_) {}
    })()</script>""")
    drawer = ui.left_drawer(value=None).props("width=232 breakpoint=900").classes(
        "bg-slate-900 text-slate-100 p-3"
    )
    with ui.header().classes("bg-slate-950 text-white items-center h-14"):
        ui.button(icon="menu", on_click=lambda: drawer.toggle()).props(
            "flat round color=white aria-label=展开或收起导航"
        )
        ui.label("Compass").classes("text-lg font-semibold tracking-wide")
        ui.label("策略研究与账户工作台").classes("text-xs text-slate-300 hidden md:block")
        if runtime_root is not None:
            label = runtime_label(runtime_root)
            ui.badge(label, color="green" if label == "正式数据" else "orange")
            ui.link("数据目录", account_url("/settings", account_id) if account_id else "/settings").classes(
                "text-xs text-slate-200"
            ).tooltip(str(runtime_root))
        if account_id:
            try:
                selected_name = account_label(account_id) if account_label else account_id
            except LookupError:
                selected_name = "账户不可用"
            ui.label(f"账户：{selected_name}").classes("text-xs text-slate-200").tooltip(account_id)
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
                if account_id
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
