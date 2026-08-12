from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from nicegui import ui


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
    with ui.header().classes("bg-slate-950 text-white items-center h-14"):
        ui.label("Compass").classes("text-lg font-semibold tracking-wide")
        ui.label("本地标的与行情管理").classes("text-xs text-slate-300")
    with ui.left_drawer(value=True).classes("bg-slate-900 text-slate-100 p-3"):
        ui.label("功能导航").classes("text-xs uppercase text-slate-400 mb-2")
        for item in navigation:
            with ui.link(target=item.route).classes(
                "w-full no-underline text-slate-100 hover:bg-slate-800 rounded px-3 py-2"
            ):
                with ui.row().classes("items-center gap-3"):
                    ui.icon(item.icon).classes("text-amber-400")
                    ui.label(item.label).classes("text-sm")
    with ui.column().classes("w-full max-w-screen-2xl mx-auto gap-4 p-4 md:p-6"):
        ui.label(title).classes("text-2xl font-semibold text-slate-900")
        ui.label(subtitle).classes("text-sm text-slate-600 -mt-3")
        yield
