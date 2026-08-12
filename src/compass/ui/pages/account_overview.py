from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Protocol

from nicegui import ui

from compass.services.export_service import DecisionExportRecord
from compass.services.local_signal_center import (
    AccountPositionInput,
    SignalDecisionFreshness,
    SignalInstrumentChoice,
)
from compass.storage.account_repository import StoredAccountSnapshot
from compass.storage.signal_account_repository import SignalAccountProfile
from compass.storage.signal_execution_repository import (
    SignalExecutionRecord,
    SignalExecutionStatus,
)


Today = Callable[[], date]


class AccountOverviewGateway(Protocol):
    def account_profiles(self) -> tuple[SignalAccountProfile, ...]: ...
    def active_account_profile(self) -> SignalAccountProfile: ...
    def select_account(self, account_id: str) -> SignalAccountProfile: ...
    def create_account(
        self, name: str, holdings_account_id: str | None = None
    ) -> SignalAccountProfile: ...
    def delete_account(self, account_id: str) -> SignalAccountProfile: ...
    def instruments(self) -> tuple[SignalInstrumentChoice, ...]: ...
    def latest_account(self) -> StoredAccountSnapshot | None: ...
    def account_history(self) -> tuple[StoredAccountSnapshot, ...]: ...
    def compact_account_history(self) -> int: ...
    def save_account(
        self, cash: object, positions: Sequence[AccountPositionInput]
    ) -> StoredAccountSnapshot: ...
    def decision_history(self) -> tuple[DecisionExportRecord, ...]: ...
    def execution(self, decision_id: str) -> SignalExecutionRecord | None: ...
    def decision_freshness(self, record: DecisionExportRecord) -> SignalDecisionFreshness: ...


@dataclass(frozen=True, slots=True)
class AccountDecisionAudit:
    decision_id: str
    decision_day: date
    valid_until: date
    status: str
    recommendation_count: int
    buy_count: int
    sell_count: int
    resulting_snapshot_row_id: int | None
    relative_impact: Decimal | None


@dataclass(frozen=True, slots=True)
class AccountOverviewState:
    profiles: tuple[SignalAccountProfile, ...]
    active_profile: SignalAccountProfile
    instruments: tuple[SignalInstrumentChoice, ...]
    latest: StoredAccountSnapshot | None
    history: tuple[StoredAccountSnapshot, ...]
    decisions: tuple[AccountDecisionAudit, ...]


def _adoption_impact(
    record: DecisionExportRecord,
    prices: dict[str, Decimal],
) -> Decimal | None:
    """Mark a full-adoption decision against current prices, relative to ignoring it."""

    impact = Decimal("0")
    for item in record.result.recommendations:
        if item.quantity_delta == 0:
            continue
        current = prices.get(str(item.instrument))
        execution = item.estimated_execution_price or item.reference_price
        if current is None:
            return None
        impact += Decimal(item.quantity_delta) * (current - execution) - item.costs.total
    return impact.quantize(Decimal("0.01"))


def _decision_status(
    execution: SignalExecutionRecord | None,
    freshness: SignalDecisionFreshness,
    valid_until: date,
    today: date,
) -> str:
    if execution is not None:
        return execution.status.value
    if freshness.stale:
        return "stale"
    if valid_until < today:
        return "expired"
    return "pending"


class AccountOverviewPageModel:
    def __init__(self, gateway: AccountOverviewGateway, *, today: Today = date.today) -> None:
        if not callable(today):
            raise TypeError("today must be callable")
        self._gateway = gateway
        self._today = today

    def select_account(self, account_id: str) -> SignalAccountProfile:
        return self._gateway.select_account(account_id)

    def create_account(
        self,
        name: object,
        holdings_account_id: object = None,
    ) -> SignalAccountProfile:
        if type(name) is not str or not name.strip():
            raise ValueError("SIGNAL_ACCOUNT_NAME_REQUIRED")
        source_id = None
        if holdings_account_id is not None:
            if type(holdings_account_id) is not str:
                raise ValueError("SIGNAL_ACCOUNT_HOLDINGS_SOURCE_REQUIRED")
            source_id = holdings_account_id
        return self._gateway.create_account(name.strip(), source_id)

    def delete_account(self, account_id: str) -> SignalAccountProfile:
        return self._gateway.delete_account(account_id)

    def compact_account_history(self) -> int:
        return self._gateway.compact_account_history()

    def save_account(
        self,
        cash: object,
        positions: tuple[tuple[str, object, object, object], ...],
    ) -> StoredAccountSnapshot:
        return self._gateway.save_account(
            cash,
            tuple(
                AccountPositionInput(instrument, quantity, available, average_cost)
                for instrument, quantity, available, average_cost in positions
            ),
        )

    def state(self) -> AccountOverviewState:
        instruments = self._gateway.instruments()
        prices = {str(item.instrument): item.close for item in instruments}
        audits = []
        today = self._today()
        if type(today) is not date:
            raise TypeError("today provider must return an exact date")
        for record in self._gateway.decision_history():
            execution = self._gateway.execution(record.decision_id)
            freshness = self._gateway.decision_freshness(record)
            audits.append(
                AccountDecisionAudit(
                    record.decision_id,
                    record.result.decision_date,
                    record.result.valid_until,
                    _decision_status(execution, freshness, record.result.valid_until, today),
                    sum(
                        item.quantity_delta != 0
                        for item in record.result.recommendations
                    ),
                    sum(item.quantity_delta > 0 for item in record.result.recommendations),
                    sum(item.quantity_delta < 0 for item in record.result.recommendations),
                    None if execution is None else execution.resulting_snapshot_row_id,
                    _adoption_impact(record, prices),
                )
            )
        return AccountOverviewState(
            tuple(self._gateway.account_profiles()),
            self._gateway.active_account_profile(),
            tuple(instruments),
            self._gateway.latest_account(),
            tuple(self._gateway.account_history()),
            tuple(audits),
        )


def _fund_chart_options(
    history: Sequence[StoredAccountSnapshot],
    decisions: Sequence[AccountDecisionAudit],
) -> dict[str, object]:
    records = tuple(history)
    categories = [
        f"{item.captured_at.strftime('%m-%d %H:%M')} · #{item.row_id}" for item in records
    ]
    category_by_row = {item.row_id: category for item, category in zip(records, categories)}
    equity_by_row = {item.row_id: float(item.snapshot.equity) for item in records}
    buy_points = []
    sell_points = []
    for item in decisions:
        row_id = item.resulting_snapshot_row_id
        if row_id is None or row_id not in category_by_row:
            continue
        point = [category_by_row[row_id], equity_by_row[row_id]]
        if item.buy_count:
            buy_points.append(point)
        if item.sell_count:
            sell_points.append(point)
    series: list[dict[str, object]] = [
        {
            "name": "账户净值",
            "type": "line",
            "showSymbol": True,
            "data": [float(item.snapshot.equity) for item in records],
        },
        {
            "name": "现金",
            "type": "bar",
            "stack": "资产",
            "data": [float(item.snapshot.cash) for item in records],
        },
        {
            "name": "持仓市值",
            "type": "bar",
            "stack": "资产",
            "data": [float(item.snapshot.equity - item.snapshot.cash) for item in records],
        },
    ]
    for name, marker, color, position, points in (
        ("B 买入成交", "B", "#dc2626", "top", buy_points),
        ("S 卖出成交", "S", "#059669", "bottom", sell_points),
    ):
        if points:
            series.append(
                {
                    "name": name,
                    "type": "scatter",
                    "symbolSize": 16,
                    "data": points,
                    "itemStyle": {"color": color},
                    "label": {
                        "show": True,
                        "formatter": marker,
                        "position": position,
                        "fontWeight": "bold",
                        "color": color,
                    },
                }
            )
    return {
        "animation": False,
        "tooltip": {"trigger": "axis"},
        "legend": {"data": [item["name"] for item in series], "top": 8},
        "grid": {"left": 72, "right": 32, "top": 56, "bottom": 72},
        "xAxis": {"type": "category", "data": categories, "axisLabel": {"rotate": 25}},
        "yAxis": {"type": "value", "name": "金额（元）", "scale": True},
        "dataZoom": (
            {"type": "inside", "start": 0, "end": 100},
            {"type": "slider", "bottom": 12, "start": 0, "end": 100},
        ),
        "series": series,
    }


def _position_chart_options(
    snapshot: StoredAccountSnapshot,
    names: dict[str, str],
) -> dict[str, object]:
    positions = snapshot.snapshot.positions
    labels = [names.get(str(item.instrument), str(item.instrument)) for item in positions]
    values = [float(item.market_value) for item in positions]
    return {
        "animation": False,
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 150, "right": 32, "top": 24, "bottom": 36},
        "xAxis": {"type": "value", "name": "市值（元）"},
        "yAxis": {"type": "category", "data": labels},
        "series": [
            {
                "name": "持仓市值",
                "type": "bar",
                "data": values,
                "itemStyle": {"color": "#2563eb"},
                "label": {"show": True, "position": "right", "formatter": "{c}"},
            }
        ],
    }


def _impact_chart_options(decisions: Sequence[AccountDecisionAudit]) -> dict[str, object]:
    rows = tuple(reversed(tuple(decisions)))
    return {
        "animation": False,
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 72, "right": 32, "top": 28, "bottom": 72},
        "xAxis": {
            "type": "category",
            "data": [f"{item.decision_day.isoformat()}\n{item.decision_id[:12]}" for item in rows],
            "axisLabel": {"rotate": 20},
        },
        "yAxis": {"type": "value", "name": "相对影响（元）"},
        "dataZoom": ({"type": "inside"}, {"type": "slider", "bottom": 12}),
        "series": [
            {
                "name": "全部采用相对不采用",
                "type": "bar",
                "data": [
                    {
                        "value": None if item.relative_impact is None else float(item.relative_impact),
                        "itemStyle": {
                            "color": (
                                "#94a3b8"
                                if item.relative_impact is None
                                else "#dc2626"
                                if item.relative_impact >= 0
                                else "#059669"
                            )
                        },
                    }
                    for item in rows
                ],
            }
        ],
    }


_STATUS_LABELS = {
    SignalExecutionStatus.EXECUTED.value: "已采用",
    SignalExecutionStatus.PARTIAL.value: "部分采用",
    SignalExecutionStatus.IGNORED.value: "明确未采用",
    "expired": "过期未记录",
    "stale": "状态变化未执行",
    "pending": "待处理",
}


def render_account_overview_page(model: AccountOverviewPageModel | None) -> None:
    if model is None:
        ui.label("账户总览服务尚未配置").classes("text-negative")
        return
    state = model.state()
    profile_options = {item.account_id: item.name for item in state.profiles}
    profile_by_id = {item.account_id: item for item in state.profiles}
    instrument_by_code = {str(item.instrument): item for item in state.instruments}
    instrument_options = {
        code: f"{item.name}（{item.instrument}）"
        for code, item in instrument_by_code.items()
    }
    cash_value = {
        "value": "100000.00" if state.latest is None else str(state.latest.snapshot.cash)
    }
    position_rows: list[dict[str, object]] = (
        []
        if state.latest is None
        else [
            {
                "instrument": str(item.instrument),
                "quantity": item.quantity,
                "available": item.available_quantity,
                "average_cost": str(item.average_cost),
            }
            for item in state.latest.snapshot.positions
        ]
    )

    def switch_account(account_id: object) -> None:
        model.select_account(str(account_id))
        ui.navigate.reload()

    with ui.row().classes("w-full items-end justify-between gap-3"):
        with ui.column().classes("gap-0"):
            ui.label("账户总览").classes("text-h6 font-semibold")
            ui.label("把持仓、资金快照、调仓建议和实际采用情况放在同一条时间线上。").classes(
                "text-sm text-grey-7"
            )
        with ui.row().classes("items-end gap-2"):
            selector = ui.select(
                profile_options,
                value=state.active_profile.account_id,
                label="账户方案",
            ).props("outlined dense options-dense").classes("min-w-64")
            selector.on_value_change(lambda event: switch_account(event.value))

            with ui.dialog() as create_dialog, ui.card():
                ui.label("新建账户").classes("font-semibold")
                ui.label("新账户可共享已有真实持仓，也可以维护独立持仓。").classes(
                    "text-sm text-grey-7"
                )
                account_name = ui.input("账户名称").props("outlined autofocus").classes("w-72")
                holdings_mode = ui.radio(
                    {
                        "shared": "共享已有持仓（推荐）",
                        "independent": "使用独立持仓",
                    },
                    value="shared",
                )
                holdings_source = ui.select(
                    profile_options,
                    label="共享哪个账户的持仓",
                    value=state.active_profile.account_id,
                ).props("outlined dense options-dense").classes("w-72")

                def update_holdings_source() -> None:
                    holdings_source.set_visibility(holdings_mode.value == "shared")

                holdings_mode.on_value_change(lambda _: update_holdings_source())
                update_holdings_source()

                def create_account() -> None:
                    try:
                        model.create_account(
                            account_name.value,
                            holdings_source.value if holdings_mode.value == "shared" else None,
                        )
                    except Exception as error:
                        ui.notify(str(error)[:120] or "账户创建失败", type="negative")
                        return
                    create_dialog.close()
                    ui.notify("账户已创建并切换", type="positive")
                    ui.navigate.reload()

                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("取消", on_click=create_dialog.close).props("flat")
                    ui.button("创建", icon="add", on_click=create_account)

            ui.button("新建账户", icon="add", on_click=create_dialog.open).props("outline")

            with ui.dialog() as delete_dialog, ui.card():
                ui.label(f"删除“{state.active_profile.name}”？").classes("font-semibold")
                ui.label("账户方案会被删除；既有持仓快照和信号审计记录仍会保留。").classes(
                    "text-sm text-grey-7"
                )

                def delete_account() -> None:
                    try:
                        model.delete_account(state.active_profile.account_id)
                    except Exception as error:
                        ui.notify(str(error)[:120] or "账户删除失败", type="negative")
                        return
                    delete_dialog.close()
                    ui.notify("账户方案已删除", type="positive")
                    ui.navigate.reload()

                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("取消", on_click=delete_dialog.close).props("flat")
                    ui.button("确认删除", on_click=delete_account, color="negative")

            delete_button = ui.button(
                icon="delete_outline",
                on_click=delete_dialog.open,
            ).props("flat round color=negative aria-label=删除账户")
            delete_button.set_enabled(len(state.profiles) > 1)

    active_holdings_id = state.active_profile.holdings_account_id
    if active_holdings_id != state.active_profile.account_id:
        source_profile = profile_by_id.get(str(active_holdings_id))
        ui.label(
            f"当前账户共享“{source_profile.name if source_profile else active_holdings_id}”的持仓；"
            "修改后，所有共享该持仓的账户都会使用最新版本。"
        ).classes("w-full text-sm text-blue-8 bg-blue-1 rounded px-3 py-2")

    with ui.card().classes("w-full"):
        ui.label("持仓配置").classes("text-subtitle1 font-semibold")
        ui.label(
            "这里维护真实持仓；共享同一持仓来源的账户方案会同步使用保存后的最新版本。"
        ).classes("text-sm text-grey-7")
        ui.input("现金（元）").bind_value(cash_value, "value").props(
            "outlined dense"
        ).classes("w-56")

        @ui.refreshable
        def position_editor() -> None:
            if not position_rows:
                ui.label("当前为空仓").classes("text-sm text-grey-6")
            for index, row in enumerate(position_rows):
                with ui.row().classes("w-full items-end gap-3"):
                    ui.select(instrument_options, label="标的").bind_value(
                        row, "instrument"
                    ).props("outlined dense options-dense").classes("min-w-72")
                    ui.number("持仓数量", min=0, step=100).bind_value(
                        row, "quantity"
                    ).props("outlined dense").classes("w-36")
                    ui.number("可卖数量", min=0, step=100).bind_value(
                        row, "available"
                    ).props("outlined dense").classes("w-36")
                    ui.input("平均成本").bind_value(row, "average_cost").props(
                        "outlined dense"
                    ).classes("w-36")
                    ui.button(
                        icon="delete",
                        on_click=lambda _, row_index=index: (
                            position_rows.pop(row_index), position_editor.refresh()
                        ),
                    ).props("flat round color=negative")

        position_editor()

        def add_position() -> None:
            used = {str(item["instrument"]) for item in position_rows}
            available = next(
                (code for code in instrument_options if code not in used),
                None,
            )
            if available is None:
                ui.notify("没有更多已同步标的可添加", type="warning")
                return
            position_rows.append(
                {
                    "instrument": available,
                    "quantity": 0,
                    "available": 0,
                    "average_cost": str(instrument_by_code[available].close),
                }
            )
            position_editor.refresh()

        def save_positions() -> None:
            try:
                saved = model.save_account(
                    cash_value["value"],
                    tuple(
                        (
                            str(row["instrument"]),
                            row["quantity"],
                            row["available"],
                            row["average_cost"],
                        )
                        for row in position_rows
                    ),
                )
            except Exception as error:
                ui.notify(str(error)[:120] or "持仓保存失败", type="negative")
                return
            ui.notify(
                "持仓无变化，未生成重复快照"
                if state.latest is not None and saved.row_id == state.latest.row_id
                else f"持仓快照 #{saved.row_id} 已保存",
                type="positive",
            )
            ui.navigate.reload()

        def load_snapshot(record: StoredAccountSnapshot) -> None:
            missing = tuple(
                str(item.instrument)
                for item in record.snapshot.positions
                if str(item.instrument) not in instrument_by_code
            )
            if missing:
                ui.notify(
                    f"该版本有 {len(missing)} 个持仓缺少当前行情，暂时不能载入编辑",
                    type="warning",
                )
                return
            cash_value["value"] = str(record.snapshot.cash)
            position_rows.clear()
            position_rows.extend(
                {
                    "instrument": str(item.instrument),
                    "quantity": item.quantity,
                    "available": item.available_quantity,
                    "average_cost": str(item.average_cost),
                }
                for item in record.snapshot.positions
            )
            history_dialog.close()
            position_editor.refresh()
            ui.notify(f"已载入快照 #{record.row_id}，保存后才会生成新版本", type="positive")

        with ui.dialog() as history_dialog, ui.card().classes("w-[900px] max-w-[95vw]"):
            ui.label(f"{state.active_profile.name} · 持仓快照历史").classes(
                "text-subtitle1 font-semibold"
            )
            ui.label("历史快照只读；可载入编辑，保存后形成新版本。").classes(
                "text-sm text-grey-7"
            )
            with ui.column().classes("w-full gap-2 max-h-[65vh] overflow-y-auto"):
                for history_record in reversed(state.history):
                    history_snapshot = history_record.snapshot
                    with ui.expansion(
                        f"快照 #{history_record.row_id} · {history_snapshot.as_of.isoformat()} · "
                        f"净值 ¥{history_snapshot.equity:,.2f} · "
                        f"现金 ¥{history_snapshot.cash:,.2f} · "
                        f"持仓 {len(history_snapshot.positions)} 个"
                    ).classes("w-full border rounded"):
                        if history_snapshot.positions:
                            ui.table(
                                columns=[
                                    {"name": "instrument", "label": "标的", "field": "instrument"},
                                    {"name": "quantity", "label": "持仓", "field": "quantity", "align": "right"},
                                    {"name": "available", "label": "可卖", "field": "available", "align": "right"},
                                    {"name": "cost", "label": "平均成本", "field": "cost", "align": "right"},
                                ],
                                rows=[
                                    {
                                        "instrument": instrument_options.get(
                                            str(item.instrument), str(item.instrument)
                                        ),
                                        "quantity": item.quantity,
                                        "available": item.available_quantity,
                                        "cost": str(item.average_cost),
                                    }
                                    for item in history_snapshot.positions
                                ],
                                row_key="instrument",
                            ).classes("w-full").props("flat bordered dense")
                        else:
                            ui.label("该版本为空仓").classes("text-sm text-grey-6")
                        ui.button(
                            "载入编辑",
                            icon="edit_note",
                            on_click=lambda _, item=history_record: load_snapshot(item),
                        ).props("outline")
            with ui.row().classes("w-full justify-end"):
                ui.button("关闭", on_click=history_dialog.close).props("flat")

        def compact_history() -> None:
            try:
                deleted = model.compact_account_history()
            except Exception as error:
                ui.notify(str(error)[:120] or "历史清理失败", type="negative")
                return
            ui.notify(
                f"已清理 {deleted} 个重复快照" if deleted else "没有可安全清理的重复快照",
                type="positive",
            )
            ui.navigate.reload()

        with ui.row().classes("gap-2"):
            ui.button("添加持仓", icon="add", on_click=add_position).props("outline")
            ui.button("保存持仓", icon="save", on_click=save_positions)
            if state.history:
                ui.button(
                    f"查看历史（{len(state.history)}）",
                    icon="history",
                    on_click=history_dialog.open,
                ).props("flat")
                ui.button(
                    "清理重复历史",
                    icon="cleaning_services",
                    on_click=compact_history,
                ).props("flat")

    if state.latest is None:
        return

    latest = state.latest.snapshot
    market_value = latest.equity - latest.cash
    with ui.row().classes("w-full gap-3"):
        for label, value in (
            ("账户净值", f"¥{latest.equity:,.2f}"),
            ("现金", f"¥{latest.cash:,.2f}"),
            ("持仓市值", f"¥{market_value:,.2f}"),
            ("持仓标的", f"{len(latest.positions)} 个"),
            ("持仓版本", f"{len(state.history)} 版"),
        ):
            with ui.card().classes("min-w-40 border border-slate-200 shadow-none"):
                ui.label(label).classes("text-xs text-grey-6")
                ui.label(value).classes("text-lg font-semibold")

    with ui.card().classes("w-full"):
        ui.label("资金与仓位变化").classes("text-subtitle1 font-semibold")
        ui.label("按保存持仓或记录成交时形成的快照展示，不代表每个交易日自动盯市。").classes(
            "text-xs text-grey-6"
        )
        ui.label("B / S 分别表示已记录的实际买入 / 卖出成交。").classes(
            "text-xs text-grey-6"
        )
        ui.echart(_fund_chart_options(state.history, state.decisions)).classes("w-full h-96")

    names = {str(item.instrument): item.name for item in state.instruments}
    with ui.card().classes("w-full"):
        ui.label("当前持仓结构").classes("text-subtitle1 font-semibold")
        if latest.positions:
            ui.echart(_position_chart_options(state.latest, names)).classes("w-full h-80")
            ui.table(
                columns=[
                    {"name": "instrument", "label": "标的", "field": "instrument"},
                    {"name": "quantity", "label": "数量", "field": "quantity", "align": "right"},
                    {"name": "cost", "label": "成本", "field": "cost", "align": "right"},
                    {"name": "price", "label": "现价", "field": "price", "align": "right"},
                    {"name": "value", "label": "市值", "field": "value", "align": "right"},
                    {"name": "pnl", "label": "浮动盈亏", "field": "pnl", "align": "right"},
                ],
                rows=[
                    {
                        "instrument": f"{names.get(str(item.instrument), item.instrument)}（{item.instrument}）",
                        "quantity": item.quantity,
                        "cost": f"¥{item.average_cost:,.4f}",
                        "price": f"¥{item.mark_price:,.4f}",
                        "value": f"¥{item.market_value:,.2f}",
                        "pnl": f"¥{(item.mark_price - item.average_cost) * item.quantity:,.2f}",
                    }
                    for item in latest.positions
                ],
                row_key="instrument",
            ).classes("w-full").props("flat bordered dense")
        else:
            ui.label("当前为空仓")

    with ui.card().classes("w-full"):
        ui.label("建议采用情况与事后影响试算").classes("text-subtitle1 font-semibold")
        ui.label(
            "柱状图按“建议成交价 → 当前价”估算全部采用相对不采用的差额，包含建议费用；"
            "它用于快速复盘，不是严格历史回测，也不处理后续信号重叠和资金再投资。"
        ).classes("text-xs text-orange-8")
        if state.decisions:
            ui.echart(_impact_chart_options(state.decisions)).classes("w-full h-80")
            ui.table(
                columns=[
                    {"name": "day", "label": "信号日", "field": "day"},
                    {"name": "status", "label": "采用情况", "field": "status"},
                    {"name": "count", "label": "调仓标的", "field": "count", "align": "right"},
                    {"name": "direction", "label": "建议方向", "field": "direction"},
                    {"name": "impact", "label": "全部采用相对影响", "field": "impact", "align": "right"},
                    {"name": "id", "label": "决策 ID", "field": "id"},
                ],
                rows=[
                    {
                        "day": item.decision_day.isoformat(),
                        "status": _STATUS_LABELS[item.status],
                        "count": item.recommendation_count,
                        "direction": f"买 {item.buy_count} / 卖 {item.sell_count}",
                        "impact": (
                            "行情不足"
                            if item.relative_impact is None
                            else f"¥{item.relative_impact:+,.2f}"
                        ),
                        "id": item.decision_id,
                    }
                    for item in state.decisions
                ],
                row_key="id",
            ).classes("w-full").props("flat bordered dense")
        else:
            ui.label("暂无调仓建议记录。")

        with ui.row().classes("w-full items-center justify-between bg-blue-1 rounded px-3 py-3"):
            ui.label(
                "严格比较“采用 / 不采用”时，应使用同一初始持仓和行情区间：一组执行策略信号，"
                "另一组保持初始持仓不调仓，再比较收益、回撤和费用。"
            ).classes("text-sm text-blue-9")
            ui.button(
                "去策略回测",
                icon="query_stats",
                on_click=lambda: ui.navigate.to("/backtests"),
            ).props("outline")
