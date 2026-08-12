from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, DecimalException
import re
from typing import Protocol

from nicegui import ui
from nicegui.elements.dialog import Dialog

from compass.services.decision_service import (
    DecisionSide,
    RebalanceRecommendation,
    StrategyDecisionTrace,
)
from compass.services.export_service import DecisionExportRecord
from compass.services.local_decision_gateway import SelectedDecisionStrategy
from compass.services.local_signal_center import (
    SignalDecisionComparison,
    SignalDecisionFreshness,
    SignalExecutionFillInput,
    SignalInstrumentChoice,
    SignalStrategyChoice,
)
from compass.storage.account_repository import StoredAccountSnapshot
from compass.storage.signal_account_repository import SignalAccountProfile
from compass.storage.signal_execution_repository import (
    SignalExecutionRecord,
    SignalExecutionStatus,
)


class SignalGateway(Protocol):
    def account_profiles(self) -> tuple[SignalAccountProfile, ...]: ...
    def active_account_profile(self) -> SignalAccountProfile: ...
    def select_account(self, account_id: str) -> SignalAccountProfile: ...
    def save_strategy_configuration(
        self,
        selections: tuple[SelectedDecisionStrategy, ...],
        *,
        cash_reserve: Decimal,
        minimum_trade_amount: Decimal,
    ) -> SignalAccountProfile: ...
    def instruments(self) -> tuple[SignalInstrumentChoice, ...]: ...
    def strategies(self) -> tuple[SignalStrategyChoice, ...]: ...
    def latest_account(self) -> StoredAccountSnapshot | None: ...
    def generate(
        self,
        selections: tuple[SelectedDecisionStrategy, ...],
        *,
        cash_reserve: Decimal,
        minimum_trade_amount: Decimal,
    ) -> DecisionExportRecord: ...
    def readable_decisions(self) -> tuple[tuple[DecisionExportRecord, ...], int]: ...
    def decision(self, decision_id: str) -> DecisionExportRecord | None: ...
    def execution(self, decision_id: str) -> SignalExecutionRecord | None: ...
    def decision_freshness(self, record: DecisionExportRecord) -> SignalDecisionFreshness: ...
    def delete_decision(self, decision_id: str) -> bool: ...
    def clear_decisions(self) -> int: ...
    def clear_invalid_decisions(self) -> int: ...
    def compare_decision(self, decision_id: str) -> SignalDecisionComparison: ...
    def record_execution(
        self,
        decision_id: str,
        status: SignalExecutionStatus,
        fills: tuple[SignalExecutionFillInput, ...],
        *,
        fees: object,
        recorded_at: datetime,
    ) -> SignalExecutionRecord: ...


@dataclass(frozen=True, slots=True)
class SignalPageState:
    account_profiles: tuple[SignalAccountProfile, ...]
    active_account_profile: SignalAccountProfile
    instruments: tuple[SignalInstrumentChoice, ...]
    strategies: tuple[SignalStrategyChoice, ...]
    account: StoredAccountSnapshot | None
    latest_decision: DecisionExportRecord | None
    decision_history: tuple[DecisionExportRecord, ...]
    invalid_decision_count: int


class SignalPageModel:
    def __init__(self, gateway: SignalGateway) -> None:
        self._gateway = gateway

    def state(self) -> SignalPageState:
        decisions, invalid_count = self._gateway.readable_decisions()
        return SignalPageState(
            self._gateway.account_profiles(),
            self._gateway.active_account_profile(),
            self._gateway.instruments(),
            self._gateway.strategies(),
            self._gateway.latest_account(),
            decisions[0] if decisions else None,
            decisions,
            invalid_count,
        )

    def select_account(self, account_id: str) -> SignalAccountProfile:
        return self._gateway.select_account(account_id)

    def generate(
        self,
        allocations: tuple[tuple[str, object], ...],
        cash_reserve_percent: object,
        minimum_trade_amount: object,
    ) -> DecisionExportRecord:
        selections, reserve, minimum = self._configuration(
            allocations,
            cash_reserve_percent,
            minimum_trade_amount,
        )
        return self._gateway.generate(
            selections,
            cash_reserve=reserve,
            minimum_trade_amount=minimum,
        )

    def save_strategy_configuration(
        self,
        allocations: tuple[tuple[str, object], ...],
        cash_reserve_percent: object,
        minimum_trade_amount: object,
    ) -> SignalAccountProfile:
        selections, reserve, minimum = self._configuration(
            allocations,
            cash_reserve_percent,
            minimum_trade_amount,
        )
        return self._gateway.save_strategy_configuration(
            selections,
            cash_reserve=reserve,
            minimum_trade_amount=minimum,
        )

    @staticmethod
    def _configuration(
        allocations: tuple[tuple[str, object], ...],
        cash_reserve_percent: object,
        minimum_trade_amount: object,
    ) -> tuple[tuple[SelectedDecisionStrategy, ...], Decimal, Decimal]:
        selections = tuple(
            SelectedDecisionStrategy(instance_id, _percent(value, label="策略预算"))
            for instance_id, value in allocations
        )
        if not selections:
            raise ValueError("SIGNAL_STRATEGY_REQUIRED")
        return (
            selections,
            _percent(cash_reserve_percent, label="现金预留"),
            _number(minimum_trade_amount, label="最小交易金额"),
        )

    def decision(self, decision_id: str) -> DecisionExportRecord | None:
        return self._gateway.decision(decision_id)

    def execution(self, decision_id: str) -> SignalExecutionRecord | None:
        return self._gateway.execution(decision_id)

    def decision_freshness(self, record: DecisionExportRecord) -> SignalDecisionFreshness:
        return self._gateway.decision_freshness(record)

    def delete_decision(self, decision_id: str) -> bool:
        return self._gateway.delete_decision(decision_id)

    def clear_decisions(self) -> int:
        return self._gateway.clear_decisions()

    def clear_invalid_decisions(self) -> int:
        return self._gateway.clear_invalid_decisions()

    def compare_decision(self, decision_id: str) -> SignalDecisionComparison:
        result = self._gateway.compare_decision(decision_id)
        if type(result) is not SignalDecisionComparison:
            raise TypeError("signal comparison result is invalid")
        return result

    def record_execution(
        self,
        decision_id: str,
        status: SignalExecutionStatus,
        fills: tuple[tuple[str, object, object], ...],
        fees: object,
    ) -> SignalExecutionRecord:
        return self._gateway.record_execution(
            decision_id,
            status,
            tuple(
                SignalExecutionFillInput(instrument, quantity, price)
                for instrument, quantity, price in fills
            ),
            fees=fees,
            recorded_at=datetime.now().astimezone(),
        )


def _number(value: object, *, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (DecimalException, ValueError):
        raise ValueError(f"{label}必须是数字") from None
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{label}必须是非负数")
    return parsed


def _percent(value: object, *, label: str) -> Decimal:
    parsed = _number(value, label=label)
    if parsed > 100:
        raise ValueError(f"{label}不能超过 100%")
    return parsed / Decimal("100")


def _error_text(error: Exception) -> str:
    raw = str(error)
    code = raw.split(":", 1)[0]
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", code):
        translations = {
            "ACCOUNT_SNAPSHOT_MISSING": "请先保存当前持仓",
            "DECISION_BUDGET_EXCEEDS_AVAILABLE_CAPITAL": "策略预算与现金预留合计不能超过 100%",
            "DECISION_POOL_DATA_MISSING": "策略或持仓标的缺少本地行情，请先同步",
            "DECISION_CALENDAR_UNAVAILABLE": "无法取得下一交易日，请检查交易日历网络",
            "DECISION_STRATEGY_UNAVAILABLE": "所选策略已停用或不存在",
            "SIGNAL_MARKET_DATA_MISSING": "暂无可用本地行情",
            "SIGNAL_ACCOUNT_EQUITY_REQUIRED": "账户净值必须大于零",
            "SIGNAL_POSITION_DATA_MISSING": "持仓标的缺少本地行情",
            "SIGNAL_STRATEGY_REQUIRED": "请至少选择一个策略",
            "SIGNAL_ACCOUNT_NAME_REQUIRED": "请输入账户名称",
            "SIGNAL_ACCOUNT_LAST_DELETE_FORBIDDEN": "至少需要保留一个账户",
            "SIGNAL_ACCOUNT_NOT_FOUND": "账户不存在或已被删除",
            "SIGNAL_ACCOUNT_NAME_DUPLICATE": "账户名称已经存在",
            "SIGNAL_ACCOUNT_HOLDINGS_SOURCE_NOT_FOUND": "共享的持仓来源不存在",
            "SIGNAL_ACCOUNT_HOLDINGS_IN_USE": "该账户的持仓正在被其他方案共享，不能删除",
            "SIGNAL_EXECUTION_ALREADY_RECORDED": "该信号已经记录过执行结果",
            "SIGNAL_EXECUTION_QUANTITY_INVALID": "成交数量必须与建议方向一致且不能超过建议数量",
            "SIGNAL_EXECUTION_CASH_NEGATIVE": "成交后现金不足，请检查成交数量、价格或费用",
            "SIGNAL_EXECUTION_PRICE_INVALID": "成交价格必须大于零",
            "SIGNAL_EXECUTION_INCOMPLETE": "仍有建议数量未记录，请选择“部分执行”",
            "SIGNAL_EXECUTION_ALREADY_COMPLETE": "成交已覆盖全部建议，请选择“已执行”",
            "SIGNAL_DECISION_STALE": "持仓、策略或行情已变化，请重新生成建议",
            "SIGNAL_DECISION_EXPIRED": "该建议已经过期，请同步行情并重新生成",
            "SIGNAL_DECISION_NOT_FOUND": "该信号不存在或不属于当前方案",
            "SIGNAL_DECISION_ADOPTED_DELETE_FORBIDDEN": "该信号已采用并形成持仓审计记录，不能清理",
            "SIGNAL_COMPARISON_ACCOUNT_MISSING": "生成信号时的账户快照已不可用，无法进行对照回测",
            "SIGNAL_COMPARISON_MARKET_DATA_MISSING": "对照回测所需的标的行情不完整，请先同步行情",
            "SIGNAL_COMPARISON_RANGE_MISSING": "信号日期之后暂无共同交易日行情，暂时无法对照回测",
        }
        return f"{translations.get(code, '操作未完成')}（{code}）"
    return raw if raw and len(raw) <= 120 else "操作未完成（SIGNAL_OPERATION_FAILED）"


def _instrument_label(choice: SignalInstrumentChoice) -> str:
    return f"{choice.name}（{choice.instrument}）"


def _reason_text(codes: tuple[str, ...]) -> str:
    names = {
        "NO_TRADE": "无需交易",
        "ROUNDED": "按整手取整",
        "AVAILABILITY_LIMITED": "受可卖数量限制",
        "MINIMUM_TRADE_AMOUNT": "低于最小交易金额",
        "RISK_BLOCKED": "风控阻止",
        "CASH_SCALED": "按可用现金缩减",
        "SELL_FEES_EXCEED_AVAILABLE_CASH": "卖出费用超过现金",
    }
    return "、".join(names.get(code, code) for code in codes) or "策略目标变化"


def _strategy_reason_text(code: str) -> str:
    names = {
        "GENERATED": "策略已正常生成目标仓位",
        "MA_BULL_CONFIRMED": "短期均线连续高于长期均线，趋势信号看多",
        "MA_BEAR_EXIT": "短期均线连续低于长期均线，趋势信号要求退出",
        "MOMENTUM_TOP_N": "动量排名进入候选前列，且趋势和波动条件通过",
        "CROSS_SECTIONAL_TOP_N": "横截面动量排名进入前列",
        "MEAN_REVERSION_BUY": "价格偏离均值达到买入条件",
        "MEAN_REVERSION_EXIT": "价格回归或达到退出条件",
        "BUY_AND_HOLD_TARGET": "买入并持有策略要求维持目标仓位",
        "DSL_BUY": "自定义规则触发买入条件",
        "DSL_SELL": "自定义规则触发卖出条件",
        "DSL_HOLD": "自定义规则未触发卖出，继续持有",
        "RISK_ALTERNATIVE": "主候选未通过时转入风险替代标的",
        "NO_CANDIDATES_CASH": "没有标的通过条件，本策略建议持有现金",
        "DSL_NO_SIGNAL_CASH": "自定义规则没有产生有效信号，建议持有现金",
        "INSUFFICIENT_HISTORY": "历史行情长度不足，本次跳过",
        "STALE_DATA": "存在非最新行情，本次跳过",
        "NOT_REBALANCE_SESSION": "尚未到策略设定的调仓日，本次跳过",
        "NO_INSTRUMENTS": "没有可供策略计算的标的，本次跳过",
    }
    return names.get(code, code)


def _strategy_trace_text(trace: StrategyDecisionTrace) -> str:
    status = {
        "GENERATED": "已生成仓位目标",
        "CASH": "建议现金",
        "SKIPPED": "本次跳过",
    }.get(trace.status.value, trace.status.value)
    return f"{status}：{_strategy_reason_text(trace.reason_code)}"


def _recommendation_explanation(
    item: RebalanceRecommendation,
    strategy_id: str | None = None,
) -> str:
    signals = tuple(
        dict.fromkeys(
            _strategy_reason_text(intent.reason_code)
            for intent in item.raw_intents
            if strategy_id is None or intent.strategy_id == strategy_id
        )
    )
    signal_text = "；".join(signals) if signals else "组合中没有策略要求持有该标的"
    action = {
        DecisionSide.BUY: "组合目标高于当前仓位，因此建议买入",
        DecisionSide.SELL: "组合目标低于当前仓位，因此建议卖出",
        DecisionSide.NONE: "目标数量与当前数量一致，因此无需交易",
    }[item.side]
    adjustments = _reason_text(item.reason_codes)
    return f"{signal_text}；{action}。数量换算：{adjustments}。"


def _recommendations_by_strategy(
    traces: tuple[StrategyDecisionTrace, ...],
    recommendations: tuple[RebalanceRecommendation, ...],
) -> tuple[
    tuple[tuple[StrategyDecisionTrace, tuple[RebalanceRecommendation, ...]], ...],
    tuple[RebalanceRecommendation, ...],
]:
    groups = []
    assigned: set[int] = set()
    for trace in traces:
        matched = tuple(
            item
            for item in recommendations
            if len(traces) == 1
            or any(intent.strategy_id == trace.strategy_id for intent in item.raw_intents)
        )
        groups.append((trace, matched))
        assigned.update(id(item) for item in matched)
    return tuple(groups), tuple(item for item in recommendations if id(item) not in assigned)


def _decision_deletable(status: SignalExecutionStatus | None) -> bool:
    return status is None or status is SignalExecutionStatus.IGNORED


def _decision_notice(
    freshness: SignalDecisionFreshness,
    decision_day: date,
    valid_until: date,
    today: date,
) -> tuple[str, str]:
    freshness_names = {
        "HOLDINGS_CHANGED": "当前持仓已变化",
        "STRATEGY_CONFIGURATION_CHANGED": "策略设置已变化",
        "MARKET_DATA_CHANGED": "行情版本已变化",
    }
    if freshness.stale:
        reasons = "、".join(freshness_names.get(item, item) for item in freshness.reasons)
        return f"该建议已失效：{reasons}。请重新生成后再执行。", "negative"
    if valid_until < today:
        return "该建议已过有效期，仅供复盘；请同步最新行情并重新生成。", "negative"
    if decision_day != today:
        return (
            f"该建议基于最近完整交易日 {decision_day.isoformat()} 的收盘数据，"
            f"有效至 {valid_until.isoformat()}；当前账户、策略与行情版本一致。",
            "warning",
        )
    return "该建议与当前账户、策略和行情版本一致，且仍在有效期内。", "positive"


def _side_label(side: DecisionSide) -> tuple[str, str]:
    if side is DecisionSide.BUY:
        return "买入", "text-positive"
    if side is DecisionSide.SELL:
        return "卖出", "text-negative"
    return "不操作", "text-grey-7"


def render_signals_page(model: SignalPageModel | None) -> None:
    if model is None:
        ui.label("今日信号服务尚未配置").classes("text-negative")
        return
    state = model.state()
    choice_by_code = {str(item.instrument): item for item in state.instruments}
    option_labels = {code: _instrument_label(item) for code, item in choice_by_code.items()}
    configured_strategies = {
        item.strategy_instance_id: item.budget for item in state.active_account_profile.strategies
    }
    strategy_rows = {
        item.instance_id: {
            "selected": item.instance_id in configured_strategies,
            "budget": float(configured_strategies.get(item.instance_id, Decimal("0.3")) * 100),
        }
        for item in state.strategies
    }
    strategy_name_by_id = {item.instance_id: item.name for item in state.strategies}
    selected_decision = {"record": state.latest_decision}
    selected_comparison: dict[str, SignalDecisionComparison | None] = {"result": None}

    ui.label("多账户 → 最新收盘信号 → 调仓建议").classes("text-h6 font-semibold")
    ui.label(
        "使用本地完整日线生成建议，不连接券商、不自动下单。建议生成后会固化账户、策略版本和行情清单。"
    ).classes("text-sm text-grey-7")
    if state.invalid_decision_count:
        ui.label(
            f"有 {state.invalid_decision_count} 条旧信号的行情依赖已被历史清理规则删除，"
            "现已隔离；这些记录无法复现，请使用当前行情重新生成。"
        ).classes("w-full text-sm text-amber-900 bg-amber-1 rounded px-3 py-2")

    with ui.card().classes("w-full"):
        ui.label("查看账户信号").classes("text-subtitle1 font-semibold")
        ui.label("账户资料和真实持仓统一到账户页面维护；这里仅切换要查看和生成信号的账户。").classes(
            "text-sm text-grey-7"
        )

        def switch_account(account_id: object) -> None:
            try:
                model.select_account(str(account_id))
            except Exception as error:
                ui.notify(_error_text(error), type="negative")
                return
            ui.navigate.reload()

        account_options = {item.account_id: item.name for item in state.account_profiles}
        with ui.row().classes("w-full items-end gap-2"):
            account_select = (
                ui.select(
                    account_options,
                    label="当前方案",
                    value=state.active_account_profile.account_id,
                )
                .props("outlined dense options-dense")
                .classes("min-w-64")
            )
            account_select.on_value_change(lambda event: switch_account(event.value))
            ui.button(
                "前往账户页维护",
                icon="manage_accounts",
                on_click=lambda: ui.navigate.to("/account"),
            ).props("outline")

    with ui.card().classes("w-full"):
        ui.label(f"1. 当前持仓 · {state.active_account_profile.name}").classes(
            "text-subtitle1 font-semibold"
        )
        if state.account is None:
            ui.label("尚未保存持仓，请到账户页面完成账户和持仓配置。")
            ui.button(
                "配置账户持仓",
                icon="manage_accounts",
                on_click=lambda: ui.navigate.to("/account"),
            ).props("outline")
        else:
            snapshot = state.account.snapshot
            ui.label(
                f"账户净值 ¥{snapshot.equity:,.2f} · 现金 ¥{snapshot.cash:,.2f} · "
                f"持仓 {len(snapshot.positions)} 个 · 快照 #{state.account.row_id} · "
                f"估值日 {snapshot.as_of.isoformat()}"
            ).classes("text-sm text-grey-7")
            if snapshot.positions:
                ui.table(
                    columns=[
                        {"name": "instrument", "label": "标的", "field": "instrument"},
                        {"name": "quantity", "label": "持仓", "field": "quantity", "align": "right"},
                        {"name": "available", "label": "可卖", "field": "available", "align": "right"},
                        {"name": "cost", "label": "平均成本", "field": "cost", "align": "right"},
                        {"name": "price", "label": "当前价", "field": "price", "align": "right"},
                        {"name": "value", "label": "市值", "field": "value", "align": "right"},
                    ],
                    rows=[
                        {
                            "instrument": option_labels.get(
                                str(position.instrument), str(position.instrument)
                            ),
                            "quantity": position.quantity,
                            "available": position.available_quantity,
                            "cost": f"¥{position.average_cost:,.4f}",
                            "price": f"¥{position.mark_price:,.4f}",
                            "value": f"¥{position.market_value:,.2f}",
                        }
                        for position in snapshot.positions
                    ],
                    row_key="instrument",
                ).classes("w-full").props("flat bordered dense")
            else:
                ui.label("当前为空仓").classes("text-sm text-grey-6")

    with ui.card().classes("w-full"):
        ui.label("2. 选择策略并生成最新收盘信号").classes("text-subtitle1 font-semibold")
        if not state.strategies:
            ui.label("暂无启用的已保存策略，请先到策略实验室创建策略模板。")
        for strategy in state.strategies:
            row = strategy_rows[strategy.instance_id]
            with ui.row().classes("w-full items-center gap-3"):
                ui.checkbox(strategy.name).bind_value(row, "selected")
                ui.label(strategy.strategy_type).classes("text-xs text-grey-6 min-w-40")
                ui.number("资金预算（%）", min=1, max=100, step=5).bind_value(
                    row, "budget"
                ).props("outlined dense").classes("w-40")
        parameters = {
            "reserve": float(state.active_account_profile.cash_reserve * 100),
            "minimum": float(state.active_account_profile.minimum_trade_amount),
        }
        with ui.row().classes("items-end gap-3"):
            ui.number("现金预留（%）", min=0, max=99, step=5).bind_value(
                parameters, "reserve"
            ).props("outlined dense").classes("w-40")
            ui.number("最小交易金额（元）", min=0, step=1000).bind_value(
                parameters, "minimum"
            ).props("outlined dense").classes("w-48")

            def selected_allocations() -> tuple[tuple[str, object], ...]:
                return tuple(
                    (instance_id, row["budget"])
                    for instance_id, row in strategy_rows.items()
                    if row["selected"] is True
                )

            def save_strategy_configuration() -> None:
                try:
                    model.save_strategy_configuration(
                        selected_allocations(),
                        parameters["reserve"],
                        parameters["minimum"],
                    )
                except Exception as error:
                    ui.notify(_error_text(error), type="negative")
                    return
                ui.notify("当前方案的策略设置已保存", type="positive")

            def generate_signal() -> None:
                try:
                    record = model.generate(
                        selected_allocations(),
                        parameters["reserve"],
                        parameters["minimum"],
                    )
                except Exception as error:
                    ui.notify(_error_text(error), type="negative")
                    return
                selected_decision["record"] = record
                ui.notify("最新收盘信号与调仓建议已生成", type="positive")
                ui.navigate.reload()

            ui.button(
                "保存策略设置",
                icon="save",
                on_click=save_strategy_configuration,
            ).props("outline").set_enabled(bool(state.strategies))
            ui.button("生成调仓建议", icon="auto_graph", on_click=generate_signal).props(
                "color=primary"
            ).set_enabled(state.account is not None and bool(state.strategies))

    @ui.refreshable
    def decision_view() -> None:
        record = selected_decision["record"]
        with ui.card().classes("w-full"):
            ui.label("3. 调仓建议").classes("text-subtitle1 font-semibold")
            if record is None:
                ui.label("尚未生成信号。保存持仓并选择策略后即可生成。")
                return
            result = record.result
            ui.label(
                f"{result.decision_date.isoformat()} 收盘信号 · 建议有效至 "
                f"{result.valid_until.isoformat()} · 账户快照 #{result.account_snapshot_row_id}"
            ).classes("font-medium")
            ui.label(
                f"决策净值 ¥{result.decision_equity:,.2f} · 调仓后预计现金 "
                f"¥{result.remaining_cash:,.2f} · 决策 ID {record.decision_id}"
            ).classes("text-sm text-grey-7")
            freshness = model.decision_freshness(record)
            notice, notice_type = _decision_notice(
                freshness,
                result.decision_date,
                result.valid_until,
                date.today(),
            )
            notice_classes = {
                "negative": "text-negative bg-red-1",
                "warning": "text-orange-9 bg-orange-1",
                "positive": "text-green-9 bg-green-1",
            }[notice_type]
            ui.label(notice).classes(
                f"w-full text-sm rounded px-3 py-2 {notice_classes}"
            )
            execution = model.execution(record.decision_id)
            execution_names = {
                SignalExecutionStatus.EXECUTED: "已执行",
                SignalExecutionStatus.PARTIAL: "部分执行",
                SignalExecutionStatus.IGNORED: "已忽略",
            }
            if execution is not None:
                ui.label(
                    f"执行状态：{execution_names[execution.status]} · "
                    f"记录于 {execution.recorded_at.strftime('%Y-%m-%d %H:%M')}"
                ).classes("w-full text-sm text-green-9 bg-green-1 rounded px-3 py-2")
            ui.label("策略依据与标的操作").classes("text-sm font-semibold")
            strategy_groups, residual_recommendations = _recommendations_by_strategy(
                result.strategy_decisions,
                result.recommendations,
            )
            with ui.column().classes("w-full gap-2"):
                for trace, recommendations in strategy_groups:
                    strategy_name = strategy_name_by_id.get(
                        trace.strategy_id,
                        trace.strategy_id,
                    )
                    with ui.expansion(
                        f"{strategy_name} · {_strategy_trace_text(trace)} · "
                        f"涉及 {len(recommendations)} 个标的",
                        icon="account_tree",
                    ).classes("w-full border rounded"):
                        if not recommendations:
                            ui.label("该策略本次没有产生具体标的操作。已保留策略判断供复盘。")
                        for item in recommendations:
                            choice = choice_by_code.get(str(item.instrument))
                            label = (
                                str(item.instrument)
                                if choice is None
                                else _instrument_label(choice)
                            )
                            side, side_class = _side_label(item.side)
                            with ui.row().classes("w-full items-start gap-2 py-1"):
                                ui.label(side).classes(
                                    f"min-w-12 font-semibold {side_class}"
                                )
                                with ui.column().classes("gap-0"):
                                    ui.label(
                                        f"{label} · {item.current_quantity} → "
                                        f"{item.target_quantity}"
                                    ).classes("font-medium")
                                    ui.label(
                                        _recommendation_explanation(item, trace.strategy_id)
                                    ).classes("text-sm text-grey-7")
                if residual_recommendations:
                    with ui.expansion(
                        f"组合合并与风控 · 涉及 {len(residual_recommendations)} 个标的",
                        icon="tune",
                    ).classes("w-full border rounded"):
                        ui.label(
                            "这些操作由多个策略目标合并、退出原持仓或风控换算共同形成，"
                            "不能只归因于一个策略。"
                        ).classes("text-sm text-grey-7")
                        for item in residual_recommendations:
                            choice = choice_by_code.get(str(item.instrument))
                            label = (
                                str(item.instrument)
                                if choice is None
                                else _instrument_label(choice)
                            )
                            side, side_class = _side_label(item.side)
                            ui.label(
                                f"{side} {label} · {item.current_quantity} → "
                                f"{item.target_quantity} · {_recommendation_explanation(item)}"
                            ).classes(f"text-sm {side_class}")
            rows = []
            for item in result.recommendations:
                side, _ = _side_label(item.side)
                choice = choice_by_code.get(str(item.instrument))
                executed_weight = (
                    item.reference_price * item.target_quantity / result.decision_equity
                )
                rows.append(
                    {
                        "instrument": (
                            str(item.instrument)
                            if choice is None
                            else _instrument_label(choice)
                        ),
                        "side": side,
                        "quantity": (
                            "—" if item.quantity_delta == 0 else f"{item.quantity_delta:+d}"
                        ),
                        "position": f"{item.current_quantity} → {item.target_quantity}",
                        "weight": (
                            f"{item.current_weight * 100:.1f}% → {executed_weight * 100:.1f}%"
                        ),
                        "price": f"¥{item.reference_price:,.4f}",
                        "amount": f"¥{item.gross_amount:,.2f}",
                        "reason": _reason_text(item.reason_codes),
                    }
                )
            if not rows:
                ui.label("策略当前建议保持现金，账户无需调仓。").classes(
                    "text-sm bg-blue-1 rounded px-3 py-2"
                )
            else:
                ui.table(
                    columns=[
                        {"name": "instrument", "label": "标的", "field": "instrument", "align": "left"},
                        {"name": "side", "label": "建议", "field": "side", "align": "left"},
                        {"name": "quantity", "label": "数量变化", "field": "quantity", "align": "right"},
                        {"name": "position", "label": "当前 → 目标", "field": "position", "align": "right"},
                        {"name": "weight", "label": "当前 → 执行后仓位", "field": "weight", "align": "right"},
                        {"name": "price", "label": "收盘价", "field": "price", "align": "right"},
                        {"name": "amount", "label": "预计金额", "field": "amount", "align": "right"},
                        {"name": "reason", "label": "原因", "field": "reason", "align": "left"},
                    ],
                    rows=rows,
                    row_key="instrument",
                ).classes("w-full").props("flat bordered dense")
                with ui.expansion("如何理解这份建议", icon="help_outline").classes(
                    "w-full bg-blue-1 rounded"
                ):
                    ui.label(
                        "先由每个策略根据收盘行情产生目标仓位，再按该账户设置的策略预算合并；"
                        "随后应用现金预留、风控和最小交易金额，最后按 A 股整手数量换算成买卖数量。"
                    ).classes("text-sm")
                    ui.label(
                        "“当前 → 执行后仓位”按本次决策净值和收盘价估算；实际成交价、费用以及"
                        "部分成交会造成偏差。表中的“原因”说明目标数量在换算阶段受到的约束。"
                    ).classes("text-sm text-grey-7")

                @ui.refreshable
                def comparison_view() -> None:
                    comparison = selected_comparison["result"]
                    if comparison is None or comparison.decision_id != record.decision_id:
                        return
                    ui.label("采用建议 vs 完全不采用").classes("text-sm font-semibold mt-2")
                    ui.label(
                        "使用生成信号时冻结的同一账户作为起点：一条曲线按本次建议立即调仓，"
                        "另一条保持原持仓，随后都按本地收盘价计值。它用于复盘这一次建议，"
                        "不代表连续执行所有历史信号。"
                    ).classes("text-xs text-grey-7")
                    ui.label(
                        f"截至 {comparison.points[-1].day.isoformat()}：采用收益 "
                        f"{comparison.adopted_return * 100:+.2f}% · 不采用收益 "
                        f"{comparison.ignored_return * 100:+.2f}% · "
                        f"采用相对影响 ¥{comparison.relative_impact:+,.2f}"
                    ).classes("text-sm")
                    ui.echart(
                        {
                            "animation": False,
                            "tooltip": {"trigger": "axis"},
                            "legend": {"data": ["采用建议", "完全不采用"]},
                            "xAxis": {
                                "type": "category",
                                "data": [item.day.isoformat() for item in comparison.points],
                            },
                            "yAxis": {"type": "value", "scale": True, "name": "账户净值"},
                            "dataZoom": ({"type": "inside"}, {"type": "slider"}),
                            "series": [
                                {
                                    "name": "采用建议",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": [float(item.adopted_equity) for item in comparison.points],
                                },
                                {
                                    "name": "完全不采用",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": [float(item.ignored_equity) for item in comparison.points],
                                },
                            ],
                        }
                    ).classes("w-full h-80")

                def compare_adoption() -> None:
                    try:
                        selected_comparison["result"] = model.compare_decision(
                            record.decision_id
                        )
                    except Exception as error:
                        ui.notify(_error_text(error), type="negative")
                        return
                    comparison_view.refresh()

                ui.button(
                    "对照回测：采用与不采用",
                    icon="compare_arrows",
                    on_click=compare_adoption,
                ).props("outline")
                comparison_view()

            actionable = tuple(
                item for item in result.recommendations if item.quantity_delta != 0
            )
            if execution is None and actionable:
                with ui.dialog() as execution_dialog, ui.card().classes(
                    "w-[760px] max-w-[95vw]"
                ):
                    ui.label("记录调仓执行结果").classes("text-subtitle1 font-semibold")
                    ui.label(
                        "成交后将按实际数量和价格更新共享持仓；历史信号本身保持不变。"
                    ).classes("text-sm text-grey-7")
                    execution_rows = {
                        str(item.instrument): {
                            "quantity": item.quantity_delta,
                            "price": str(
                                item.estimated_execution_price or item.reference_price
                            ),
                        }
                        for item in actionable
                    }
                    for item in actionable:
                        choice = choice_by_code.get(str(item.instrument))
                        row = execution_rows[str(item.instrument)]
                        with ui.row().classes("w-full items-end gap-3"):
                            ui.label(
                                str(item.instrument)
                                if choice is None
                                else _instrument_label(choice)
                            ).classes("min-w-72")
                            ui.number("实际成交数量", step=100).bind_value(
                                row, "quantity"
                            ).props("outlined dense").classes("w-40")
                            ui.input("实际成交价").bind_value(row, "price").props(
                                "outlined dense"
                            ).classes("w-40")
                    execution_parameters = {"fees": 0}
                    ui.number("总费用（元）", min=0, step=1).bind_value(
                        execution_parameters, "fees"
                    ).props("outlined dense").classes("w-40")

                    def submit_execution(status: SignalExecutionStatus) -> None:
                        try:
                            model.record_execution(
                                record.decision_id,
                                status,
                                tuple(
                                    (
                                        instrument,
                                        row["quantity"],
                                        row["price"],
                                    )
                                    for instrument, row in execution_rows.items()
                                    if row["quantity"] not in {0, "0", None}
                                ),
                                execution_parameters["fees"],
                            )
                        except Exception as error:
                            ui.notify(_error_text(error), type="negative")
                            return
                        execution_dialog.close()
                        ui.notify("执行结果和最新持仓已保存", type="positive")
                        ui.navigate.reload()

                    with ui.row().classes("w-full justify-end gap-2"):
                        ui.button("取消", on_click=execution_dialog.close).props("flat")
                        ui.button(
                            "保存为部分执行",
                            on_click=lambda: submit_execution(
                                SignalExecutionStatus.PARTIAL
                            ),
                        ).props("outline")
                        ui.button(
                            "保存为已执行",
                            on_click=lambda: submit_execution(
                                SignalExecutionStatus.EXECUTED
                            ),
                        )

                with ui.row().classes("gap-2"):
                    ui.button(
                        "记录执行结果",
                        icon="fact_check",
                        on_click=execution_dialog.open,
                    ).set_enabled(not freshness.stale and result.valid_until >= date.today())

                    def ignore_signal() -> None:
                        try:
                            model.record_execution(
                                record.decision_id,
                                SignalExecutionStatus.IGNORED,
                                (),
                                0,
                            )
                        except Exception as error:
                            ui.notify(_error_text(error), type="negative")
                            return
                        ui.notify("该信号已标记为忽略", type="positive")
                        ui.navigate.reload()

                    ui.button("忽略本次建议", icon="block", on_click=ignore_signal).props(
                        "flat"
                    )

    decision_view()

    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("信号历史").classes("text-subtitle1 font-semibold")
            with ui.row().classes("gap-2"):
                if state.invalid_decision_count:
                    with ui.dialog() as invalid_dialog, ui.card():
                        ui.label("清理全部无效信号记录？").classes("font-semibold")
                        ui.label(
                            "无法恢复的信号及其执行记录会被删除；账户持仓和当前行情不会删除。"
                        )

                        def clear_invalid() -> None:
                            try:
                                deleted = model.clear_invalid_decisions()
                            except Exception as error:
                                ui.notify(_error_text(error), type="negative")
                                return
                            invalid_dialog.close()
                            ui.notify(f"已清理 {deleted} 条无效信号", type="positive")
                            ui.navigate.reload()

                        with ui.row().classes("w-full justify-end gap-2"):
                            ui.button("取消", on_click=invalid_dialog.close).props("flat")
                            ui.button("确认清理", on_click=clear_invalid, color="negative")

                    ui.button(
                        f"清理无效记录（{state.invalid_decision_count}）",
                        icon="cleaning_services",
                        on_click=invalid_dialog.open,
                    ).props("outline")
                if state.decision_history:
                    with ui.dialog() as clear_dialog, ui.card():
                        ui.label("清理当前账户的全部信号历史？").classes("font-semibold")
                        ui.label(
                            "仅清理未采用或已忽略的信号；已执行、部分执行的信号作为审计记录永久保留。"
                        )

                        def clear_all() -> None:
                            try:
                                deleted = model.clear_decisions()
                            except Exception as error:
                                ui.notify(_error_text(error), type="negative")
                                return
                            clear_dialog.close()
                            ui.notify(f"已清理 {deleted} 条信号", type="positive")
                            ui.navigate.reload()

                        with ui.row().classes("w-full justify-end gap-2"):
                            ui.button("取消", on_click=clear_dialog.close).props("flat")
                            ui.button("确认清理", on_click=clear_all, color="negative")
                    ui.button(
                        "清理全部历史",
                        icon="delete_sweep",
                        on_click=clear_dialog.open,
                    ).props("flat color=negative")
        if not state.decision_history:
            ui.label("暂无历史记录").classes("text-sm text-grey-6")
        for record in state.decision_history[:20]:
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(
                    f"{record.result.decision_date.isoformat()} · {record.decision_id} · "
                    f"建议 {len(record.result.recommendations)} 个标的"
                ).classes("text-sm")

                def show_history(decision_id: str = record.decision_id) -> None:
                    loaded = model.decision(decision_id)
                    if loaded is not None:
                        selected_decision["record"] = loaded
                        selected_comparison["result"] = None
                        decision_view.refresh()

                ui.button(icon="visibility", on_click=show_history).props("flat round")

                history_execution = model.execution(record.decision_id)
                if not _decision_deletable(
                    None if history_execution is None else history_execution.status
                ):
                    ui.icon("verified", color="positive").tooltip(
                        "已采用的信号属于账户执行审计，不允许清理"
                    )
                    continue

                with ui.dialog() as delete_dialog, ui.card():
                    ui.label("清理这条未采用信号？").classes("font-semibold")
                    ui.label("信号和关联执行记录会被删除；账户持仓快照不会删除。")

                    def delete_history(
                        decision_id: str = record.decision_id,
                        dialog: Dialog = delete_dialog,
                    ) -> None:
                        try:
                            model.delete_decision(decision_id)
                        except Exception as error:
                            ui.notify(_error_text(error), type="negative")
                            return
                        dialog.close()
                        ui.notify("信号历史已清理", type="positive")
                        ui.navigate.reload()

                    with ui.row().classes("w-full justify-end gap-2"):
                        ui.button("取消", on_click=delete_dialog.close).props("flat")
                        ui.button("确认清理", on_click=delete_history, color="negative")

                ui.button(icon="delete_outline", on_click=delete_dialog.open).props(
                    "flat round color=negative aria-label=清理该信号"
                )
