from __future__ import annotations

from decimal import Decimal
from typing import Any

from nicegui import ui
from compass.ui.navigation import draft_url
from compass.ui.edit_guard import EditGuard
from compass.ui.condition_builder import INDICATORS, OPERATORS, build_condition
from pydantic import ValidationError

from compass.domain.market import InstrumentId
from compass.strategies.rule_document import (
    RuleExecution,
    RuleSide,
    StrategyRule,
    StrategyRuleDocument,
    document_required_fields,
)
from compass.strategies.rule_dsl import DslAction, DslVariable
from compass.ui.pages.strategies import (
    StrategyPageError,
    StrategyPageModel,
    render_strategies_page,
)


def _notify_error(error: Exception, fallback: str) -> None:
    code = getattr(error, "code", fallback)
    messages = {
        "STRATEGY_DRAFT_UNKNOWN": "草稿已删除或地址失效，请从策略库重新打开。",
        "STRATEGY_POOL_UNAVAILABLE": "标的池暂不可用，请先检查标的池与行情。",
        "STRATEGY_DRAFT_CREATE_FAILED": "草稿未创建，请检查标的池后重试。",
    }
    ui.notify(messages.get(code, f"操作未完成，请检查输入或查看日志。错误编号：{code}"), type="negative")


def render_strategy_library_page(model: StrategyPageModel | None) -> None:
    if model is None:
        ui.label("策略服务未配置。")
        return
    try:
        state = model.state()
        drafts = model.rule_drafts()
    except Exception as error:
        _notify_error(error, "STRATEGY_STATE_UNAVAILABLE")
        return
    ui.label("策略库管理草稿和已发布版本；回测与参数调优保持独立。 ").classes(
        "text-sm text-slate-600"
    )
    with ui.row().classes("w-full items-end justify-between gap-3"):
        ui.label("策略库").classes("text-xl font-semibold")

        def create_rule_strategy() -> None:
            if not state.pools:
                ui.notify("请先在标的池中启用至少一个可交易标的。", type="warning")
                return
            try:
                created = model.new_rule_draft(state.pools[0].watchlist_id)
                ui.navigate.to(draft_url("/strategies/editor", created.draft_id))
            except Exception as error:
                _notify_error(error, "STRATEGY_DRAFT_CREATE_FAILED")

        with ui.row().classes("gap-2"):
            ui.button(
                "内置模板与参数调优",
                icon="tune",
                on_click=lambda: ui.navigate.to("/strategies/templates"),
            ).props("outline")
            ui.button("新建规则策略", icon="add", on_click=create_rule_strategy).props(
                "color=primary"
            )

    ui.label("草稿").classes("text-lg font-semibold mt-4")
    if not drafts:
        ui.label("暂无草稿。新建策略后可以反复保存，不会影响账户正在使用的版本。 ").classes(
            "text-sm text-slate-500"
        )
    for draft in drafts:
        with ui.card().classes("w-full border border-amber-200 shadow-none"):
            with ui.row().classes("w-full justify-between items-start gap-3"):
                with ui.column().classes("gap-1"):
                    ui.label(draft.document.name).classes("font-semibold")
                    source = "新策略" if draft.source_instance_id is None else "修改已发布版本"
                    ui.label(
                        f"{source} · {len(draft.document.rules)} 条规则 · "
                        f"{len(draft.document.variables)} 个变量 · "
                        f"更新于 {draft.updated_at.strftime('%Y-%m-%d %H:%M')}"
                    ).classes("text-sm text-slate-600")
                actions = ui.row().classes("gap-1")

            def edit(selected_id: str = draft.draft_id) -> None:
                ui.navigate.to(draft_url("/strategies/editor", selected_id))

            def remove(selected_id: str = draft.draft_id) -> None:
                try:
                    model.delete_rule_draft(selected_id)
                    ui.navigate.to("/strategies")
                except Exception as error:
                    _notify_error(error, "STRATEGY_DRAFT_DELETE_FAILED")

            with actions:
                ui.button(icon="edit", on_click=edit).props(
                    "flat round aria-label=编辑草稿"
                ).tooltip("编辑")
                ui.button(icon="delete_outline", on_click=remove).props(
                    "flat round color=negative aria-label=删除草稿"
                ).tooltip("删除")

    deleted = model.deleted_rule_drafts()
    if deleted:
        with ui.expansion(f"草稿回收站（{len(deleted)}）", icon="restore_from_trash"):
            for item in deleted:
                with ui.row().classes("items-center"):
                    ui.label(item.document.name)

                    def restore(selected_id: str = item.draft_id) -> None:
                        try:
                            restored = model.restore_rule_draft(selected_id)
                            ui.navigate.to(draft_url("/strategies/editor", restored.draft_id))
                        except Exception as error:
                            _notify_error(error, "STRATEGY_DRAFT_RESTORE_FAILED")

                    ui.button("恢复", icon="restore", on_click=restore).props("outline")

    ui.label("已发布策略").classes("text-lg font-semibold mt-4")
    latest = tuple(
        item
        for item in state.instances
        if item.version
        == max(
            candidate.version
            for candidate in state.instances
            if candidate.lineage_id == item.lineage_id
        )
    )
    if not latest:
        ui.label("暂无已发布策略。 ").classes("text-sm text-slate-500")
    for instance in latest:
        with ui.card().classes("w-full border border-slate-200 shadow-none"):
            with ui.row().classes("w-full justify-between items-start gap-3"):
                with ui.column().classes("gap-1"):
                    ui.label(instance.name).classes("font-semibold")
                    ui.label(
                        f"v{instance.version} · {instance.strategy_type} · "
                        f"{'已启用' if instance.enabled else '已停用'} · {instance.frequency.value}"
                    ).classes("text-sm text-slate-600")
                actions = ui.row().classes("gap-1")

            def edit_published(selected_id: str = instance.instance_id) -> None:
                try:
                    created = model.edit_rule_draft(selected_id)
                    ui.navigate.to(draft_url("/strategies/editor", created.draft_id))
                except Exception as error:
                    _notify_error(error, "STRATEGY_RULE_EDITOR_UNSUPPORTED")

            def copy_published(selected_id: str = instance.instance_id) -> None:
                try:
                    model.copy(selected_id)
                    ui.navigate.to("/strategies")
                except Exception as error:
                    _notify_error(error, "STRATEGY_COPY_FAILED")

            def disable_published(selected_id: str = instance.instance_id) -> None:
                try:
                    model.disable(selected_id)
                    ui.navigate.to("/strategies")
                except Exception as error:
                    _notify_error(error, "STRATEGY_DISABLE_FAILED")

            def delete_published(selected_id: str = instance.instance_id) -> None:
                try:
                    model.delete(selected_id)
                    ui.navigate.to("/strategies")
                except Exception as error:
                    _notify_error(error, "STRATEGY_DELETE_FAILED")

            with actions:
                if instance.strategy_type == "rule_dsl":
                    ui.button(icon="edit", on_click=edit_published).props(
                        "flat round aria-label=编辑并创建新版本"
                    ).tooltip("编辑为草稿")
                ui.button(icon="content_copy", on_click=copy_published).props(
                    "flat round aria-label=复制策略"
                ).tooltip("复制")
                if instance.enabled:
                    ui.button(icon="pause_circle_outline", on_click=disable_published).props(
                        "flat round aria-label=停用策略"
                    ).tooltip("停用")
                ui.button(icon="delete_outline", on_click=delete_published).props(
                    "flat round color=negative aria-label=删除策略"
                ).tooltip("删除")


def render_strategy_templates_page(model: StrategyPageModel | None) -> None:
    """Keep classic templates and optimizer available beside the rule editor."""
    ui.label("这里保留经典策略模板、参数表单和实验调优；自定义买卖逻辑请使用规则编辑器。").classes(
        "text-sm text-slate-600 mb-3"
    )
    render_strategies_page(model)


def render_rule_editor_page(
    model: StrategyPageModel | None, *, draft_id: str | None = None
) -> None:
    if model is None:
        ui.label("策略服务未配置。")
        return
    try:
        draft = model.active_rule_draft(draft_id) if draft_id else None
    except Exception as error:
        _notify_error(error, "STRATEGY_DRAFT_UNAVAILABLE")
        return
    if draft is None:
        ui.label("没有可编辑的策略草稿。 ")
        ui.button("返回策略库", on_click=lambda: ui.navigate.to("/strategies"))
        return
    document = draft.document
    guard = EditGuard()
    saved_draft = [draft]
    with ui.row().classes("w-full justify-between items-start gap-3"):
        with ui.column().classes("gap-1"):
            ui.label(document.name).classes("text-xl font-semibold")
            ui.label("未发布草稿 · 日线收盘确认 · 下一交易时点执行").classes(
                "text-sm text-slate-600"
            )
        ui.button(
            "返回策略库", icon="arrow_back", on_click=lambda: guard.navigate("/strategies")
        ).props("flat")

    name_input = ui.input("策略名称", value=document.name).classes("w-full")
    description_input = ui.textarea("策略说明", value=document.description).classes("w-full")
    execution_select = ui.select(
        {
            RuleExecution.NEXT_OPEN.value: "下一交易日开盘",
            RuleExecution.NEXT_CLOSE.value: "下一交易日收盘",
        },
        value=document.execute.value,
        label="执行时点",
    )
    ui.label("规则").classes("text-lg font-semibold mt-3")
    rule_controls: list[dict[str, Any]] = []
    rules_container = ui.column().classes("w-full gap-3")
    rule_counter = [len(document.rules)]

    def add_rule(rule: StrategyRule | None = None, *, side_value: RuleSide = RuleSide.BUY) -> None:
        rule_counter[0] += 1
        selected_side = rule.side if rule is not None else side_value
        selected_action = (
            rule.action
            if rule is not None and rule.action is not None
            else DslAction.TARGET_WEIGHT
            if selected_side is RuleSide.BUY
            else DslAction.SELL_ALL
        )
        rule_id = rule.rule_id if rule is not None else f"rule_{rule_counter[0]}"
        with rules_container:
            card = ui.card().classes("w-full border border-slate-200 shadow-none")
            with card:
                with ui.row().classes("w-full gap-3 items-end"):
                    name = ui.input(
                        "规则名称",
                        value=rule.name
                        if rule is not None
                        else ("新增买入条件" if selected_side is RuleSide.BUY else "新增卖出条件"),
                    )
                    action = ui.select(
                        {
                            DslAction.TARGET_WEIGHT.value: "调整到目标仓位",
                            DslAction.INCREASE_BY.value: "增加仓位",
                            DslAction.REDUCE_TO.value: "减仓到",
                            DslAction.SELL_ALL.value: "全部卖出",
                            DslAction.HOLD.value: "保持仓位",
                        },
                        value=selected_action.value,
                        label="动作",
                    )
                    priority = ui.number(
                        "优先级",
                        value=rule.priority if rule is not None else 100,
                        min=1,
                        max=10_000,
                        step=10,
                    )
                    target = ui.number(
                        "动作数值（%，调仓/加减仓时填写）",
                        value=(
                            float(rule.target_weight * 100)
                            if rule is not None and rule.target_weight is not None
                            else (100 if selected_action is DslAction.TARGET_WEIGHT else None)
                        ),
                        min=1,
                        max=100,
                        step=5,
                    )
                    remove_button = (
                        ui.button(icon="delete_outline")
                        .props("flat round color=negative aria-label=删除规则")
                        .tooltip("删除规则")
                    )
                expression = ui.textarea(
                    "触发条件",
                    value=rule.expression if rule is not None else "close > sma(close, 20)",
                ).classes("w-full font-mono")
                with ui.expansion("用表单组合条件", icon="build").classes("w-full"):
                    with ui.row().classes("items-end gap-2"):
                        left = ui.select(INDICATORS, value="close", label="指标")
                        operator = ui.select(OPERATORS, value=">", label="比较方式")
                        right = ui.select(
                            {**INDICATORS, "number": "填写数值"},
                            value="sma(close, 20)", label="比较对象",
                        )
                        number = ui.input("数值", value="0.05").bind_visibility_from(
                            right, "value", backward=lambda value: value == "number"
                        )
                        combine = ui.select(
                            {"replace": "替换当前条件", "and": "并且满足", "or": "或者满足"},
                            value="replace", label="组合方式",
                        )

                    def apply_condition() -> None:
                        try:
                            condition = build_condition(
                                str(left.value), str(operator.value),
                                str(number.value) if right.value == "number" else str(right.value),
                            )
                            current = str(expression.value or "").strip()
                            expression.set_value(
                                f"({current}) {combine.value} ({condition})"
                                if current and combine.value in {"and", "or"} else condition
                            )
                            guard.mark()
                        except ValueError as error:
                            ui.notify(str(error), type="negative")

                    ui.button("应用条件", on_click=apply_condition).props("outline")
                    ui.label("收益率使用小数：0.05 表示 5%。高级表达式可继续在上方编辑。").classes("text-xs text-slate-500")
                ui.label(
                    "可用：open/high/low/close/volume/amount、sma、rsi、pct_change、"
                    "highest、lowest、cross_above、cross_below、has_position、holding_days、"
                    "position_return、position_drawdown 以及 and/or/not。"
                ).classes("text-xs text-slate-500")
        controls = {
            "rule_id": rule_id,
            "name": name,
            "action": action,
            "default_side": selected_side,
            "priority": priority,
            "target": target,
            "expression": expression,
        }
        rule_controls.append(controls)
        if rule is None:
            guard.mark()

        def remove_rule() -> None:
            if controls in rule_controls:
                rule_controls.remove(controls)
            card.delete()
            guard.mark()

        remove_button.on_click(remove_rule)

    for existing_rule in document.rules:
        add_rule(existing_rule)
    with ui.row().classes("gap-2"):
        ui.button(
            "添加买入规则",
            icon="add",
            on_click=lambda: add_rule(side_value=RuleSide.BUY),
        ).props("outline")
        ui.button(
            "添加卖出规则",
            icon="add",
            on_click=lambda: add_rule(side_value=RuleSide.SELL),
        ).props("outline")

    ui.label("导出变量").classes("text-lg font-semibold mt-3")
    variable_controls: list[dict[str, Any]] = []
    variables_container = ui.column().classes("w-full gap-2")
    variable_counter = [len(document.variables)]

    def add_variable(variable: DslVariable | None = None) -> None:
        variable_counter[0] += 1
        with variables_container:
            row = ui.row().classes("w-full gap-3 items-end")
            with row:
                name = ui.input(
                    "变量",
                    value=variable.name
                    if variable is not None
                    else f"variable_{variable_counter[0]}",
                )
                value = ui.number("当前值", value=float(variable.value) if variable else 20)
                minimum = ui.number("最小值", value=float(variable.minimum) if variable else 5)
                maximum = ui.number("最大值", value=float(variable.maximum) if variable else 100)
                step = ui.number(
                    "步长",
                    value=float(variable.step) if variable else 5,
                    min=0.000001,
                )
                optimize = ui.checkbox(
                    "参与优化",
                    value=variable.optimize if variable is not None else True,
                )
                remove_button = (
                    ui.button(icon="delete_outline")
                    .props("flat round color=negative aria-label=删除变量")
                    .tooltip("删除变量")
                )
        controls = {
            "name": name,
            "value": value,
            "minimum": minimum,
            "maximum": maximum,
            "step": step,
            "optimize": optimize,
        }
        variable_controls.append(controls)
        if variable is None:
            guard.mark()

        def remove_variable() -> None:
            if controls in variable_controls:
                variable_controls.remove(controls)
            row.delete()
            guard.mark()

        remove_button.on_click(remove_variable)

    for existing_variable in document.variables:
        add_variable(existing_variable)
    ui.button("添加变量", icon="add", on_click=lambda: add_variable()).props("outline")
    feedback = ui.label("").classes("text-sm text-red-700")

    def build_document() -> StrategyRuleDocument:
        variables = tuple(
            DslVariable(
                name=str(item["name"].value),
                value=Decimal(str(item["value"].value)),
                minimum=Decimal(str(item["minimum"].value)),
                maximum=Decimal(str(item["maximum"].value)),
                step=Decimal(str(item["step"].value)),
                optimize=bool(item["optimize"].value),
            )
            for item in variable_controls
        )
        rules = []
        for item in rule_controls:
            selected_action = DslAction(str(item["action"].value))
            selected_side = (
                RuleSide.BUY
                if selected_action in {DslAction.TARGET_WEIGHT, DslAction.INCREASE_BY}
                else RuleSide.SELL
                if selected_action in {DslAction.REDUCE_TO, DslAction.SELL_ALL}
                else item["default_side"]
            )
            target_value = item["target"].value
            needs_value = selected_action in {
                DslAction.TARGET_WEIGHT,
                DslAction.INCREASE_BY,
                DslAction.REDUCE_TO,
            }
            if needs_value and target_value is None:
                raise ValueError("调仓、加仓或减仓动作必须填写仓位数值")
            rules.append(
                StrategyRule(
                    rule_id=item["rule_id"],
                    name=str(item["name"].value),
                    side=selected_side,
                    priority=int(item["priority"].value),
                    expression=str(item["expression"].value),
                    action=selected_action,
                    target_weight=(
                        Decimal(str(target_value)) / Decimal("100")
                        if needs_value
                        else None
                    ),
                )
            )
        return StrategyRuleDocument(
            name=str(name_input.value),
            description=str(description_input.value),
            variables=variables,
            rules=tuple(rules),
            execute=RuleExecution(str(execution_select.value)),
        )

    async def save(next_page: str | None = None, *, automatic: bool = False) -> None:
        try:
            updated = build_document()
            if updated != saved_draft[0].document:
                saved_draft[0] = model.save_rule_draft(
                    draft.draft_id, updated, expected=saved_draft[0]
                )
            await guard.clear()
            feedback.set_text("已自动保存" if automatic else "草稿已保存")
            feedback.classes(replace="text-sm text-emerald-700")
            if not automatic:
                ui.notify("草稿已保存", type="positive")
            if next_page is not None:
                ui.navigate.to(draft_url(next_page, draft.draft_id))
        except (StrategyPageError, ValidationError, TypeError, ValueError) as error:
            guard.mark()
            feedback.classes(replace="text-sm text-red-700")
            if isinstance(error, ValidationError):
                details = []
                for issue in error.errors()[:3]:
                    location = ".".join(str(item) for item in issue["loc"]) or "策略"
                    details.append(f"{location}：{issue['msg']}")
                feedback.set_text("尚未保存，请修正名称、条件或动作数值。" + "；".join(details))
            else:
                feedback.set_text(f"尚未保存：{getattr(error, 'code', str(error))}")

    ui.timer(3.0, lambda: save(automatic=True))

    with ui.row().classes("w-full justify-end gap-2 mt-3"):
        ui.button("保存草稿", icon="save", on_click=lambda: save())
        ui.button(
            "验证并预览",
            icon="query_stats",
            on_click=lambda: save("/strategies/preview"),
        ).props("color=primary")


def render_rule_preview_page(
    model: StrategyPageModel | None, *, draft_id: str | None = None
) -> None:
    if model is None:
        ui.label("策略服务未配置。")
        return
    try:
        draft = model.active_rule_draft(draft_id) if draft_id else None
    except StrategyPageError as error:
        _notify_error(error, "STRATEGY_DRAFT_UNKNOWN")
        ui.link("返回策略库", "/strategies")
        return
    if draft is None:
        ui.label("没有可预览的草稿。 ")
        return
    try:
        instruments = model.rule_draft_instruments(draft.draft_id)
    except Exception as error:
        _notify_error(error, "STRATEGY_POOL_UNAVAILABLE")
        return
    ui.label("信号预览").classes("text-xl font-semibold")
    ui.label("这里检查规则何时命中，不计算策略收益；收益与基准比较仍在策略回测。 ").classes(
        "text-sm text-slate-600"
    )
    instrument_select = ui.select(
        {str(item): str(item) for item in instruments},
        value=str(instruments[0]),
        label="预览标的",
    )
    result_area = ui.column().classes("w-full")
    if draft.document.execute is RuleExecution.NEXT_CLOSE:
        ui.label(
            "当前策略计划在下一交易日收盘执行；正式回测页面的执行模型需要选择相同设置。"
        ).classes("text-sm text-amber-700")

    def run_preview() -> None:
        result_area.clear()
        try:
            result = model.preview_rule_draft(
                draft.draft_id,
                InstrumentId.parse(str(instrument_select.value)),
            )
        except Exception as error:
            with result_area:
                ui.label(f"预览失败：{getattr(error, 'code', 'STRATEGY_PREVIEW_FAILED')}").classes(
                    "text-red-700"
                )
            return
        with result_area:
            ui.label(
                f"{result.first_day} 至 {result.last_day} · {result.bars} 根日线 · "
                f"{len(result.signals)} 次目标仓位变化"
            ).classes("text-sm text-slate-600")
            if not result.signals:
                ui.label("所选历史中没有产生目标仓位变化。 ").classes("text-amber-700")
            else:
                rows = [
                    {
                        "day": str(item.day),
                        "side": {
                            DslAction.TARGET_WEIGHT: "B 调整仓位",
                            DslAction.INCREASE_BY: "B 加仓",
                            DslAction.REDUCE_TO: "S 减仓",
                            DslAction.SELL_ALL: "S 全部卖出",
                            DslAction.HOLD: "— 保持",
                        }[item.action],
                        "rule": item.rule_name,
                        "overridden": "、".join(item.overridden_rule_ids) or "—",
                        "close": str(item.close),
                        "target": f"{item.target_weight * 100}%",
                        "execution": draft.document.execute.value,
                    }
                    for item in reversed(result.signals)
                ]
                ui.table(
                    columns=[
                        {"name": "day", "label": "日期", "field": "day"},
                        {"name": "side", "label": "信号", "field": "side"},
                        {"name": "rule", "label": "命中规则", "field": "rule"},
                        {"name": "overridden", "label": "被覆盖规则", "field": "overridden"},
                        {"name": "close", "label": "收盘价", "field": "close"},
                        {"name": "target", "label": "目标仓位", "field": "target"},
                        {"name": "execution", "label": "计划执行", "field": "execution"},
                    ],
                    rows=rows,
                    row_key="day",
                ).classes("w-full")

    with ui.row().classes("items-end gap-3"):
        ui.button("运行预览", icon="play_arrow", on_click=run_preview).props("color=primary")
        ui.button("返回编辑", on_click=lambda: ui.navigate.to(
            draft_url("/strategies/editor", draft.draft_id)
        )).props("flat")
        ui.button("进入发布检查", on_click=lambda: ui.navigate.to(
            draft_url("/strategies/release", draft.draft_id)
        ))
    run_preview()


def render_rule_release_page(
    model: StrategyPageModel | None, *, draft_id: str | None = None
) -> None:
    if model is None:
        ui.label("策略服务未配置。")
        return
    try:
        draft = model.active_rule_draft(draft_id) if draft_id else None
    except StrategyPageError as error:
        _notify_error(error, "STRATEGY_DRAFT_UNKNOWN")
        ui.link("返回策略库", "/strategies")
        return
    if draft is None:
        ui.label("没有可发布的草稿。 ")
        return
    document = draft.document
    ui.label("验证与发布").classes("text-xl font-semibold")
    ui.label("发布后形成不可变版本；不会自动替换账户当前引用的策略版本。 ").classes(
        "text-sm text-slate-600"
    )
    with ui.card().classes("w-full border border-slate-200 shadow-none"):
        ui.label("发布门禁").classes("font-semibold")
        ui.label("✓ 规则语法与类型检查通过").classes("text-sm text-emerald-700")
        ui.label("✓ 使用受限 AST 执行，不调用 Python eval").classes("text-sm text-emerald-700")
        ui.label(f"✓ 最少历史：{document.minimum_history} 个交易日").classes(
            "text-sm text-emerald-700"
        )
        ui.label(f"✓ 所需行情字段：{'、'.join(document_required_fields(document))}").classes(
            "text-sm text-emerald-700"
        )
        ui.label(
            f"✓ 可优化变量组合：{document.optimization_trial_count} 组（正式调优另行启动）"
        ).classes("text-sm text-emerald-700")
        ui.label(
            f"内容哈希：{document.document_hash[:16]}… · "
            f"{len(document.rules)} 条规则 · {len(document.variables)} 个变量"
        ).classes("text-xs text-slate-500")
    with ui.card().classes("w-full border border-indigo-200 bg-indigo-50 shadow-none"):
        ui.label("执行语义").classes("font-semibold")
        ui.label(
            "同一交易日先按优先级从高到低处理；同优先级卖出优先。策略只生成目标仓位，"
            "实际成交仍受账户资金、停牌、涨跌停、费用和风控约束。"
        ).classes("text-sm text-slate-700")
    feedback = ui.label("").classes("text-sm text-red-700")

    def publish() -> None:
        try:
            instance = model.publish_rule_draft(draft.draft_id, expected=draft)
            ui.notify(f"已发布：{instance.name} v{instance.version}", type="positive")
            ui.navigate.to("/strategies")
        except Exception as error:
            feedback.set_text(f"发布失败：{getattr(error, 'code', 'STRATEGY_PUBLISH_FAILED')}")

    with ui.row().classes("w-full justify-end gap-2"):
        ui.button("返回编辑", on_click=lambda: ui.navigate.to(
            draft_url("/strategies/editor", draft.draft_id)
        ))
        ui.button("发布策略版本", icon="publish", on_click=publish).props("color=primary")
