from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, DecimalException
from typing import Protocol, TypeVar

from compass.data.base import default_instrument_type
from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import AccountSnapshot, Position
from compass.storage.account_repository import StoredAccountSnapshot
from compass.services.safe_display import frozen_errors, stable_code

from nicegui import ui


InstrumentResolver = Callable[[InstrumentId], AssetType]
Today = Callable[[], date]


T = TypeVar("T")


class AccountPageError(RuntimeError):
    """A stable, secret-safe account gateway boundary failure."""

    def __init__(self, code: str) -> None:
        self.code = stable_code(code, label="account page error code")
        super().__init__(self.code)


def _gateway_call(code: str, operation: Callable[[], T]) -> T:
    failed = False
    result: T | None = None
    try:
        result = operation()
    except Exception:
        failed = True
    if failed:
        raise AccountPageError(code)
    assert result is not None
    return result


def _decimal_text(
    value: object,
    *,
    text_error: str,
    value_error: str,
) -> tuple[Decimal | None, str | None]:
    if type(value) is not str:
        return None, text_error
    assert isinstance(value, str)
    if not value or value != value.strip():
        return None, text_error
    try:
        parsed = Decimal(value)
    except DecimalException:
        return None, value_error
    if not parsed.is_finite() or parsed < 0:
        return None, value_error
    return parsed, None


def _integer(value: object, message: str) -> tuple[int | None, str | None]:
    if type(value) is int:
        assert isinstance(value, int)
        return (value, None) if value >= 0 else (None, message)
    if type(value) is str:
        assert isinstance(value, str)
        if value == "0" or (value.isascii() and value.isdigit() and not value.startswith("0")):
            return int(value), None
    return None, message


@dataclass(frozen=True, slots=True)
class PositionFormModel:
    instrument: object
    asset_type: object
    quantity: object
    available_quantity: object
    average_cost: object
    mark_price: object


@dataclass(frozen=True, slots=True)
class AccountValidationResult:
    errors: Mapping[str, str]
    snapshot: AccountSnapshot | None

    def __post_init__(self) -> None:
        errors = frozen_errors(self.errors)
        if self.snapshot is not None and type(self.snapshot) is not AccountSnapshot:
            raise TypeError("snapshot must be an exact AccountSnapshot or None")
        if (self.snapshot is None) == (not bool(errors)):
            raise ValueError("validation must contain either errors or a snapshot")
        object.__setattr__(self, "errors", errors)


@dataclass(frozen=True, slots=True)
class AccountSaveResult:
    errors: Mapping[str, str]
    saved: StoredAccountSnapshot | None

    def __post_init__(self) -> None:
        errors = frozen_errors(self.errors)
        if self.saved is not None and type(self.saved) is not StoredAccountSnapshot:
            raise TypeError("saved must be an exact StoredAccountSnapshot or None")
        if (self.saved is None) == (not bool(errors)):
            raise ValueError("save result must contain either errors or a saved record")
        object.__setattr__(self, "errors", errors)


@dataclass(frozen=True, slots=True)
class AccountPageState:
    latest: StoredAccountSnapshot | None
    history: tuple[StoredAccountSnapshot, ...]

    def __post_init__(self) -> None:
        history = tuple(self.history)
        if any(type(record) is not StoredAccountSnapshot for record in history):
            raise TypeError("account history must contain exact stored snapshots")
        row_ids = tuple(record.row_id for record in history)
        if row_ids != tuple(sorted(row_ids)) or len(set(row_ids)) != len(row_ids):
            raise ValueError("account history must have unique ascending row ids")
        account_ids = {record.account_id for record in history}
        if len(account_ids) > 1:
            raise ValueError("account history must belong to one account")
        if self.latest is not None and type(self.latest) is not StoredAccountSnapshot:
            raise TypeError("latest must be an exact stored snapshot or None")
        expected_latest = history[-1] if history else None
        if self.latest != expected_latest:
            raise ValueError("latest must match the final account history record")
        object.__setattr__(self, "history", history)


@dataclass(frozen=True, slots=True)
class AccountFormModel:
    as_of: object
    cash: object
    positions: Sequence[PositionFormModel]
    resolver: InstrumentResolver = default_instrument_type
    today: Today = date.today

    def __post_init__(self) -> None:
        if not callable(self.resolver):
            raise TypeError("resolver must be callable")
        if not callable(self.today):
            raise TypeError("today must be callable")
        if not isinstance(self.positions, Sequence) or isinstance(
            self.positions, (str, bytes, bytearray)
        ):
            raise TypeError("positions must be a sequence")
        positions = tuple(self.positions)
        if any(type(position) is not PositionFormModel for position in positions):
            raise TypeError("positions must contain PositionFormModel values")
        object.__setattr__(self, "positions", positions)

    def validate(self) -> AccountValidationResult:
        errors: dict[str, str] = {}
        parsed_date: date | None = None
        if type(self.as_of) is str:
            assert isinstance(self.as_of, str)
            try:
                candidate = date.fromisoformat(self.as_of)
            except ValueError:
                candidate = None
            if candidate is not None and candidate.isoformat() == self.as_of:
                parsed_date = candidate
        if parsed_date is None:
            errors["as_of"] = "日期必须使用 YYYY-MM-DD 格式"
        else:
            try:
                today = self.today()
                if type(today) is not date:
                    raise TypeError
            except Exception:
                errors["as_of"] = "无法确认当前日期"
            else:
                if parsed_date > today:
                    errors["as_of"] = "快照日期不能晚于今天"

        parsed_cash, cash_error = _decimal_text(
            self.cash,
            text_error="现金必须使用精确数字文本",
            value_error="现金必须是有限非负数",
        )
        if cash_error is not None:
            errors["cash"] = cash_error
        elif parsed_cash is not None:
            exponent = parsed_cash.as_tuple().exponent
            if isinstance(exponent, int) and exponent < -2:
                errors["cash"] = "现金最多保留两位小数"

        parsed_positions: list[Position] = []
        symbols: set[InstrumentId] = set()
        for index, form in enumerate(self.positions):
            position = self._position(index, form, errors)
            if position is None:
                continue
            if position.instrument in symbols:
                errors[f"positions.{index}.instrument"] = "持仓标的不能重复"
                continue
            symbols.add(position.instrument)
            parsed_positions.append(position)

        if errors or parsed_date is None or parsed_cash is None:
            return AccountValidationResult(errors, None)
        try:
            snapshot = AccountSnapshot(parsed_date, parsed_cash, tuple(parsed_positions))
        except (TypeError, ValueError, ArithmeticError):
            return AccountValidationResult({"form": "账户数据超出支持范围"}, None)
        return AccountValidationResult({}, snapshot)

    def _position(
        self,
        index: int,
        form: PositionFormModel,
        errors: dict[str, str],
    ) -> Position | None:
        prefix = f"positions.{index}"
        instrument: InstrumentId | None = None
        if type(form.instrument) is str:
            assert isinstance(form.instrument, str)
            try:
                candidate = InstrumentId.parse(form.instrument)
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and str(candidate) == form.instrument:
                instrument = candidate
        if instrument is None:
            errors[f"{prefix}.instrument"] = "代码必须使用 SSE.XXXXXX 或 SZSE.XXXXXX"

        declared_type: AssetType | None = None
        if type(form.asset_type) is str:
            try:
                declared_type = AssetType(form.asset_type)
            except ValueError:
                pass
        if declared_type is None:
            errors[f"{prefix}.asset_type"] = "资产类型必须是 stock 或 etf"

        if instrument is not None:
            try:
                resolved_type = self.resolver(instrument)
                if type(resolved_type) is not AssetType:
                    raise ValueError
            except Exception:
                errors[f"{prefix}.instrument"] = "代码不属于支持的股票或 ETF 家族"
            else:
                if declared_type is not None and declared_type is not resolved_type:
                    errors[f"{prefix}.asset_type"] = "资产类型与代码不匹配"

        quantity, quantity_error = _integer(form.quantity, "持仓数量必须是非负整数")
        if quantity_error is not None:
            errors[f"{prefix}.quantity"] = quantity_error
        available, available_error = _integer(
            form.available_quantity, "可用数量必须是非负整数"
        )
        if available_error is not None:
            errors[f"{prefix}.available_quantity"] = available_error
        elif quantity is not None and available is not None and available > quantity:
            errors[f"{prefix}.available_quantity"] = "可用数量不能大于持仓数量"

        average_cost, average_error = _decimal_text(
            form.average_cost,
            text_error="平均成本必须使用精确数字文本",
            value_error="平均成本必须是有限非负数",
        )
        if average_error is not None:
            errors[f"{prefix}.average_cost"] = average_error
        mark_price, mark_error = _decimal_text(
            form.mark_price,
            text_error="标记价格必须使用精确数字文本",
            value_error="标记价格必须是有限非负数",
        )
        if mark_error is not None:
            errors[f"{prefix}.mark_price"] = mark_error

        position_fields = (
            instrument,
            declared_type,
            quantity,
            available,
            average_cost,
            mark_price,
        )
        if any(value is None for value in position_fields):
            return None
        if any(key.startswith(f"{prefix}.") for key in errors):
            return None
        assert instrument is not None
        assert quantity is not None
        assert available is not None
        assert average_cost is not None
        assert mark_price is not None
        try:
            return Position(instrument, quantity, available, average_cost, mark_price)
        except (TypeError, ValueError, ArithmeticError):
            errors[f"{prefix}.instrument"] = "持仓数据超出支持范围"
            return None


class AccountGateway(Protocol):
    def save(self, snapshot: AccountSnapshot) -> StoredAccountSnapshot: ...

    def latest(self) -> StoredAccountSnapshot | None: ...

    def history(self) -> tuple[StoredAccountSnapshot, ...]: ...


class AccountPageModel:
    def __init__(self, gateway: AccountGateway) -> None:
        self._gateway = gateway

    def state(self) -> AccountPageState:
        return _gateway_call(
            "ACCOUNT_STATE_UNAVAILABLE",
            lambda: AccountPageState(
                self._gateway.latest(),
                tuple(self._gateway.history()),
            ),
        )

    def save(self, form: AccountFormModel) -> AccountSaveResult:
        if type(form) is not AccountFormModel:
            raise TypeError("form must be an AccountFormModel")
        validation = form.validate()
        snapshot = validation.snapshot
        if snapshot is None:
            return AccountSaveResult(validation.errors, None)
        return _gateway_call(
            "ACCOUNT_SAVE_FAILED",
            lambda: AccountSaveResult({}, self._gateway.save(snapshot)),
        )


def render_account_page(model: AccountPageModel | None) -> None:
    ui.label("本应用只提供研究与记录，不连接券商，也不会自动下单。").classes(
        "w-full rounded bg-amber-50 border border-amber-200 text-amber-900 px-4 py-3 text-sm"
    )
    if model is None:
        ui.label("账户服务未配置；当前不会显示或保存任何虚构持仓。").classes(
            "text-sm text-slate-600"
        )
        return

    @ui.refreshable
    def content() -> None:
        try:
            state = model.state()
        except Exception:
            ui.label("账户状态读取失败，请查看本地日志。").classes("text-red-700")
            return
        with ui.row().classes("w-full gap-3"):
            latest = state.latest
            summaries = (
                ("快照日期", latest.snapshot.as_of.isoformat() if latest else "暂无"),
                ("现金", f"{latest.snapshot.cash:.2f}" if latest else "暂无"),
                ("账户权益", f"{latest.snapshot.equity:.2f}" if latest else "暂无"),
                ("历史版本", str(len(state.history))),
            )
            for label, value in summaries:
                with ui.card().classes("min-w-40 border border-slate-200 shadow-none"):
                    ui.label(label).classes("text-xs text-slate-500")
                    ui.label(value).classes("font-semibold text-slate-900")
        rows = [
            {
                "version": record.row_id,
                "date": record.snapshot.as_of.isoformat(),
                "cash": str(record.snapshot.cash),
                "positions": len(record.snapshot.positions),
                "equity": str(record.snapshot.equity),
            }
            for record in reversed(state.history)
        ]
        ui.table(
            columns=[
                {"name": "version", "label": "版本", "field": "version"},
                {"name": "date", "label": "日期", "field": "date"},
                {"name": "cash", "label": "现金", "field": "cash"},
                {"name": "positions", "label": "持仓数", "field": "positions"},
                {"name": "equity", "label": "权益", "field": "equity"},
            ],
            rows=rows,
            row_key="version",
        ).classes("w-full")

    content()
    with ui.card().classes("w-full border border-slate-200 shadow-none"):
        ui.label("追加账户快照").classes("font-semibold")
        ui.label("每次保存都会新增历史版本，不修改或删除旧快照。").classes(
            "text-xs text-slate-500"
        )
        with ui.row().classes("w-full gap-3"):
            as_of = ui.input("快照日期（YYYY-MM-DD）").props("aria-label=快照日期").classes("grow")
            cash = ui.input("现金（元）").props("aria-label=现金").classes("grow")
        ui.label("持仓（完全留空的行不会写入）").classes("text-sm font-medium")
        position_inputs: list[tuple[object, ...]] = []
        with ui.column().classes("w-full gap-2") as position_rows:
            pass

        def add_position() -> None:
            index = len(position_inputs) + 1
            with position_rows:
                with ui.row().classes("w-full grid grid-cols-2 md:grid-cols-6 gap-2"):
                    symbol = ui.input(f"代码 {index}").props(f"aria-label=持仓代码{index}")
                    asset_type = ui.select(
                        {"etf": "ETF", "stock": "股票"},
                        label=f"类型 {index}",
                    ).props(f"aria-label=资产类型{index}")
                    quantity = ui.input(f"数量 {index}").props(f"aria-label=持仓数量{index}")
                    available = ui.input(f"可用 {index}").props(f"aria-label=可用数量{index}")
                    average = ui.input(f"平均成本 {index}").props(
                        f"aria-label=平均成本{index}"
                    )
                    mark = ui.input(f"标记价格 {index}").props(
                        f"aria-label=标记价格{index}"
                    )
            position_inputs.append((symbol, asset_type, quantity, available, average, mark))

        add_position()
        ui.button("添加持仓行", on_click=add_position, icon="add").props(
            "outline aria-label=添加持仓行"
        )
        errors = ui.column().classes("w-full gap-1")

        def save() -> None:
            positions = []
            for inputs in position_inputs:
                values = tuple(getattr(item, "value", None) for item in inputs)
                if all(value in (None, "") for value in values):
                    continue
                positions.append(PositionFormModel(*values))
            form = AccountFormModel(as_of.value, cash.value, tuple(positions))
            try:
                result = model.save(form)
            except Exception:
                errors.clear()
                with errors:
                    ui.label("保存失败，请查看本地日志。").classes("text-red-700")
                return
            errors.clear()
            with errors:
                if result.errors:
                    for message in result.errors.values():
                        ui.label(message).classes("text-sm text-red-700")
                else:
                    ui.label("账户快照已追加保存。").classes("text-sm text-emerald-700")
                    content.refresh()

        ui.button("保存新快照", on_click=save, icon="save").props("aria-label=保存新账户快照")
