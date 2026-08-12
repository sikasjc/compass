from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import MAX_EMAX, MIN_EMIN, Decimal, localcontext
from math import isfinite
import re
from threading import RLock
from typing import Protocol, TypeAlias
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]

from compass.domain.market import AssetType, BarFrame, InstrumentId
from compass.domain.trading import TargetIntent
from compass.strategies.base import HoldingSummary, StrategyContext


SHANGHAI = ZoneInfo("Asia/Shanghai")
_REASON_CODE = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_DEFAULT_SESSIONS = ((time(9, 30), time(11, 30)), (time(13), time(15)))

TradingDayCalendar: TypeAlias = Callable[[date], bool]
IntradayCalculator: TypeAlias = Callable[[StrategyContext], Sequence[TargetIntent]]
CompletionClock: TypeAlias = Callable[[], datetime]


class SnapshotProvider(Protocol):
    name: str

    def fetch_snapshot(self, instruments: tuple[InstrumentId, ...]) -> pd.DataFrame: ...


def _shanghai_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{label} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(SHANGHAI)


def _completion_timestamp(
    value: object,
    invoked_at: datetime,
    previous_observed_at: datetime | None,
) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("completion clock must return a datetime")
    if value.tzinfo != SHANGHAI or value.utcoffset() is None:
        raise ValueError("completion clock must use Asia/Shanghai")
    if value < invoked_at:
        raise ValueError("completion clock must not precede invocation")
    if previous_observed_at is not None and value < previous_observed_at:
        raise ValueError("completion clock must not precede the last observation")
    return value


def _exact_decimal(
    value: object,
    *,
    label: str,
    positive: bool = False,
    non_negative: bool = False,
) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    if not value.is_finite():
        raise ValueError(f"{label} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    if non_negative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _snapshot_decimal(
    value: object,
    *,
    label: str,
    positive: bool = False,
    non_negative: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{label} must not be boolean")
    if type(value) is Decimal:
        result = value
    elif isinstance(value, (int, float)):
        result = Decimal(str(value))
    else:
        try:
            result = Decimal(str(value))
        except Exception as error:
            raise ValueError(f"{label} must be numeric") from error
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    if non_negative and result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _exact_product(left: Decimal, right: Decimal) -> Decimal:
    precision = max(50, len(left.as_tuple().digits) + len(right.as_tuple().digits))
    with localcontext() as context:
        context.prec = precision
        context.Emax = MAX_EMAX
        context.Emin = MIN_EMIN
        context.clamp = 0
        return left * right


def _canonical_instrument(value: object) -> InstrumentId:
    if type(value) not in (InstrumentId, str):
        raise TypeError("snapshot instrument must be an InstrumentId or canonical text")
    try:
        if type(value) is InstrumentId:
            instrument = value
        else:
            assert isinstance(value, str)
            instrument = InstrumentId.parse(value)
            if value != str(instrument):
                raise ValueError("snapshot instrument must be canonical")
        if InstrumentId.parse(str(instrument)) != instrument:
            raise ValueError("snapshot instrument must be canonical")
    except (AttributeError, TypeError, ValueError):
        raise ValueError("snapshot instrument must be canonical") from None
    return instrument


@dataclass(frozen=True, slots=True)
class IntradayQuote:
    instrument: InstrumentId
    source_at: datetime
    raw_open: Decimal
    raw_high: Decimal
    raw_low: Decimal
    raw_price: Decimal
    comparable_open: Decimal
    comparable_high: Decimal
    comparable_low: Decimal
    comparable_price: Decimal
    volume: Decimal
    amount: Decimal
    adjust_factor: Decimal

    def __post_init__(self) -> None:
        if type(self.instrument) is not InstrumentId:
            raise TypeError("quote instrument must be an exact InstrumentId")
        _canonical_instrument(self.instrument)
        object.__setattr__(
            self,
            "source_at",
            _shanghai_timestamp(self.source_at, label="quote source_at"),
        )
        for label, value in (
            ("raw open", self.raw_open),
            ("raw high", self.raw_high),
            ("raw low", self.raw_low),
            ("raw price", self.raw_price),
            ("comparable open", self.comparable_open),
            ("comparable high", self.comparable_high),
            ("comparable low", self.comparable_low),
            ("comparable price", self.comparable_price),
            ("adjust factor", self.adjust_factor),
        ):
            _exact_decimal(value, label=label, positive=True)
        _exact_decimal(self.volume, label="volume", non_negative=True)
        _exact_decimal(self.amount, label="amount", non_negative=True)
        if self.raw_high < max(self.raw_open, self.raw_low, self.raw_price):
            raise ValueError("raw high is below another price")
        if self.raw_low > min(self.raw_open, self.raw_high, self.raw_price):
            raise ValueError("raw low is above another price")
        products = (
            ("open", self.raw_open, self.comparable_open),
            ("high", self.raw_high, self.comparable_high),
            ("low", self.raw_low, self.comparable_low),
            ("price", self.raw_price, self.comparable_price),
        )
        for label, raw, comparable in products:
            if comparable != _exact_product(raw, self.adjust_factor):
                raise ValueError(
                    f"comparable {label} must equal raw {label} times adjust_factor"
                )
        if self.comparable_high < max(
            self.comparable_open,
            self.comparable_low,
            self.comparable_price,
        ):
            raise ValueError("comparable high is below another price")
        if self.comparable_low > min(
            self.comparable_open,
            self.comparable_high,
            self.comparable_price,
        ):
            raise ValueError("comparable low is above another price")


@dataclass(frozen=True, slots=True)
class IntradaySignal:
    strategy_id: str
    instrument: InstrumentId
    target_weight: Decimal
    score: float
    confidence: float
    reason_code: str
    source_at: datetime
    raw_price: Decimal
    comparable_price: Decimal
    status: str = field(default="temporary", init=False)
    confirmed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.strategy_id) is not str or not self.strategy_id.strip():
            raise ValueError("strategy_id must be non-empty text")
        if type(self.instrument) is not InstrumentId:
            raise TypeError("signal instrument must be an exact InstrumentId")
        _canonical_instrument(self.instrument)
        weight = _exact_decimal(self.target_weight, label="target_weight")
        if not Decimal("0") <= weight <= Decimal("1"):
            raise ValueError("target_weight must be between zero and one")
        if type(self.score) is not float or not isfinite(self.score):
            raise ValueError("score must be an exact finite float")
        if (
            type(self.confidence) is not float
            or not isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be an exact finite float between zero and one")
        if type(self.reason_code) is not str or _REASON_CODE.fullmatch(self.reason_code) is None:
            raise ValueError("reason_code must be a stable upper-snake identifier")
        object.__setattr__(
            self,
            "source_at",
            _shanghai_timestamp(self.source_at, label="signal source_at"),
        )
        _exact_decimal(self.raw_price, label="raw_price", positive=True)
        _exact_decimal(self.comparable_price, label="comparable_price", positive=True)


@dataclass(frozen=True, slots=True)
class IntradayState:
    observed_at: datetime
    source_at: datetime | None
    quotes: tuple[IntradayQuote, ...]
    signals: tuple[IntradaySignal, ...]
    notifications: tuple[IntradaySignal, ...]
    active_session: bool
    stale: bool
    consecutive_failures: int
    failure_code: str | None
    persist_to_daily_results: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        observed = _shanghai_timestamp(self.observed_at, label="state observed_at")
        object.__setattr__(self, "observed_at", observed)
        if self.source_at is not None:
            source = _shanghai_timestamp(self.source_at, label="state source_at")
            if source > observed:
                raise ValueError("state source_at must not be in the future")
            object.__setattr__(self, "source_at", source)
        quotes = tuple(self.quotes)
        signals = tuple(self.signals)
        notifications = tuple(self.notifications)
        if any(type(item) is not IntradayQuote for item in quotes):
            raise TypeError("state quotes must contain exact IntradayQuote values")
        if any(type(item) is not IntradaySignal for item in signals):
            raise TypeError("state signals must contain exact IntradaySignal values")
        if any(type(item) is not IntradaySignal for item in notifications):
            raise TypeError("state notifications must contain exact IntradaySignal values")
        if any(item not in signals for item in notifications):
            raise ValueError("state notifications must also be visible signals")
        if tuple(sorted(quotes, key=lambda item: str(item.instrument))) != quotes:
            raise ValueError("state quotes must be symbol-sorted")
        quote_symbols = tuple(item.instrument for item in quotes)
        if len(set(quote_symbols)) != len(quote_symbols):
            raise ValueError("state quotes must be unique by instrument")

        def signal_key(item: IntradaySignal) -> tuple[str, str, str]:
            return (str(item.instrument), item.strategy_id, item.reason_code)

        if tuple(sorted(signals, key=signal_key)) != signals:
            raise ValueError("state signals must be symbol-sorted")
        if tuple(sorted(notifications, key=signal_key)) != notifications:
            raise ValueError("state notifications must be symbol-sorted")
        decision_keys = tuple((item.strategy_id, item.instrument) for item in signals)
        if len(set(decision_keys)) != len(decision_keys):
            raise ValueError("state signals must be unique by strategy and instrument")
        notification_keys = tuple(
            (item.strategy_id, item.instrument) for item in notifications
        )
        if len(set(notification_keys)) != len(notification_keys):
            raise ValueError("state notifications must be unique by strategy and instrument")
        if quotes:
            expected_source = min(item.source_at for item in quotes)
            if self.source_at != expected_source:
                raise ValueError("state source_at must be the oldest visible quote timestamp")
        elif self.source_at is not None:
            raise ValueError("state source_at requires visible quotes")
        quote_by_symbol = {item.instrument: item for item in quotes}
        for signal in signals:
            quote = quote_by_symbol.get(signal.instrument)
            if quote is None:
                raise ValueError("every state signal must have a display quote")
            if signal.source_at != quote.source_at:
                raise ValueError("signal source_at must match its display quote")
            if signal.raw_price != quote.raw_price:
                raise ValueError("signal raw_price must match its display quote")
            if signal.comparable_price != quote.comparable_price:
                raise ValueError("signal comparable_price must match its display quote")
        if any(item.source_at > observed for item in quotes):
            raise ValueError("state quotes must not be from the future")
        if type(self.active_session) is not bool or type(self.stale) is not bool:
            raise TypeError("state flags must be exact booleans")
        if type(self.consecutive_failures) is not int:
            raise TypeError("consecutive_failures must be an exact integer")
        if self.consecutive_failures < 0:
            raise ValueError("consecutive_failures must be non-negative")
        if self.failure_code is not None and (
            type(self.failure_code) is not str
            or _REASON_CODE.fullmatch(self.failure_code) is None
        ):
            raise ValueError("failure_code must be a stable upper-snake identifier")
        if self.consecutive_failures and self.failure_code is None:
            raise ValueError("consecutive failures require a failure_code")
        if self.failure_code is None and self.stale:
            raise ValueError("stale state requires a failure_code")
        if self.failure_code == "LAST_QUOTE_STALE" and not self.stale:
            raise ValueError("LAST_QUOTE_STALE requires stale state")
        if (
            self.failure_code is not None
            and self.failure_code != "LAST_QUOTE_STALE"
            and self.consecutive_failures == 0
        ):
            raise ValueError("failure state requires a positive consecutive failure count")
        if notifications and (
            not self.active_session
            or self.stale
            or self.failure_code is not None
            or self.consecutive_failures
        ):
            raise ValueError("notifications require a fresh successful active-session state")
        object.__setattr__(self, "quotes", quotes)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "notifications", notifications)


class _StaleSnapshotError(ValueError):
    pass


class IntradayService:
    """Recalculate temporary advice from confirmed daily history and one live row."""

    def __init__(
        self,
        *,
        instruments: Sequence[InstrumentId],
        provider: SnapshotProvider,
        daily_history: Mapping[InstrumentId, pd.DataFrame],
        calculator: IntradayCalculator,
        is_trading_day: TradingDayCalendar,
        account_equity: Decimal,
        asset_types: Mapping[InstrumentId, AssetType],
        cash: Decimal = Decimal("0"),
        holdings: Mapping[InstrumentId, HoldingSummary] | Sequence[HoldingSummary] = (),
        freshness: timedelta = timedelta(minutes=10),
        cooldown: timedelta = timedelta(minutes=30),
        material_change: Decimal = Decimal("0.01"),
        stale_after_failures: int = 2,
        sessions: Sequence[tuple[time, time]] = _DEFAULT_SESSIONS,
        completion_clock: CompletionClock | None = None,
    ) -> None:
        ordered = tuple(sorted(tuple(instruments), key=str))
        if not ordered or any(type(item) is not InstrumentId for item in ordered):
            raise ValueError("instruments must contain exact InstrumentId values")
        if len(set(ordered)) != len(ordered):
            raise ValueError("instruments must be unique")
        if not callable(getattr(provider, "fetch_snapshot", None)):
            raise TypeError("provider must implement fetch_snapshot")
        if not callable(calculator):
            raise TypeError("calculator must be callable")
        if not callable(is_trading_day):
            raise TypeError("is_trading_day must be callable")
        if not isinstance(daily_history, Mapping) or set(daily_history) != set(ordered):
            raise ValueError("daily_history must exist for exactly the configured instruments")
        histories: dict[InstrumentId, pd.DataFrame] = {}
        for instrument in ordered:
            frame = daily_history[instrument]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("daily_history values must be DataFrames")
            validated = BarFrame.validate(frame.copy(deep=True))
            if "adjust_flag" in validated:
                for flag in validated["adjust_flag"]:
                    if type(flag) is not str or flag != "3":
                        raise ValueError(
                            "daily_history must use raw prices with adjust_flag code '3'"
                        )
            histories[instrument] = self._comparable_history(validated)
        checked_asset_types = self._validate_asset_types(asset_types, ordered)
        checked_holdings = self._validate_holdings(holdings, ordered)
        _exact_decimal(account_equity, label="account_equity", non_negative=True)
        _exact_decimal(cash, label="cash", non_negative=True)
        if type(freshness) is not timedelta or freshness <= timedelta(0):
            raise ValueError("freshness must be a positive timedelta")
        if type(cooldown) is not timedelta or cooldown <= timedelta(0):
            raise ValueError("cooldown must be a positive timedelta")
        threshold = _exact_decimal(material_change, label="material_change", non_negative=True)
        if threshold > Decimal("1"):
            raise ValueError("material_change must not exceed one")
        if type(stale_after_failures) is not int or stale_after_failures <= 0:
            raise ValueError("stale_after_failures must be a positive exact integer")
        checked_sessions = self._validate_sessions(sessions)

        self._instruments = ordered
        self._provider = provider
        self._histories = histories
        self._calculator = calculator
        self._is_trading_day = is_trading_day
        self._account_equity = account_equity
        self._cash = cash
        self._asset_types = checked_asset_types
        self._holdings = checked_holdings
        self._freshness = freshness
        self._cooldown = cooldown
        self._material_change = material_change
        self._stale_after_failures = stale_after_failures
        self._sessions = checked_sessions
        self._completion_clock = completion_clock
        self._state: IntradayState | None = None
        self._announced: dict[
            tuple[str, InstrumentId], tuple[IntradaySignal, datetime]
        ] = {}
        self._lock = RLock()

    @staticmethod
    def _comparable_history(raw: pd.DataFrame) -> pd.DataFrame:
        """Copy raw daily bars into strategy price space; a missing factor means one."""

        comparable = raw.copy(deep=True)
        if "adjust_factor" not in raw:
            return comparable
        factors = tuple(
            _snapshot_decimal(value, label="daily adjust_factor", positive=True)
            for value in raw["adjust_factor"]
        )
        comparable["adjust_factor"] = pd.Series(
            factors,
            index=raw.index,
            dtype=object,
        )
        for column in ("open", "high", "low", "close"):
            comparable[column] = pd.Series(
                (
                    product
                    if type(value) is Decimal
                    else float(product)
                    for value, factor in zip(raw[column], factors, strict=True)
                    for product in (
                        _exact_product(
                            _snapshot_decimal(
                                value,
                                label=f"daily {column}",
                                positive=True,
                            ),
                            factor,
                        ),
                    )
                ),
                index=raw.index,
            )
        return BarFrame.validate(comparable)

    @staticmethod
    def _validate_asset_types(
        asset_types: Mapping[InstrumentId, AssetType],
        instruments: tuple[InstrumentId, ...],
    ) -> dict[InstrumentId, AssetType]:
        if not isinstance(asset_types, Mapping) or set(asset_types) != set(instruments):
            raise ValueError("asset_types must exist for exactly the configured instruments")
        checked: dict[InstrumentId, AssetType] = {}
        for instrument, asset_type in asset_types.items():
            if type(instrument) is not InstrumentId or type(asset_type) is not AssetType:
                raise TypeError("asset_types must map exact InstrumentId to exact AssetType")
            _canonical_instrument(instrument)
            checked[instrument] = asset_type
        return checked

    @staticmethod
    def _validate_holdings(
        holdings: Mapping[InstrumentId, HoldingSummary] | Sequence[HoldingSummary],
        instruments: tuple[InstrumentId, ...],
    ) -> tuple[HoldingSummary, ...]:
        if isinstance(holdings, Mapping):
            pairs = tuple(holdings.items())
        elif isinstance(holdings, Sequence) and not isinstance(
            holdings, (str, bytes, bytearray)
        ):
            values = tuple(holdings)
            if any(type(holding) is not HoldingSummary for holding in values):
                raise TypeError("holdings must contain exact HoldingSummary values")
            pairs = tuple((holding.instrument, holding) for holding in values)
        else:
            raise TypeError("holdings must be a mapping or sequence")
        configured = set(instruments)
        checked: dict[InstrumentId, HoldingSummary] = {}
        for instrument, holding in pairs:
            if type(instrument) is not InstrumentId or type(holding) is not HoldingSummary:
                raise TypeError("holdings must contain exact HoldingSummary values")
            if instrument != holding.instrument:
                raise ValueError("holding key must match holding instrument")
            if instrument not in configured:
                raise ValueError("holdings must belong to configured instruments")
            if instrument in checked:
                raise ValueError("holdings must be unique by instrument")
            checked[instrument] = holding
        return tuple(checked[instrument] for instrument in sorted(checked, key=str))

    @staticmethod
    def _validate_sessions(
        sessions: Sequence[tuple[time, time]],
    ) -> tuple[tuple[time, time], ...]:
        checked = tuple(sessions)
        if not checked:
            raise ValueError("sessions must not be empty")
        previous_end: time | None = None
        for item in checked:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("sessions must contain time pairs")
            start, end = item
            if type(start) is not time or type(end) is not time:
                raise TypeError("session boundaries must be exact time values")
            if start.tzinfo is not None or end.tzinfo is not None:
                raise ValueError("session boundaries must be timezone-naive local times")
            if start >= end:
                raise ValueError("session start must precede end")
            if previous_end is not None and start <= previous_end:
                raise ValueError("sessions must be ordered and non-overlapping")
            previous_end = end
        return checked

    @property
    def state(self) -> IntradayState | None:
        with self._lock:
            return self._state

    @property
    def instruments(self) -> tuple[InstrumentId, ...]:
        return self._instruments

    def refresh(self, now: datetime) -> IntradayState:
        invoked_at = _shanghai_timestamp(now, label="refresh now")
        with self._lock:
            try:
                trading_day = self._is_trading_day(invoked_at.date())
                if type(trading_day) is not bool:
                    raise TypeError("calendar must return an exact bool")
            except Exception:
                return self._failed(
                    invoked_at, "CALENDAR_FAILED", active_session=False
                )
            active = trading_day and any(
                start <= invoked_at.timetz().replace(tzinfo=None) <= end
                for start, end in self._sessions
            )
            if not active:
                return self._outside_session(invoked_at)

            try:
                fetched = self._provider.fetch_snapshot(self._instruments)
            except Exception:
                return self._failed(
                    invoked_at, "SNAPSHOT_FETCH_FAILED", active_session=True
                )
            try:
                completed_at = (
                    invoked_at
                    if self._completion_clock is None
                    else _completion_timestamp(
                        self._completion_clock(),
                        invoked_at,
                        (
                            None
                            if self._state is None
                            else self._state.observed_at
                        ),
                    )
                )
            except Exception:
                return self._failed(
                    invoked_at,
                    "COMPLETION_CLOCK_FAILED",
                    active_session=True,
                )
            try:
                quotes = self._validated_quotes(fetched, completed_at)
            except _StaleSnapshotError:
                return self._failed(
                    completed_at,
                    "SNAPSHOT_INVALID",
                    active_session=True,
                    force_stale=True,
                )
            except (TypeError, ValueError, KeyError, ArithmeticError):
                return self._failed(
                    completed_at, "SNAPSHOT_INVALID", active_session=True
                )

            try:
                context = self._calculation_context(invoked_at.date(), quotes)
                calculated = tuple(self._calculator(context))
                signals = self._temporary_signals(
                    calculated,
                    quotes,
                    invoked_at.date(),
                )
            except Exception:
                return self._failed(
                    completed_at,
                    "SIGNAL_CALCULATION_FAILED",
                    active_session=True,
                )

            notifications = self._notifications(signals, completed_at)
            source_at = min(quote.source_at for quote in quotes)
            state = IntradayState(
                observed_at=completed_at,
                source_at=source_at,
                quotes=quotes,
                signals=signals,
                notifications=notifications,
                active_session=True,
                stale=False,
                consecutive_failures=0,
                failure_code=None,
            )
            self._state = state
            return state

    @staticmethod
    def _retained_observed_at(
        observed: datetime,
        previous: IntradayState | None,
    ) -> datetime:
        if previous is None:
            return observed
        candidates = [observed, previous.observed_at]
        if previous.source_at is not None:
            candidates.append(previous.source_at)
        return max(candidates)

    def _outside_session(self, observed: datetime) -> IntradayState:
        previous = self._state
        quotes = () if previous is None else previous.quotes
        signals = () if previous is None else previous.signals
        source = None if previous is None else previous.source_at
        retained_observed = self._retained_observed_at(observed, previous)
        aged = (
            source is not None
            and retained_observed - source > self._freshness
        )
        stale = aged or (False if previous is None else previous.stale)
        failure_code = (
            "LAST_QUOTE_STALE"
            if aged
            else None
            if previous is None
            else previous.failure_code
        )
        state = IntradayState(
            observed_at=retained_observed,
            source_at=source,
            quotes=quotes,
            signals=signals,
            notifications=(),
            active_session=False,
            stale=stale,
            consecutive_failures=(0 if previous is None else previous.consecutive_failures),
            failure_code=failure_code,
        )
        self._state = state
        return state

    def _failed(
        self,
        observed: datetime,
        code: str,
        *,
        active_session: bool,
        force_stale: bool = False,
    ) -> IntradayState:
        previous = self._state
        failures = (0 if previous is None else previous.consecutive_failures) + 1
        quotes = () if previous is None else previous.quotes
        signals = () if previous is None else previous.signals
        source = None if previous is None else previous.source_at
        retained_observed = self._retained_observed_at(observed, previous)
        aged = (
            source is not None
            and retained_observed - source > self._freshness
        )
        state = IntradayState(
            observed_at=retained_observed,
            source_at=source,
            quotes=quotes,
            signals=signals,
            notifications=(),
            active_session=active_session,
            stale=force_stale or aged or failures >= self._stale_after_failures,
            consecutive_failures=failures,
            failure_code=code,
        )
        self._state = state
        return state

    def _validated_quotes(
        self,
        frame: object,
        observed: datetime,
    ) -> tuple[IntradayQuote, ...]:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("snapshot provider must return a DataFrame")
        if frame.columns.has_duplicates:
            raise ValueError("snapshot columns must be unique")
        required = {
            "instrument",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"snapshot is missing {sorted(missing)[0]}")
        quotes: list[IntradayQuote] = []
        for record in frame.copy(deep=True).to_dict(orient="records"):
            instrument = _canonical_instrument(record["instrument"])
            timestamp_value = record["timestamp"]
            if isinstance(timestamp_value, pd.Timestamp):
                timestamp_value = timestamp_value.to_pydatetime()
            source_at = _shanghai_timestamp(timestamp_value, label="snapshot timestamp")
            if source_at > observed:
                raise ValueError("snapshot timestamp must not be in the future")
            if source_at.date() != observed.date() or observed - source_at > self._freshness:
                raise _StaleSnapshotError("snapshot timestamp is stale")
            raw_open = _snapshot_decimal(record["open"], label="snapshot open", positive=True)
            raw_high = _snapshot_decimal(record["high"], label="snapshot high", positive=True)
            raw_low = _snapshot_decimal(record["low"], label="snapshot low", positive=True)
            raw_price = _snapshot_decimal(record["close"], label="snapshot close", positive=True)
            volume = _snapshot_decimal(
                record["volume"], label="snapshot volume", non_negative=True
            )
            amount = _snapshot_decimal(
                record["amount"], label="snapshot amount", non_negative=True
            )
            factor = _snapshot_decimal(
                record.get("adjust_factor", Decimal("1")),
                label="snapshot adjust_factor",
                positive=True,
            )
            quote = IntradayQuote(
                instrument=instrument,
                source_at=source_at,
                raw_open=raw_open,
                raw_high=raw_high,
                raw_low=raw_low,
                raw_price=raw_price,
                comparable_open=_exact_product(raw_open, factor),
                comparable_high=_exact_product(raw_high, factor),
                comparable_low=_exact_product(raw_low, factor),
                comparable_price=_exact_product(raw_price, factor),
                volume=volume,
                amount=amount,
                adjust_factor=factor,
            )
            quotes.append(quote)
        symbols = tuple(quote.instrument for quote in quotes)
        if len(set(symbols)) != len(symbols):
            raise ValueError("snapshot instruments must be unique")
        if set(symbols) != set(self._instruments):
            raise ValueError("snapshot must contain exactly the configured instruments")
        return tuple(sorted(quotes, key=lambda item: str(item.instrument)))

    def _calculation_context(
        self,
        day: date,
        quotes: tuple[IntradayQuote, ...],
    ) -> StrategyContext:
        bars: dict[InstrumentId, pd.DataFrame] = {}
        day_timestamp = pd.Timestamp(day)
        for quote in quotes:
            confirmed = self._histories[quote.instrument]
            confirmed = confirmed.loc[confirmed.index < day_timestamp].copy(deep=True)
            comparable_values: dict[str, Decimal | float] = {
                "open": quote.comparable_open,
                "high": quote.comparable_high,
                "low": quote.comparable_low,
                "close": quote.comparable_price,
            }
            for column, value in tuple(comparable_values.items()):
                if not confirmed.empty and not any(
                    type(item) is Decimal for item in confirmed[column]
                ):
                    comparable_values[column] = float(value)
            incomplete = pd.DataFrame(
                {
                    "open": [comparable_values["open"]],
                    "high": [comparable_values["high"]],
                    "low": [comparable_values["low"]],
                    "close": [comparable_values["close"]],
                    "volume": [quote.volume],
                    "amount": [quote.amount],
                    "adjust_factor": [quote.adjust_factor],
                },
                index=pd.DatetimeIndex([day_timestamp], name="date"),
            )
            bars[quote.instrument] = pd.concat((confirmed, incomplete), axis=0)
        return StrategyContext(
            as_of=day,
            bars=bars,
            instruments=self._instruments,
            account_equity=self._account_equity,
            cash=self._cash,
            holdings=self._holdings,
            asset_types=self._asset_types,
        )

    def _temporary_signals(
        self,
        intents: tuple[TargetIntent, ...],
        quotes: tuple[IntradayQuote, ...],
        day: date,
    ) -> tuple[IntradaySignal, ...]:
        by_symbol = {quote.instrument: quote for quote in quotes}
        signals: list[IntradaySignal] = []
        keys: set[tuple[str, InstrumentId]] = set()
        for intent in intents:
            if type(intent) is not TargetIntent:
                raise TypeError("calculator must return exact TargetIntent values")
            if type(intent.valid_until) is not date or intent.valid_until < day:
                raise ValueError("intraday intent valid_until must not precede today")
            if intent.instrument not in by_symbol:
                raise ValueError("intraday intent instrument is not configured")
            key = (intent.strategy_id, intent.instrument)
            if key in keys:
                raise ValueError("calculator returned duplicate strategy/instrument intents")
            keys.add(key)
            quote = by_symbol[intent.instrument]
            signals.append(
                IntradaySignal(
                    strategy_id=intent.strategy_id,
                    instrument=intent.instrument,
                    target_weight=intent.target_weight,
                    score=intent.score,
                    confidence=intent.confidence,
                    reason_code=intent.reason_code,
                    source_at=quote.source_at,
                    raw_price=quote.raw_price,
                    comparable_price=quote.comparable_price,
                )
            )
        return tuple(
            sorted(signals, key=lambda item: (str(item.instrument), item.strategy_id, item.reason_code))
        )

    def _notifications(
        self,
        signals: tuple[IntradaySignal, ...],
        observed: datetime,
    ) -> tuple[IntradaySignal, ...]:
        notifications: list[IntradaySignal] = []
        for signal in signals:
            key = (signal.strategy_id, signal.instrument)
            previous = self._announced.get(key)
            announce = previous is None
            if previous is not None:
                previous_signal, announced_at = previous
                changed_reason = previous_signal.reason_code != signal.reason_code
                delta = abs(previous_signal.target_weight - signal.target_weight)
                material = delta > 0 and delta >= self._material_change
                cooled_down = observed - announced_at >= self._cooldown
                announce = changed_reason or material or cooled_down
            if announce:
                notifications.append(signal)
                self._announced[key] = (signal, observed)
        return tuple(notifications)
