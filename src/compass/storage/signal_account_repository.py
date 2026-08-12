from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, DecimalException
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from compass.services.safe_display import safe_display_text, safe_identifier
from compass.storage.canonical_json import (
    canonical_json,
    content_hash,
    decode_canonical_json,
)


_SCHEMA_VERSION = 2


def _decimal(value: object, *, label: str) -> Decimal:
    if type(value) is not str:
        raise ValueError(f"{label} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except DecimalException:
        raise ValueError(f"{label} must be a canonical decimal string") from None
    if not parsed.is_finite() or str(parsed.normalize()) != value:
        raise ValueError(f"{label} must be a canonical decimal string")
    return parsed


def _decimal_text(value: Decimal) -> str:
    if type(value) is not Decimal or not value.is_finite():
        raise TypeError("configuration values must be finite exact Decimals")
    return str(value.normalize())


@dataclass(frozen=True, slots=True)
class SignalAccountStrategySetting:
    strategy_instance_id: str
    budget: Decimal

    def __post_init__(self) -> None:
        safe_identifier(self.strategy_instance_id, label="strategy instance id")
        if (
            type(self.budget) is not Decimal
            or not self.budget.is_finite()
            or not Decimal("0") < self.budget <= Decimal("1")
        ):
            raise ValueError("strategy budget must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class SignalAccountProfile:
    account_id: str
    name: str
    strategies: tuple[SignalAccountStrategySetting, ...] = ()
    cash_reserve: Decimal = Decimal("0.10")
    minimum_trade_amount: Decimal = Decimal("5000")
    holdings_account_id: str | None = None

    def __post_init__(self) -> None:
        safe_identifier(self.account_id, label="signal account id")
        safe_display_text(self.name, label="signal account name", maximum=64)
        holdings_account_id = (
            self.account_id if self.holdings_account_id is None else self.holdings_account_id
        )
        safe_identifier(holdings_account_id, label="holdings account id")
        strategies = tuple(self.strategies)
        if any(type(item) is not SignalAccountStrategySetting for item in strategies):
            raise TypeError("account strategies must contain exact settings")
        strategy_ids = tuple(item.strategy_instance_id for item in strategies)
        if strategy_ids != tuple(sorted(set(strategy_ids))):
            raise ValueError("account strategies must be unique and sorted")
        if (
            type(self.cash_reserve) is not Decimal
            or not self.cash_reserve.is_finite()
            or not Decimal("0") <= self.cash_reserve < Decimal("1")
        ):
            raise ValueError("cash reserve must be in [0, 1)")
        if (
            type(self.minimum_trade_amount) is not Decimal
            or not self.minimum_trade_amount.is_finite()
            or self.minimum_trade_amount < 0
        ):
            raise ValueError("minimum trade amount must be non-negative")
        if sum((item.budget for item in strategies), Decimal("0")) + self.cash_reserve > Decimal(
            "1"
        ):
            raise ValueError("account strategy budgets exceed available capital")
        object.__setattr__(self, "strategies", strategies)
        object.__setattr__(self, "holdings_account_id", holdings_account_id)


@dataclass(frozen=True, slots=True)
class SignalAccountRegistryState:
    active_account_id: str
    profiles: tuple[SignalAccountProfile, ...]

    def __post_init__(self) -> None:
        safe_identifier(self.active_account_id, label="active signal account id")
        profiles = tuple(self.profiles)
        if any(type(item) is not SignalAccountProfile for item in profiles):
            raise TypeError("signal account registry must contain exact profiles")
        account_ids = tuple(item.account_id for item in profiles)
        if not profiles or account_ids != tuple(sorted(set(account_ids))):
            raise ValueError("signal account profiles must be non-empty, unique and sorted")
        if self.active_account_id not in account_ids:
            raise ValueError("active signal account must exist")
        if any(item.holdings_account_id not in account_ids for item in profiles):
            raise ValueError("holdings account source must exist")
        object.__setattr__(self, "profiles", profiles)

    @property
    def active(self) -> SignalAccountProfile:
        return next(item for item in self.profiles if item.account_id == self.active_account_id)


class SignalAccountRepository:
    """Atomic local registry for account names and account-scoped strategy settings."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("signal account registry path must be a Path")
        self._path = path
        self._lock = RLock()
        if not path.exists():
            self._write(
                SignalAccountRegistryState(
                    "main",
                    (SignalAccountProfile("main", "默认账户"),),
                )
            )

    def state(self) -> SignalAccountRegistryState:
        with self._lock:
            return self._read()

    def select(self, account_id: str) -> SignalAccountRegistryState:
        checked = safe_identifier(account_id, label="signal account id")
        with self._lock:
            state = self._read()
            if checked not in {item.account_id for item in state.profiles}:
                raise LookupError("SIGNAL_ACCOUNT_NOT_FOUND")
            updated = SignalAccountRegistryState(checked, state.profiles)
            self._write(updated)
            return updated

    def create(
        self,
        account_id: str,
        name: str,
        *,
        holdings_account_id: str | None = None,
    ) -> SignalAccountProfile:
        checked_account_id = safe_identifier(account_id, label="signal account id")
        with self._lock:
            state = self._read()
            source_id = (
                checked_account_id
                if holdings_account_id is None
                else safe_identifier(holdings_account_id, label="holdings account id")
            )
            if source_id != checked_account_id:
                source = next(
                    (item for item in state.profiles if item.account_id == source_id),
                    None,
                )
                if source is None:
                    raise LookupError("SIGNAL_ACCOUNT_HOLDINGS_SOURCE_NOT_FOUND")
                assert source.holdings_account_id is not None
                source_id = source.holdings_account_id
            profile = SignalAccountProfile(
                checked_account_id,
                name,
                holdings_account_id=source_id,
            )
            if profile.account_id in {item.account_id for item in state.profiles}:
                raise ValueError("SIGNAL_ACCOUNT_DUPLICATE")
            if profile.name.casefold() in {item.name.casefold() for item in state.profiles}:
                raise ValueError("SIGNAL_ACCOUNT_NAME_DUPLICATE")
            profiles = tuple(sorted((*state.profiles, profile), key=lambda item: item.account_id))
            self._write(SignalAccountRegistryState(profile.account_id, profiles))
            return profile

    def delete(self, account_id: str) -> SignalAccountRegistryState:
        checked = safe_identifier(account_id, label="signal account id")
        with self._lock:
            state = self._read()
            if len(state.profiles) == 1:
                raise ValueError("SIGNAL_ACCOUNT_LAST_DELETE_FORBIDDEN")
            if any(
                item.account_id != checked and item.holdings_account_id == checked
                for item in state.profiles
            ):
                raise ValueError("SIGNAL_ACCOUNT_HOLDINGS_IN_USE")
            profiles = tuple(item for item in state.profiles if item.account_id != checked)
            if len(profiles) == len(state.profiles):
                raise LookupError("SIGNAL_ACCOUNT_NOT_FOUND")
            active = (
                profiles[0].account_id
                if state.active_account_id == checked
                else state.active_account_id
            )
            updated = SignalAccountRegistryState(active, profiles)
            self._write(updated)
            return updated

    def save_configuration(
        self,
        account_id: str,
        strategies: Sequence[SignalAccountStrategySetting],
        *,
        cash_reserve: Decimal,
        minimum_trade_amount: Decimal,
    ) -> SignalAccountProfile:
        checked = safe_identifier(account_id, label="signal account id")
        selected = tuple(sorted(tuple(strategies), key=lambda item: item.strategy_instance_id))
        with self._lock:
            state = self._read()
            existing = next((item for item in state.profiles if item.account_id == checked), None)
            if existing is None:
                raise LookupError("SIGNAL_ACCOUNT_NOT_FOUND")
            updated_profile = SignalAccountProfile(
                existing.account_id,
                existing.name,
                selected,
                cash_reserve,
                minimum_trade_amount,
                existing.holdings_account_id,
            )
            profiles = tuple(
                updated_profile if item.account_id == checked else item for item in state.profiles
            )
            self._write(SignalAccountRegistryState(state.active_account_id, profiles))
            return updated_profile

    def _read(self) -> SignalAccountRegistryState:
        try:
            text = self._path.read_text("utf-8")
            raw = json.loads(text)
            if type(raw) is not dict or set(raw) != {"content_hash", "payload_json"}:
                raise ValueError
            if canonical_json(raw) != text:
                raise ValueError
            payload = decode_canonical_json(raw["payload_json"], raw["content_hash"])
            if set(payload) != {"active_account_id", "profiles", "schema_version"}:
                raise ValueError
            schema_version = payload["schema_version"]
            if schema_version not in {1, _SCHEMA_VERSION}:
                raise ValueError
            active_account_id = payload["active_account_id"]
            if type(active_account_id) is not str:
                raise ValueError
            raw_profiles = payload["profiles"]
            if type(raw_profiles) is not list:
                raise ValueError
            profiles = []
            for raw_profile in raw_profiles:
                expected_profile_fields = {
                    "account_id",
                    "cash_reserve",
                    "minimum_trade_amount",
                    "name",
                    "strategies",
                }
                if schema_version == _SCHEMA_VERSION:
                    expected_profile_fields.add("holdings_account_id")
                if type(raw_profile) is not dict or set(raw_profile) != expected_profile_fields:
                    raise ValueError
                raw_strategies = raw_profile["strategies"]
                if type(raw_strategies) is not list:
                    raise ValueError
                strategies = tuple(
                    SignalAccountStrategySetting(
                        item["strategy_instance_id"],
                        _decimal(item["budget"], label="strategy budget"),
                    )
                    for item in raw_strategies
                    if type(item) is dict and set(item) == {"budget", "strategy_instance_id"}
                )
                if len(strategies) != len(raw_strategies):
                    raise ValueError
                profiles.append(
                    SignalAccountProfile(
                        raw_profile["account_id"],
                        raw_profile["name"],
                        strategies,
                        _decimal(raw_profile["cash_reserve"], label="cash reserve"),
                        _decimal(
                            raw_profile["minimum_trade_amount"],
                            label="minimum trade amount",
                        ),
                        (
                            raw_profile["account_id"]
                            if schema_version == 1
                            else raw_profile["holdings_account_id"]
                        ),
                    )
                )
            return SignalAccountRegistryState(active_account_id, tuple(profiles))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("SIGNAL_ACCOUNT_REGISTRY_INTEGRITY") from None

    def _write(self, state: SignalAccountRegistryState) -> None:
        payload_json = canonical_json(
            {
                "active_account_id": state.active_account_id,
                "profiles": [
                    {
                        "account_id": profile.account_id,
                        "cash_reserve": _decimal_text(profile.cash_reserve),
                        "holdings_account_id": profile.holdings_account_id,
                        "minimum_trade_amount": _decimal_text(profile.minimum_trade_amount),
                        "name": profile.name,
                        "strategies": [
                            {
                                "budget": _decimal_text(setting.budget),
                                "strategy_instance_id": setting.strategy_instance_id,
                            }
                            for setting in profile.strategies
                        ],
                    }
                    for profile in state.profiles
                ],
                "schema_version": _SCHEMA_VERSION,
            }
        )
        document = canonical_json(
            {
                "content_hash": content_hash(payload_json),
                "payload_json": payload_json,
            }
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(document, encoding="utf-8")
            os.replace(temporary, self._path)
        finally:
            temporary.unlink(missing_ok=True)
