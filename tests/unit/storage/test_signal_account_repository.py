from __future__ import annotations

from decimal import Decimal
import json

import pytest

from compass.storage.signal_account_repository import (
    SignalAccountRepository,
    SignalAccountStrategySetting,
)
from compass.storage.canonical_json import canonical_json, content_hash


def test_signal_account_registry_keeps_independent_account_settings(tmp_path) -> None:
    path = tmp_path / "signal_accounts.json"
    repository = SignalAccountRepository(path)

    assert repository.state().active.name == "默认账户"

    second = repository.create("account-second", "长期账户")
    repository.save_configuration(
        second.account_id,
        (SignalAccountStrategySetting("strategy-b", Decimal("0.6")),),
        cash_reserve=Decimal("0.2"),
        minimum_trade_amount=Decimal("3000"),
    )
    repository.select("main")

    restored = SignalAccountRepository(path).state()
    assert restored.active_account_id == "main"
    assert restored.active.strategies == ()
    assert next(item for item in restored.profiles if item.account_id == second.account_id) == (
        repository.select(second.account_id).active
    )


def test_signal_account_registry_can_share_an_existing_holdings_source(tmp_path) -> None:
    path = tmp_path / "signal_accounts.json"
    repository = SignalAccountRepository(path)

    shared = repository.create(
        "account-shared",
        "共享策略方案",
        holdings_account_id="main",
    )

    assert shared.holdings_account_id == "main"
    assert SignalAccountRepository(path).state().active.holdings_account_id == "main"
    with pytest.raises(ValueError, match="SIGNAL_ACCOUNT_HOLDINGS_IN_USE"):
        repository.delete("main")


def test_signal_account_registry_migrates_version_one_profiles_to_independent_holdings(
    tmp_path,
) -> None:
    path = tmp_path / "signal_accounts.json"
    SignalAccountRepository(path)
    document = path.read_text("utf-8")

    wrapper = json.loads(document)
    payload = json.loads(wrapper["payload_json"])
    payload["schema_version"] = 1
    for profile in payload["profiles"]:
        profile.pop("holdings_account_id")
    payload_json = canonical_json(payload)
    path.write_text(
        canonical_json(
            {"content_hash": content_hash(payload_json), "payload_json": payload_json}
        ),
        "utf-8",
    )

    migrated = SignalAccountRepository(path).state().active

    assert migrated.holdings_account_id == "main"


def test_signal_account_registry_delete_switches_active_and_keeps_one(tmp_path) -> None:
    repository = SignalAccountRepository(tmp_path / "signal_accounts.json")
    second = repository.create("account-second", "短线账户")

    state = repository.delete(second.account_id)

    assert state.active_account_id == "main"
    with pytest.raises(ValueError, match="SIGNAL_ACCOUNT_LAST_DELETE_FORBIDDEN"):
        repository.delete("main")


def test_signal_account_registry_rejects_duplicate_display_name(tmp_path) -> None:
    repository = SignalAccountRepository(tmp_path / "signal_accounts.json")

    with pytest.raises(ValueError, match="SIGNAL_ACCOUNT_NAME_DUPLICATE"):
        repository.create("account-second", "默认账户")


def test_signal_account_registry_detects_tampering(tmp_path) -> None:
    path = tmp_path / "signal_accounts.json"
    repository = SignalAccountRepository(path)
    path.write_text(path.read_text("utf-8").replace("默认账户", "被修改"), "utf-8")

    with pytest.raises(ValueError, match="SIGNAL_ACCOUNT_REGISTRY_INTEGRITY"):
        repository.state()
