from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from compass.storage.strategy_draft_repository import StrategyDraftRepository
from compass.strategies.rule_document import RuleStrategyDraft, default_rule_document


NOW = datetime(2026, 8, 12, 10, tzinfo=ZoneInfo("Asia/Shanghai"))


def draft() -> RuleStrategyDraft:
    return RuleStrategyDraft(
        "draft-one",
        "main",
        "main-0123456789abcdef",
        default_rule_document(),
        NOW,
    )


def test_strategy_drafts_are_atomic_round_trippable_and_deletable(tmp_path: Path) -> None:
    repository = StrategyDraftRepository(tmp_path / "strategy-drafts.json")

    repository.save(draft())

    assert repository.list() == (draft(),)
    assert repository.get("draft-one") == draft()
    assert repository.delete("draft-one") is True
    assert repository.list() == ()
    assert repository.delete("draft-one") is False


def test_draft_conflicts_and_trash_restore_preserve_work(tmp_path: Path) -> None:
    from dataclasses import replace
    import pytest

    repository = StrategyDraftRepository(tmp_path / "drafts.json")
    original = repository.save(draft())
    changed = replace(
        original, document=original.document.model_copy(update={"name": "更新后的策略"})
    )
    repository.save(changed, expected=original)
    with pytest.raises(ValueError, match="其他页面更新"):
        repository.save(original, expected=original)
    assert repository.get(original.draft_id) == changed
    repository.move_to_trash(original.draft_id)
    assert repository.list() == ()
    with pytest.raises(ValueError, match="其他页面更新"):
        repository.save(original, expected=changed)
    assert repository.restore(original.draft_id) == changed
    assert repository.trash().list() == ()
