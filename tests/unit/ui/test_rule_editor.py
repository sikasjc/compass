from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from compass.domain.market import AssetType, InstrumentId
from compass.storage.strategy_draft_repository import StrategyDraftRepository
from compass.strategies.base import StrategyFrequency
from compass.strategies.registry import StrategyRegistry
from compass.strategies.rule_document import RuleSide, default_rule_document
from compass.strategies.rule_dsl import RuleDslStrategy
from compass.ui.pages.strategies import (
    StrategyInstance,
    StrategyPageModel,
    StrategyPool,
)


NOW = datetime(2026, 8, 12, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
INSTRUMENT = InstrumentId.parse("SSE.510300")


class Tasks:
    pass


class Gateway:
    def __init__(self) -> None:
        self.created = None

    def list(self):  # type: ignore[no-untyped-def]
        return ()

    def pools(self):  # type: ignore[no-untyped-def]
        return ()

    def pool(self, watchlist_id: str) -> StrategyPool:
        assert watchlist_id == "main"
        return StrategyPool(
            "main",
            "main-0123456789abcdef",
            (INSTRUMENT,),
            AssetType.ETF,
            StrategyFrequency.DAILY,
        )

    def create(self, draft):  # type: ignore[no-untyped-def]
        self.created = draft
        return StrategyInstance(
            "strategy-one-v1",
            "strategy-one",
            1,
            draft.name,
            draft.strategy_type,
            draft.strategy_version,
            draft.watchlist_id,
            draft.pool_snapshot_id,
            draft.frequency,
            draft.parameters,
            True,
            NOW,
        )


def bars() -> pd.DataFrame:
    values = [3.0] * 60 + [4.0, 3.0]
    return pd.DataFrame(
        {
            "open": values,
            "high": [value + 0.1 for value in values],
            "low": [value - 0.1 for value in values],
            "close": values,
            "volume": [1000.0] * len(values),
            "amount": [3000.0] * len(values),
        },
        index=pd.date_range("2026-05-01", periods=len(values), freq="D"),
    )


def model(tmp_path: Path) -> tuple[StrategyPageModel, Gateway]:
    registry = StrategyRegistry()
    registry.register("rule_dsl", RuleDslStrategy)
    gateway = Gateway()
    return (
        StrategyPageModel(
            registry,
            gateway,  # type: ignore[arg-type]
            Tasks(),  # type: ignore[arg-type]
            drafts=StrategyDraftRepository(tmp_path / "strategy-drafts.json"),
            preview_reader=lambda _: bars(),
            clock=lambda: NOW,
        ),
        gateway,
    )


def test_rule_editor_draft_preview_and_publish_flow(tmp_path: Path) -> None:
    page, gateway = model(tmp_path)
    draft = page.new_rule_draft("main")

    preview = page.preview_rule_draft(draft.draft_id, INSTRUMENT)

    assert preview.signals
    assert preview.signals[0].side is RuleSide.BUY
    assert preview.signals[0].target_weight == Decimal("1")
    published = page.publish_rule_draft(draft.draft_id)
    assert published.version == 1
    assert gateway.created is not None
    assert page.rule_drafts() == ()


def test_rule_editor_saves_revised_document_without_publishing(tmp_path: Path) -> None:
    page, gateway = model(tmp_path)
    draft = page.new_rule_draft("main")
    revised = default_rule_document("新版趋势策略")

    saved = page.save_rule_draft(draft.draft_id, revised)

    assert saved.document.name == "新版趋势策略"
    assert gateway.created is None
