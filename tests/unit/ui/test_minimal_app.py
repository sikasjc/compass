from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from compass.domain.market import InstrumentId
from compass.services.task_manager import TaskStatus
from compass.ui.app import NAV_ITEMS, ROUTES, AppViewModels, register_pages
from compass.ui.components.charts import MarketBarPoint
from compass.ui.pages.data import (
    DataSyncHistoryEntry,
    DataSyncRange,
    MarketDataPreview,
    instrument_display_label,
    sync_coverage_notice,
)
from compass.ui.pages.watchlists import (
    WatchlistDraft,
    WatchlistDataRange,
    WatchlistEntry,
    WatchlistFormModel,
    WatchlistPageModel,
    WatchlistPageState,
)
from compass.ui.pages.home import _market_readiness


NOW = datetime(2026, 8, 9, 10, tzinfo=ZoneInfo("Asia/Shanghai"))


class WatchlistGateway:
    def __init__(self) -> None:
        self.entry: WatchlistEntry | None = None
        self.saved: WatchlistDraft | None = None

    def primary(self) -> WatchlistEntry | None:
        return self.entry

    def save_primary(self, draft: WatchlistDraft) -> None:
        self.saved = draft


def test_application_exposes_market_data_and_strategy_pages() -> None:
    assert tuple(item.label for item in NAV_ITEMS) == (
        "开始",
        "今日信号",
        "账户",
        "策略回测",
        "策略实验室",
        "行情数据",
        "标的池",
        "设置",
        "日志",
    )
    assert ROUTES == (
        "/",
        "/signals",
        "/account",
        "/backtests",
        "/strategies",
        "/data",
        "/watchlists",
        "/settings",
        "/logs",
        "/strategies/editor",
        "/strategies/preview",
        "/strategies/release",
        "/strategies/templates",
    )
    registered: list[str] = []

    def registrar(route: str):  # type: ignore[no-untyped-def]
        registered.append(route)
        return lambda handler: handler

    register_pages(AppViewModels(), registrar=registrar)
    assert tuple(registered) == ROUTES


def test_single_watchlist_model_saves_one_canonical_pool() -> None:
    gateway = WatchlistGateway()
    model = WatchlistPageModel(gateway)
    result = model.save(WatchlistFormModel("关注标的", "SZSE.159949\nSSE.510300"))

    assert result.errors == {}
    assert gateway.saved == WatchlistDraft(
        "关注标的",
        (
            InstrumentId.parse("SSE.510300"),
            InstrumentId.parse("SZSE.159949"),
        ),
    )
    assert model.state().entry is None


def test_watchlist_state_includes_current_market_data_ranges() -> None:
    gateway = WatchlistGateway()
    instrument = InstrumentId.parse("SSE.510300")
    expected = WatchlistDataRange(
        instrument,
        date(2024, 8, 9),
        date(2026, 8, 7),
    )
    model = WatchlistPageModel(gateway, lambda: (expected,))

    assert model.state().data_ranges == (expected,)


def test_workbench_market_readiness_uses_the_complete_watchlist_ranges() -> None:
    second = InstrumentId.parse("SSE.000300")
    completed = date(2026, 9, 11)
    state = WatchlistPageState(
        WatchlistEntry("primary", "关注标的", (second, InstrumentId.parse("SSE.510300")), True),
        (
            WatchlistDataRange(second, date(2024, 9, 11), completed),
            WatchlistDataRange(
                InstrumentId.parse("SSE.510300"),
                date(2024, 9, 11),
                completed,
            ),
        ),
    )

    assert _market_readiness(state, completed) == (completed, True)


def test_watchlist_can_retry_one_instrument() -> None:
    gateway = WatchlistGateway()
    instrument = InstrumentId.parse("SSE.000688")
    retried: list[InstrumentId] = []
    model = WatchlistPageModel(
        gateway,
        instrument_retry=lambda item: retried.append(item),
    )

    model.retry_instrument(instrument)

    assert retried == [instrument]


def test_sync_history_can_confirm_download_and_reuse_counts() -> None:
    entry = DataSyncHistoryEntry(
        1,
        "akshare",
        DataSyncRange(date(2024, 8, 9), date(2026, 8, 7)),
        TaskStatus.SUCCEEDED,
        NOW,
        NOW,
        instrument_count=2,
        completed_instrument_count=2,
        downloaded_rows=10,
        reused_rows=900,
        remaining_requested_sessions=0,
    )

    assert entry.downloaded_rows == 10
    assert entry.reused_rows == 900
    assert entry.remaining_requested_sessions == 0


def test_instrument_label_uses_latest_synced_market_metadata() -> None:
    instrument = InstrumentId.parse("SZSE.159326")
    bar = MarketBarPoint("2026-08-11", 1.0, 1.1, 0.9, 1.05, 1000.0)
    preview = MarketDataPreview(
        instrument,
        "tencent",
        "manifest-159326",
        1,
        "2026-08-11",
        "2026-08-11",
        (bar,),
        True,
        0,
        instrument_name="创业板成长ETF",
    )

    assert instrument_display_label(instrument, (preview,)) == (
        "创业板成长ETF（SZSE.159326）"
    )
    assert instrument_display_label(InstrumentId.parse("SZSE.159999"), (preview,)) == (
        "SZSE.159999"
    )


def test_sync_coverage_notice_separates_pre_listing_history_from_real_gaps() -> None:
    entry = DataSyncHistoryEntry(
        2,
        "tencent",
        DataSyncRange(date(2024, 8, 11), date(2026, 8, 11)),
        TaskStatus.SUCCEEDED,
        NOW,
        NOW,
        instrument_count=2,
        completed_instrument_count=2,
        downloaded_rows=100,
        reused_rows=200,
        remaining_requested_sessions=196,
    )
    bar = MarketBarPoint("2026-08-11", 4.0, 4.2, 3.9, 4.1, 1000.0)
    previews = (
        MarketDataPreview(
            InstrumentId.parse("SZSE.159326"),
            "tencent",
            "manifest-159326",
            1,
            "2024-09-09",
            "2026-08-11",
            (bar,),
            True,
            20,
            leading_missing_session_count=20,
        ),
        MarketDataPreview(
            InstrumentId.parse("SZSE.159382"),
            "tencent",
            "manifest-159382",
            1,
            "2025-05-09",
            "2026-08-11",
            (bar,),
            True,
            176,
            leading_missing_session_count=176,
        ),
    )

    text, css_class = sync_coverage_notice(
        entry,
        previews,
        use_current_previews=True,
    )

    assert text == (
        "同步成功：2/2 个标的。2 个标的仅有 196 个上市前或较早历史交易日未覆盖；"
        "现有行情区间连续，不属于内部缺口。"
    )
    assert css_class == "text-xs text-amber-800"
