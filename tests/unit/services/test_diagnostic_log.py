from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest

from compass.data.base import DailyBarRequest, ProviderError
from compass.data.providers.akshare_provider import AkshareProvider
from compass.domain.market import InstrumentId
from compass.services.diagnostic_log import (
    configure_application_logging,
    diagnostic_request,
    read_application_logs,
)


def test_application_log_is_readable_filterable_and_secret_safe(tmp_path: Path) -> None:
    path = configure_application_logging(tmp_path, "DEBUG")
    logger = logging.getLogger("compass.test")

    logger.info(
        "request url=https://example.test/bars?symbol=510300&token=highly-secret"
    )
    logger.error("response authentication failed Authorization: Bearer hidden-value")

    entries = read_application_logs(path, query="authentication")
    assert len(entries) == 1
    assert entries[0].level == "ERROR"
    assert "authentication failed" in entries[0].message
    combined = "\n".join(item.message for item in read_application_logs(path))
    assert "highly-secret" not in combined
    assert "hidden-value" not in combined
    assert "[redacted]" in combined


def test_provider_diagnostic_logs_endpoint_and_response_code(tmp_path: Path) -> None:
    path = configure_application_logging(tmp_path, "INFO")

    class Response:
        error_code = "10001001"
        error_msg = "用户未登录"
        fields = ("date", "close")

    result = diagnostic_request(
        provider="BaoStock",
        transport="TCP",
        operation="login",
        endpoint="tcp://public-api.baostock.com:10030",
        details="session=anonymous",
        call=Response,
    )

    assert isinstance(result, Response)
    messages = tuple(item.message for item in read_application_logs(path, query="BaoStock"))
    assert any("tcp://public-api.baostock.com:10030" in item for item in messages)
    assert any("code=10001001" in item and "用户未登录" in item for item in messages)


def test_failed_provider_diagnostic_keeps_safe_exception_summary(tmp_path: Path) -> None:
    path = configure_application_logging(tmp_path, "INFO")

    def fail() -> object:
        raise PermissionError("token=highly-secret 权限不足")

    with pytest.raises(PermissionError):
        diagnostic_request(
            provider="东方财富",
            transport="HTTP",
            operation="fund_etf_hist_em",
            endpoint="https://push2his.eastmoney.com/api/qt/stock/kline/get",
            details="instrument=SSE.510300",
            call=fail,
        )

    entries = read_application_logs(path, level="ERROR")
    assert len(entries) == 1
    assert "PermissionError" in entries[0].message
    assert "highly-secret" not in entries[0].message
    assert "[redacted]" in entries[0].message


def test_akshare_failure_log_identifies_eastmoney_request(tmp_path: Path) -> None:
    path = configure_application_logging(tmp_path, "INFO")

    class Client:
        @staticmethod
        def stock_zh_index_daily_em(**parameters: object) -> object:
            del parameters
            raise PermissionError("您没有访问该接口的权限")

    provider = AkshareProvider(client=Client())
    with pytest.raises(ProviderError):
        provider.fetch_daily(
            DailyBarRequest(
                InstrumentId.parse("SSE.000300"),
                date(2026, 8, 1),
                date(2026, 8, 7),
            )
        )

    messages = tuple(item.message for item in read_application_logs(path))
    assert any("stock_zh_index_daily_em" in item for item in messages)
    assert any(
        "https://push2his.eastmoney.com/api/qt/stock/kline/get" in item
        for item in messages
    )
    assert any("instrument=SSE.000300" in item for item in messages)
    assert any("PermissionError" in item and "outcome=failed" in item for item in messages)
