from __future__ import annotations

from datetime import date
import traceback
from typing import Any, Callable

import httpx
import pandas as pd
import pytest

from compass.data.base import DailyBarRequest, ProviderError, ProviderErrorKind
from compass.data.providers.akshare_provider import AkshareProvider
from compass.data.providers.baostock_provider import BaostockProvider
from compass.data.providers.http_provider import HttpProvider
from compass.domain.market import InstrumentId


_SECRET = "sentinel-upstream-secret"


def _request() -> DailyBarRequest:
    return DailyBarRequest(
        InstrumentId.parse("SSE.600000"), date(2026, 7, 20), date(2026, 7, 21)
    )


def _assert_translated_without_secret(call: Callable[[], object]) -> None:
    with pytest.raises(ProviderError) as caught:
        call()

    error = caught.value
    formatted = "".join(traceback.format_exception(error))
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
    assert _SECRET not in formatted
    assert _SECRET not in str(error)
    assert _SECRET not in repr(error)


class RaisingClient:
    def __getattr__(self, name: str) -> Callable[..., object]:
        def raise_secret(*args: object, **kwargs: object) -> object:
            raise ConnectionError(f"https://upstream.invalid/?token={_SECRET}")

        return raise_secret


def test_akshare_suppresses_raw_upstream_exception_traceback() -> None:
    _assert_translated_without_secret(lambda: AkshareProvider(client=RaisingClient()).fetch_daily(_request()))


def test_baostock_suppresses_raw_upstream_exception_traceback() -> None:
    _assert_translated_without_secret(lambda: BaostockProvider(client=RaisingClient()).fetch_daily(_request()))


def test_http_suppresses_raw_transport_exception_traceback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"https://upstream.invalid/?token={_SECRET}", request=request
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = HttpProvider(
            url_template="https://market.example/bars/{symbol}",
            field_mapping={},
            client=client,
        )
        _assert_translated_without_secret(lambda: provider.fetch_daily(_request()))


@pytest.mark.parametrize(
    ("provider_factory", "malformed_client"),
    [
        (AkshareProvider, type("AkClient", (), {"stock_zh_a_hist": lambda self, **kwargs: pd.DataFrame({"bad": [1]})})()),
        (
            BaostockProvider,
            type(
                "BaoClient",
                (),
                {"query_history_k_data_plus": lambda self, *args, **kwargs: object()},
            )(),
        ),
    ],
)
def test_provider_schema_failures_are_malformed_response(
    provider_factory: Any, malformed_client: object
) -> None:
    with pytest.raises(ProviderError) as error:
        provider_factory(client=malformed_client).fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.MALFORMED_RESPONSE


@pytest.mark.parametrize("exception_type", [KeyError, AttributeError, TypeError, ValueError])
def test_schema_exception_types_are_never_misclassified_as_auth_or_network(
    exception_type: type[Exception],
) -> None:
    class Client:
        def stock_zh_a_hist(self, **kwargs: str) -> pd.DataFrame:
            raise exception_type("authentication rate limit text in malformed schema")

    with pytest.raises(ProviderError) as error:
        AkshareProvider(client=Client()).fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.MALFORMED_RESPONSE


@pytest.mark.parametrize(
    "network_error",
    [
        httpx.ConnectError("authentication service unreachable"),
        ConnectionError("rate limit endpoint connection failed"),
        TimeoutError("authentication rate limit service timed out"),
        OSError("authentication rate limit socket failure"),
    ],
)
def test_network_exception_types_take_precedence_over_message_heuristics(
    network_error: Exception,
) -> None:
    class Client:
        def stock_zh_a_hist(self, **kwargs: str) -> pd.DataFrame:
            raise network_error

    with pytest.raises(ProviderError) as error:
        AkshareProvider(client=Client()).fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.NETWORK
