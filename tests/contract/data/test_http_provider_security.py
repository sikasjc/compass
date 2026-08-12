from __future__ import annotations

from datetime import date
import traceback
from typing import Any

import httpx
import pytest

from compass.data.base import (
    DailyBarRequest,
    ProviderConfigurationError,
    ProviderError,
    ProviderErrorKind,
)
from compass.data.providers.http_provider import HttpProvider
from compass.domain.market import InstrumentId


_MAPPING = {
    "day": "date",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "a": "amount",
}
_RECORD = {
    "day": "2026-07-20",
    "o": 10.0,
    "h": 10.2,
    "l": 9.9,
    "c": 10.1,
    "v": 1000.0,
    "a": 10100.0,
}


def _request() -> DailyBarRequest:
    return DailyBarRequest(
        InstrumentId.parse("SSE.600000"), date(2026, 7, 20), date(2026, 7, 21)
    )


def _client(payload: Any = None, status_code: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"data": [_RECORD]} if payload is None else payload,
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "url",
    [
        "http://market.example/bars/{symbol}",
        "https://user:password@market.example/bars/{symbol}",
        "https://localhost/bars/{symbol}",
        "https://prices.localhost/bars/{symbol}",
        "https://metadata.google.internal/bars/{symbol}",
        "https://instance-data.ec2.internal/bars/{symbol}",
        "https://127.0.0.1/bars/{symbol}",
        "https://10.0.0.1/bars/{symbol}",
        "https://169.254.169.254/latest/meta-data",
        "https://224.0.0.1/bars/{symbol}",
        "https://240.0.0.1/bars/{symbol}",
        "https://0.0.0.0/bars/{symbol}",
        "https://[::1]/bars/{symbol}",
        "https://[fe80::1]/bars/{symbol}",
    ],
)
def test_http_rejects_insecure_or_nonpublic_destinations(url: str) -> None:
    with _client() as client:
        with pytest.raises(ProviderConfigurationError):
            HttpProvider(url_template=url, field_mapping=_MAPPING, client=client)


def test_http_allows_a_public_literal_ip_over_https() -> None:
    with _client() as client:
        provider = HttpProvider(
            url_template="https://8.8.8.8/bars/{symbol}",
            field_mapping=_MAPPING,
            client=client,
        )

    assert provider.name == "http"


@pytest.mark.parametrize(
    "url",
    [
        "{exchange}://market.example/bars/{symbol}",
        "https://{exchange}.example/bars/{symbol}",
        "https://market.example:{symbol}/bars",
        "https://market.example/bars?{exchange}=value",
        "https://market.example/bars#{symbol}",
    ],
)
def test_http_placeholders_are_for_path_or_query_values_only(url: str) -> None:
    with _client() as client:
        with pytest.raises(ProviderConfigurationError):
            HttpProvider(url_template=url, field_mapping=_MAPPING, client=client)


def test_http_allows_placeholders_in_paths_and_query_values() -> None:
    with _client() as client:
        provider = HttpProvider(
            url_template=(
                "https://market.example/{exchange}/{symbol}"
                "?instrument={instrument}&start={start}&end={end}&type={asset_type}"
            ),
            field_mapping=_MAPPING,
            client=client,
        )
        result = provider.fetch_daily(_request())

    assert result.index.tolist() == [pytest.importorskip("pandas").Timestamp("2026-07-20")]


@pytest.mark.parametrize(
    "query_key",
    [
        "credential",
        "credentials",
        "passwd",
        "password",
        "passWord",
        "session",
        "sessionid",
        "prefix_sessionid_suffix",
        "session_id",
        "sig",
        "signature",
        "auth",
        "authorization",
        "bearer",
        "access_token",
        "Access-Token",
        "token",
        "secret",
        "api_key",
        "apikey",
        "API-KEY",
        "key",
        "client_secret",
        "private_key",
        "secret_key",
        "x_api_key",
        "refresh_token",
        "session_token",
        "auth_token",
        "clientSecret",
        "accessToken",
    ],
)
def test_http_rejects_normalized_secret_query_keys(query_key: str) -> None:
    url = f"https://market.example/bars/{{symbol}}?{query_key}=sentinel-query-secret"

    with _client() as client:
        with pytest.raises(ProviderConfigurationError) as error:
            HttpProvider(url_template=url, field_mapping=_MAPPING, client=client)

    assert "sentinel-query-secret" not in str(error.value)
    assert "sentinel-query-secret" not in repr(error.value)


@pytest.mark.parametrize(
    "query_key",
    [
        "monkey",
        "hockey",
        "turkey",
        "keyboard",
        "authors",
        "session_count",
        "api_version",
        "tokenizer",
        "secretary",
        "authentication_mode",
        "sessioncount",
        "bearerstatus",
        "accesstokenizer",
        "apikeyboard",
        "keynote",
    ],
)
def test_http_secret_key_matching_does_not_reject_unrelated_words(query_key: str) -> None:
    with _client() as client:
        provider = HttpProvider(
            url_template=f"https://market.example/bars/{{symbol}}?{query_key}=value",
            field_mapping=_MAPPING,
            client=client,
        )

    assert provider.name == "http"


@pytest.mark.parametrize(
    "host",
    [
        "2130706433",
        "127.1",
        "0x7f000001",
        "017700000001",
    ],
)
def test_http_rejects_legacy_numeric_loopback_hosts(host: str) -> None:
    with _client() as client:
        with pytest.raises(ProviderConfigurationError):
            HttpProvider(
                url_template=f"https://{host}/bars/{{symbol}}",
                field_mapping=_MAPPING,
                client=client,
            )


def test_http_allows_a_legacy_numeric_public_host() -> None:
    with _client() as client:
        provider = HttpProvider(
            url_template="https://134744072/bars/{symbol}",
            field_mapping=_MAPPING,
            client=client,
        )

    assert provider.name == "http"


def test_http_template_parser_errors_are_configuration_errors_without_chaining() -> None:
    with _client() as client:
        with pytest.raises(ProviderConfigurationError) as error:
            HttpProvider(
                url_template="https://market.example/bars/{symbol",
                field_mapping=_MAPPING,
                client=client,
            )

    assert error.value.kind is ProviderErrorKind.CONFIGURATION
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True


def test_http_optional_application_envelope_accepts_only_the_success_value() -> None:
    with _client({"status": "ok", "bars": [_RECORD]}) as client:
        result = HttpProvider(
            url_template="https://market.example/bars/{symbol}",
            field_mapping=_MAPPING,
            client=client,
            data_field="bars",
            success_field="status",
            success_value="ok",
        ).fetch_daily(_request())

    assert len(result) == 1


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        ("error", ProviderErrorKind.MALFORMED_RESPONSE),
        ("unauthorized", ProviderErrorKind.AUTHENTICATION),
        ("rate_limited", ProviderErrorKind.RATE_LIMIT),
    ],
)
def test_http_application_error_status_is_never_accepted_as_data(
    status: str, kind: ProviderErrorKind
) -> None:
    with _client({"status": status, "bars": [_RECORD]}) as client:
        provider = HttpProvider(
            url_template="https://market.example/bars/{symbol}",
            field_mapping=_MAPPING,
            client=client,
            data_field="bars",
            success_field="status",
            success_value="ok",
            business_error_kinds={
                "unauthorized": ProviderErrorKind.AUTHENTICATION,
                "rate_limited": ProviderErrorKind.RATE_LIMIT,
            },
        )
        with pytest.raises(ProviderError) as error:
            provider.fetch_daily(_request())

    assert error.value.kind is kind


def test_http_application_envelope_requires_the_configured_status_field() -> None:
    with _client({"bars": [_RECORD]}) as client:
        provider = HttpProvider(
            url_template="https://market.example/bars/{symbol}",
            field_mapping=_MAPPING,
            client=client,
            data_field="bars",
            success_field="status",
            success_value="ok",
        )
        with pytest.raises(ProviderError) as error:
            provider.fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.MALFORMED_RESPONSE


@pytest.mark.parametrize(
    ("status_code", "kind"),
    [
        (400, ProviderErrorKind.MALFORMED_RESPONSE),
        (404, ProviderErrorKind.MALFORMED_RESPONSE),
        (401, ProviderErrorKind.AUTHENTICATION),
        (403, ProviderErrorKind.AUTHENTICATION),
        (429, ProviderErrorKind.RATE_LIMIT),
        (500, ProviderErrorKind.NETWORK),
        (503, ProviderErrorKind.NETWORK),
    ],
)
def test_http_status_codes_have_stable_error_kinds(
    status_code: int, kind: ProviderErrorKind
) -> None:
    with _client({"error": "sentinel-http-secret"}, status_code) as client:
        provider = HttpProvider(
            url_template="https://market.example/bars/{symbol}",
            field_mapping=_MAPPING,
            client=client,
        )
        with pytest.raises(ProviderError) as error:
            provider.fetch_daily(_request())

    assert error.value.kind is kind
    assert "sentinel-http-secret" not in str(error.value)


@pytest.mark.parametrize(
    "transport_error",
    [
        httpx.ConnectError("sentinel-transport-secret"),
        httpx.ReadTimeout("sentinel-transport-secret"),
    ],
)
def test_http_transport_and_timeout_failures_are_network_errors(
    transport_error: httpx.TransportError,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        transport_error.request = request
        raise transport_error

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = HttpProvider(
            url_template="https://market.example/bars/{symbol}",
            field_mapping=_MAPPING,
            client=client,
        )
        with pytest.raises(ProviderError) as error:
            provider.fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.NETWORK


@pytest.mark.parametrize(
    "redirect_target",
    [
        "http://market.example/final",
        "https://127.0.0.1/final",
        "https://169.254.169.254/latest/meta-data",
        "https://metadata.google.internal/final",
        "https://user:password@market.example/final",
    ],
)
def test_http_rejects_unsafe_redirect_destinations(redirect_target: str) -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if str(request.url) == "https://market.example/start/600000":
            return httpx.Response(302, headers={"location": redirect_target})
        return httpx.Response(200, json={"data": [_RECORD]})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        provider = HttpProvider(
            url_template="https://market.example/start/{symbol}",
            field_mapping=_MAPPING,
            client=client,
        )
        with pytest.raises(ProviderError) as error:
            provider.fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.CONFIGURATION
    assert requested_urls == ["https://market.example/start/600000"]


def test_http_allows_redirects_when_every_destination_is_safe() -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if str(request.url) == "https://market.example/start/600000":
            return httpx.Response(302, headers={"location": "/middle"})
        if str(request.url) == "https://market.example/middle":
            return httpx.Response(
                307, headers={"location": "https://market.example:443/final"}
            )
        return httpx.Response(200, json={"data": [_RECORD]})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        result = HttpProvider(
            url_template="https://market.example/start/{symbol}",
            field_mapping=_MAPPING,
            client=client,
        ).fetch_daily(_request())

    assert len(result) == 1
    assert requested_urls == [
        "https://market.example/start/600000",
        "https://market.example/middle",
        "https://market.example/final",
    ]


def test_http_blocks_cross_origin_redirect_before_forwarding_default_authorization() -> None:
    requested_urls: list[str] = []
    received_authorization: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        received_authorization.append(request.headers.get("authorization"))
        if str(request.url) == "https://market.example/start/600000":
            return httpx.Response(302, headers={"location": "https://api.example/final"})
        return httpx.Response(200, json={"data": [_RECORD]})

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        headers={"Authorization": "Bearer sentinel-default-authorization"},
    ) as client:
        provider = HttpProvider(
            url_template="https://market.example/start/{symbol}",
            field_mapping=_MAPPING,
            client=client,
        )
        with pytest.raises(ProviderError) as error:
            provider.fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.CONFIGURATION
    assert requested_urls == ["https://market.example/start/600000"]
    assert received_authorization == ["Bearer sentinel-default-authorization"]


@pytest.mark.parametrize(
    "location",
    [
        "/final?access_token=sentinel-redirect-query-secret",
        "/final?client_secret=sentinel-redirect-query-secret",
    ],
)
def test_http_blocks_secret_query_redirect_before_sending_or_leaking(location: str) -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if str(request.url) == "https://market.example/start/600000":
            return httpx.Response(302, headers={"location": location})
        return httpx.Response(200, json={"data": [_RECORD]})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        provider = HttpProvider(
            url_template="https://market.example/start/{symbol}",
            field_mapping=_MAPPING,
            client=client,
        )
        with pytest.raises(ProviderError) as error:
            provider.fetch_daily(_request())

    formatted = "".join(traceback.format_exception(error.value))
    assert error.value.kind is ProviderErrorKind.CONFIGURATION
    assert requested_urls == ["https://market.example/start/600000"]
    assert "sentinel-redirect-query-secret" not in str(error.value)
    assert "sentinel-redirect-query-secret" not in repr(error.value)
    assert "sentinel-redirect-query-secret" not in formatted


def test_http_redirect_loop_is_rejected_before_repeating_a_request() -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if str(request.url) == "https://market.example/start/600000":
            return httpx.Response(302, headers={"location": "/middle"})
        return httpx.Response(
            302, headers={"location": "https://market.example/start/600000"}
        )

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        provider = HttpProvider(
            url_template="https://market.example/start/{symbol}",
            field_mapping=_MAPPING,
            client=client,
        )
        with pytest.raises(ProviderError) as error:
            provider.fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.MALFORMED_RESPONSE
    assert requested_urls == [
        "https://market.example/start/600000",
        "https://market.example/middle",
    ]


def test_http_redirect_limit_is_enforced_before_sending_an_extra_hop() -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        hop = len(requested_urls)
        return httpx.Response(
            302, headers={"location": f"https://market.example/hop/{hop}"}
        )

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        provider = HttpProvider(
            url_template="https://market.example/start/{symbol}",
            field_mapping=_MAPPING,
            client=client,
            max_redirects=2,
        )
        with pytest.raises(ProviderError) as error:
            provider.fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.MALFORMED_RESPONSE
    assert requested_urls == [
        "https://market.example/start/600000",
        "https://market.example/hop/1",
        "https://market.example/hop/2",
    ]


@pytest.mark.parametrize("location", [None, "https://[::1"])
def test_http_malformed_redirect_location_is_typed_and_not_followed(
    location: str | None,
) -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        headers = {} if location is None else {"location": location}
        return httpx.Response(302, headers=headers)

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        provider = HttpProvider(
            url_template="https://market.example/start/{symbol}",
            field_mapping=_MAPPING,
            client=client,
        )
        with pytest.raises(ProviderError) as error:
            provider.fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.MALFORMED_RESPONSE
    assert requested_urls == ["https://market.example/start/600000"]
