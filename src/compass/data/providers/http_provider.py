from __future__ import annotations

from collections.abc import Mapping, Sequence
from ipaddress import IPv4Address, IPv6Address, ip_address
import re
import socket
from string import Formatter
from typing import Any, NoReturn
from urllib.parse import parse_qsl, urlsplit

import httpx
import pandas as pd  # type: ignore[import-untyped]

from compass.data.base import (
    DailyBarRequest,
    InstrumentTypeResolver,
    ProviderCapabilityError,
    ProviderConfigurationError,
    ProviderError,
    ProviderErrorKind,
    default_instrument_type,
    raise_translated_provider_error,
    resolve_instrument_type,
)
from compass.data.normalize import normalize_daily
from compass.domain.market import InstrumentId
from compass.domain.trading import CorporateAction
from compass.security import is_credential_key


_ALLOWED_TEMPLATE_FIELDS = {"instrument", "exchange", "symbol", "start", "end", "asset_type"}
_LEGACY_NUMERIC_IPV4 = re.compile(
    r"(?i)^(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+)){0,3}$"
)
_METADATA_HOSTS = {
    "metadata.google.internal",
    "metadata.google",
    "instance-data",
    "instance-data.ec2.internal",
}
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


def _configuration_error(provider: str, message: str) -> NoReturn:
    raise ProviderConfigurationError(provider, message) from None


def _literal_ip(provider: str, hostname: str) -> IPv4Address | IPv6Address | None:
    try:
        return ip_address(hostname)
    except ValueError:
        if not _LEGACY_NUMERIC_IPV4.fullmatch(hostname):
            return None
    try:
        return IPv4Address(socket.inet_aton(hostname))
    except OSError:
        _configuration_error(provider, "numeric URL host is invalid")


def _validate_destination(
    provider: str, url: str, *, allow_insecure_http: bool
) -> None:
    parts = urlsplit(url)
    allowed_schemes = {"https", "http"} if allow_insecure_http else {"https"}
    if parts.scheme not in allowed_schemes or not parts.netloc:
        _configuration_error(provider, "URL destination must use an allowed absolute URL")
    if parts.username is not None or parts.password is not None:
        _configuration_error(provider, "credentials are not allowed in URL destinations")

    hostname = (parts.hostname or "").rstrip(".").lower()
    if not hostname:
        _configuration_error(provider, "URL destination must include a host")
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname in _METADATA_HOSTS:
        _configuration_error(provider, "URL destination host is not allowed")
    literal_ip = _literal_ip(provider, hostname)
    if literal_ip is not None and (
        not literal_ip.is_global
        or literal_ip.is_private
        or literal_ip.is_loopback
        or literal_ip.is_link_local
        or literal_ip.is_multicast
        or literal_ip.is_reserved
        or literal_ip.is_unspecified
    ):
        _configuration_error(provider, "URL destination IP address must be globally routable")
    # Accessing port performs urllib's range and syntax validation.
    _ = parts.port


def _validate_query_keys(provider: str, url: str, *, reject_placeholders: bool) -> None:
    for key, _ in parse_qsl(urlsplit(url).query, keep_blank_values=True):
        if reject_placeholders and ("{" in key or "}" in key):
            _configuration_error(provider, "URL template placeholders are not allowed in query keys")
        if is_credential_key(key):
            _configuration_error(
                provider, "credential query parameters are not allowed in URLs"
            )


def _normalized_origin(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").rstrip(".").lower()
    effective_port = parts.port
    if effective_port is None:
        effective_port = 443 if scheme == "https" else 80
    return scheme, hostname, effective_port


def _validate_url_template(provider: str, template: str, *, allow_insecure_http: bool) -> None:
    try:
        format_values = {field: "placeholder" for field in _ALLOWED_TEMPLATE_FIELDS}
        for _, field, format_spec, conversion in Formatter().parse(template):
            if field is None:
                continue
            if field not in _ALLOWED_TEMPLATE_FIELDS or format_spec or conversion:
                _configuration_error(provider, "URL template contains an unsupported field")

        scheme_text, separator, authority_and_path = template.partition("://")
        authority = authority_and_path.split("/", 1)[0].split("?", 1)[0]
        if not separator or "{" in scheme_text or "{" in authority:
            _configuration_error(
                provider, "URL template placeholders are not allowed in scheme or authority"
            )
        if "{" in urlsplit(template).fragment or "}" in urlsplit(template).fragment:
            _configuration_error(provider, "URL template placeholders are not allowed in fragments")

        rendered = template.format(**format_values)
        _validate_destination(
            provider, rendered, allow_insecure_http=allow_insecure_http
        )

        _validate_query_keys(provider, template, reject_placeholders=True)
    except ProviderConfigurationError:
        raise
    except (KeyError, TypeError, ValueError):
        _configuration_error(provider, "URL template could not be parsed")


class HttpProvider:
    name = "http"

    def __init__(
        self,
        *,
        url_template: str,
        field_mapping: Mapping[str, str],
        client: httpx.Client,
        data_field: str = "data",
        success_field: str | None = None,
        success_value: object = None,
        business_error_kinds: Mapping[str, ProviderErrorKind] | None = None,
        allow_insecure_http: bool = False,
        max_redirects: int = 5,
        instrument_type_resolver: InstrumentTypeResolver = default_instrument_type,
    ) -> None:
        _validate_url_template(
            self.name, url_template, allow_insecure_http=allow_insecure_http
        )
        if not data_field or "." in data_field:
            raise ProviderConfigurationError(self.name, "data field must be one envelope field") from None
        if success_field is not None and (not success_field or "." in success_field):
            raise ProviderConfigurationError(
                self.name, "success field must be one envelope field"
            ) from None
        if success_field is None and business_error_kinds:
            raise ProviderConfigurationError(
                self.name, "business error kinds require a success field"
            ) from None
        if isinstance(max_redirects, bool) or not isinstance(max_redirects, int) or max_redirects < 0:
            raise ProviderConfigurationError(
                self.name, "max redirects must be a non-negative integer"
            ) from None
        self._url_template = url_template
        self._field_mapping = dict(field_mapping)
        self._client = client
        self._data_field = data_field
        self._success_field = success_field
        self._success_value = success_value
        self._business_error_kinds = dict(business_error_kinds or {})
        self._allow_insecure_http = allow_insecure_http
        self._max_redirects = max_redirects
        self._instrument_type_resolver = instrument_type_resolver

    def _get_with_safe_redirects(self, url: str) -> httpx.Response:
        current_url = str(httpx.URL(url))
        initial_origin = _normalized_origin(current_url)
        visited: set[str] = set()
        redirects_followed = 0
        while True:
            _validate_destination(
                self.name,
                current_url,
                allow_insecure_http=self._allow_insecure_http,
            )
            if current_url in visited:
                raise ProviderError(
                    ProviderErrorKind.MALFORMED_RESPONSE,
                    self.name,
                    "upstream redirect loop detected",
                )
            visited.add(current_url)
            try:
                response = self._client.get(current_url, follow_redirects=False)
            except httpx.RemoteProtocolError as error:
                if "location header" in str(error).lower():
                    raise ProviderError(
                        ProviderErrorKind.MALFORMED_RESPONSE,
                        self.name,
                        "upstream redirect location is invalid",
                    ) from None
                raise
            _validate_destination(
                self.name,
                str(response.url),
                allow_insecure_http=self._allow_insecure_http,
            )
            if response.status_code not in _REDIRECT_STATUS_CODES:
                return response
            location = response.headers.get("location")
            if location is None:
                raise ProviderError(
                    ProviderErrorKind.MALFORMED_RESPONSE,
                    self.name,
                    "upstream redirect is missing a location",
                )
            if redirects_followed >= self._max_redirects:
                raise ProviderError(
                    ProviderErrorKind.MALFORMED_RESPONSE,
                    self.name,
                    "upstream redirect limit exceeded",
                )
            try:
                next_url = str(httpx.URL(current_url).join(location))
            except (httpx.InvalidURL, TypeError, ValueError):
                raise ProviderError(
                    ProviderErrorKind.MALFORMED_RESPONSE,
                    self.name,
                    "upstream redirect location is invalid",
                ) from None
            _validate_destination(
                self.name,
                next_url,
                allow_insecure_http=self._allow_insecure_http,
            )
            _validate_query_keys(self.name, next_url, reject_placeholders=False)
            if _normalized_origin(next_url) != initial_origin:
                raise ProviderConfigurationError(
                    self.name, "cross-origin redirects are not allowed"
                ) from None
            if next_url in visited:
                raise ProviderError(
                    ProviderErrorKind.MALFORMED_RESPONSE,
                    self.name,
                    "upstream redirect loop detected",
                )
            redirects_followed += 1
            current_url = next_url

    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        try:
            asset_type = resolve_instrument_type(
                self.name, self._instrument_type_resolver, request.instrument
            )
            url = self._url_template.format(
                instrument=str(request.instrument),
                exchange=request.instrument.exchange.value,
                symbol=request.instrument.code,
                start=request.start.isoformat(),
                end=request.end.isoformat(),
                asset_type=asset_type.value,
            )
            response = self._get_with_safe_redirects(url)
            response.raise_for_status()
            payload: Any = response.json()
            if not isinstance(payload, dict):
                raise ProviderError(
                    ProviderErrorKind.MALFORMED_RESPONSE,
                    self.name,
                    "response envelope must be an object",
                )
            if self._success_field is not None:
                if self._success_field not in payload:
                    raise ProviderError(
                        ProviderErrorKind.MALFORMED_RESPONSE,
                        self.name,
                        "response envelope is missing its application status",
                    )
                status = payload[self._success_field]
                if status != self._success_value:
                    kind = self._business_error_kinds.get(
                        str(status), ProviderErrorKind.MALFORMED_RESPONSE
                    )
                    raise ProviderError(kind, self.name, "upstream application rejected the request")
            records = payload.get(self._data_field)
            if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
                raise ProviderError(
                    ProviderErrorKind.MALFORMED_RESPONSE,
                    self.name,
                    "response envelope data must be a record list",
                )
            frame = pd.DataFrame.from_records(records)
            return normalize_daily(frame, self._field_mapping).loc[
                pd.Timestamp(request.start) : pd.Timestamp(request.end)
            ]
        except Exception as error:
            raise_translated_provider_error(self.name, error)

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        raise ProviderCapabilityError(self.name, "snapshot")

    def fetch_corporate_actions(self, request: DailyBarRequest) -> Sequence[CorporateAction]:
        raise ProviderCapabilityError(self.name, "corporate_actions")
