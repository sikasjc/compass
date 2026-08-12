from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Callable, Mapping, NoReturn, Protocol, Sequence, TypeAlias
from urllib.parse import urlsplit, urlunsplit

import httpx
import pandas as pd  # type: ignore[import-untyped]

from compass.domain.market import AssetType, Exchange, InstrumentId
from compass.domain.trading import CorporateAction


@dataclass(frozen=True, slots=True)
class DailyBarRequest:
    instrument: InstrumentId
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("daily bar request start must be on or before end")


class ProviderErrorKind(StrEnum):
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    MALFORMED_RESPONSE = "malformed_response"
    CAPABILITY = "capability"


_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(token|api[-_]?key|secret|password|authorization|signature)(\s*=\s*)[^\s&]+"
)
_BEARER_CREDENTIAL = re.compile(r"(?i)(authorization\s*:\s*bearer\s+|bearer\s+)[^\s]+")


def _safe_provider_message(message: str) -> str:
    """Redact common credential forms and complete URL query strings."""

    redacted = _SENSITIVE_ASSIGNMENT.sub(r"\1\2[redacted]", message)
    redacted = _BEARER_CREDENTIAL.sub(r"\1[redacted]", redacted)
    for candidate in re.findall(r"https?://[^\s]+", redacted):
        parts = urlsplit(candidate)
        if parts.query or parts.username or parts.password:
            safe_host = parts.hostname or ""
            if parts.port is not None:
                safe_host = f"{safe_host}:{parts.port}"
            safe_url = urlunsplit((parts.scheme, safe_host, parts.path, "[redacted]", ""))
            redacted = redacted.replace(candidate, safe_url)
    return redacted


class ProviderError(RuntimeError):
    """A stable, secret-safe market data provider failure."""

    def __init__(self, kind: ProviderErrorKind, provider: str, message: str) -> None:
        self.kind = kind
        self.provider = provider
        self.message = _safe_provider_message(message)
        super().__init__(f"{provider}: {self.message}")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(kind={self.kind.value!r}, provider={self.provider!r}, "
            f"message={self.message!r})"
        )


class ProviderConfigurationError(ProviderError):
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(ProviderErrorKind.CONFIGURATION, provider, message)


class ProviderCapabilityError(ProviderError, NotImplementedError):
    """Raised when a provider does not expose a requested data capability."""

    def __init__(self, provider: str, capability: str) -> None:
        self.capability = capability
        super().__init__(
            ProviderErrorKind.CAPABILITY,
            provider,
            f"provider does not support {capability}",
        )


def translate_provider_error(provider: str, error: Exception) -> ProviderError:
    """Classify an upstream exception without copying its potentially secret text."""

    if isinstance(error, ProviderError):
        return error
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    lowered = str(error).lower()
    if status_code in (401, 403):
        return ProviderError(ProviderErrorKind.AUTHENTICATION, provider, "authentication failed")
    if status_code == 429:
        return ProviderError(ProviderErrorKind.RATE_LIMIT, provider, "request rate limit exceeded")
    if status_code is not None and 400 <= status_code < 500:
        return ProviderError(
            ProviderErrorKind.MALFORMED_RESPONSE, provider, "upstream request was rejected"
        )
    if status_code is not None and status_code >= 500:
        return ProviderError(ProviderErrorKind.NETWORK, provider, "network request failed")
    if isinstance(error, PermissionError):
        return ProviderError(ProviderErrorKind.AUTHENTICATION, provider, "authentication failed")
    if isinstance(error, (httpx.TransportError, ConnectionError, TimeoutError, OSError)):
        return ProviderError(ProviderErrorKind.NETWORK, provider, "network request failed")
    if error.__class__.__module__.startswith(("requests.", "urllib3.")):
        return ProviderError(ProviderErrorKind.NETWORK, provider, "network request failed")
    if isinstance(error, (KeyError, AttributeError, TypeError, ValueError)):
        return ProviderError(
            ProviderErrorKind.MALFORMED_RESPONSE, provider, "provider returned malformed data"
        )
    if any(
        marker in lowered
        for marker in ("unauthorized", "authentication", "permission denied", "权限", "未登录")
    ):
        return ProviderError(ProviderErrorKind.AUTHENTICATION, provider, "authentication failed")
    if any(marker in lowered for marker in ("rate limit", "too many requests", "频率", "每分钟")):
        return ProviderError(ProviderErrorKind.RATE_LIMIT, provider, "request rate limit exceeded")
    return ProviderError(
        ProviderErrorKind.MALFORMED_RESPONSE, provider, "provider returned malformed data"
    )


def raise_translated_provider_error(provider: str, error: Exception) -> NoReturn:
    """Raise a translated error while suppressing raw upstream traceback context."""

    raise translate_provider_error(provider, error) from None


InstrumentTypeResolver: TypeAlias = (
    Callable[[InstrumentId], AssetType] | Mapping[InstrumentId, AssetType]
)


def default_instrument_type(instrument: InstrumentId) -> AssetType:
    """Infer common mainland stocks/ETFs by documented exchange code families.

    Unknown families are rejected instead of being silently treated as stocks.
    Applications with an instrument master should inject its resolver or mapping.
    """

    prefixes = {
        Exchange.SSE: {
            AssetType.INDEX: ("000",),
            AssetType.STOCK: ("60", "68"),
            AssetType.ETF: ("50", "51", "52", "56", "58"),
        },
        Exchange.SZSE: {
            AssetType.INDEX: ("399",),
            AssetType.STOCK: ("00", "30"),
            AssetType.ETF: ("15", "16", "18"),
        },
    }
    for asset_type, families in prefixes[instrument.exchange].items():
        if instrument.code.startswith(families):
            return asset_type
    raise ValueError(f"unknown instrument code family: {instrument.exchange.value}")


def resolve_instrument_type(
    provider: str, resolver: InstrumentTypeResolver, instrument: InstrumentId
) -> AssetType:
    try:
        if isinstance(resolver, Mapping):
            return resolver[instrument]
        return resolver(instrument)
    except (KeyError, ValueError) as error:
        raise ProviderCapabilityError(provider, "instrument_type_resolution") from error


class MarketDataProvider(Protocol):
    name: str

    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        """Return canonical daily bars for the inclusive request date range."""

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        """Return a current snapshot or raise ProviderCapabilityError."""

    def fetch_corporate_actions(self, request: DailyBarRequest) -> Sequence[CorporateAction]:
        """Return actions for the request range or raise ProviderCapabilityError."""
