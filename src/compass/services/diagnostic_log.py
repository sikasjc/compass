from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
from typing import TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]

from compass.security import is_credential_key


SHANGHAI = ZoneInfo("Asia/Shanghai")
LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_LOGGER_NAME = "compass"
_LOG_FILENAME = "compass.log"
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(token|api[-_]?key|secret|password|authorization|signature)"
    r"(\s*[:=]\s*)[^\s&,;]+"
)
_BEARER = re.compile(r"(?i)(bearer\s+)[^\s&,;]+")
_URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)
T = TypeVar("T")


def safe_diagnostic_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        query = urlencode(
            tuple(
                (key, "[redacted]" if is_credential_key(key) else item)
                for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            )
        )
        return urlunsplit((parsed.scheme, host, parsed.path, query, ""))
    except (TypeError, ValueError):
        return "[invalid-url]"


def safe_diagnostic_text(value: object, *, maximum: int = 1200) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    for candidate in _URL.findall(text):
        text = text.replace(candidate, safe_diagnostic_url(candidate))
    text = _BEARER.sub(r"\1[redacted]", text)
    text = _SENSITIVE_ASSIGNMENT.sub(r"\1\2[redacted]", text)
    if len(text) > maximum:
        return text[: maximum - 1] + "…"
    return text


@dataclass(frozen=True, slots=True)
class DiagnosticLogEntry:
    occurred_at: datetime
    level: str
    category: str
    message: str

    def __post_init__(self) -> None:
        if self.occurred_at.tzinfo != SHANGHAI or self.occurred_at.utcoffset() is None:
            raise ValueError("diagnostic log timestamp must use Asia/Shanghai")
        if self.level not in LOG_LEVELS:
            raise ValueError("diagnostic log level is invalid")
        if type(self.category) is not str or not self.category.startswith(_LOGGER_NAME):
            raise ValueError("diagnostic log category is invalid")
        if type(self.message) is not str or not self.message:
            raise ValueError("diagnostic log message is invalid")


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, SHANGHAI).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "category": record.name,
            "message": safe_diagnostic_text(record.getMessage()),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_application_logging(logs_dir: Path, level: str) -> Path:
    if not isinstance(logs_dir, Path):
        raise TypeError("logs directory must be an exact Path")
    if level not in LOG_LEVELS:
        raise ValueError("LOG_LEVEL_INVALID")
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = (logs_dir / _LOG_FILENAME).resolve()
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    expected = str(path)
    for existing in tuple(logger.handlers):
        if (
            getattr(existing, "_compass_diagnostic", False)
            and str(Path(getattr(existing, "baseFilename", "")).resolve()) != expected
        ):
            logger.removeHandler(existing)
            existing.close()
    handler = next(
        (
            item
            for item in logger.handlers
            if getattr(item, "_compass_diagnostic", False)
            and str(Path(getattr(item, "baseFilename", "")).resolve()) == expected
        ),
        None,
    )
    if handler is None:
        handler = RotatingFileHandler(
            path,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(_JsonLogFormatter())
        handler.setLevel(logging.DEBUG)
        setattr(handler, "_compass_diagnostic", True)
        logger.addHandler(handler)
    logger.info("application log ready level=%s", level)
    return path


def set_application_log_level(level: str) -> None:
    if level not in LOG_LEVELS:
        raise ValueError("LOG_LEVEL_INVALID")
    logging.getLogger(_LOGGER_NAME).setLevel(level)
    logging.getLogger(f"{_LOGGER_NAME}.settings").info(
        "application log level changed level=%s",
        level,
    )


def read_application_logs(
    path: Path,
    *,
    limit: int = 300,
    level: str | None = None,
    query: str = "",
) -> tuple[DiagnosticLogEntry, ...]:
    if not isinstance(path, Path):
        raise TypeError("log path must be an exact Path")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("LOG_LIMIT_INVALID")
    if level is not None and level not in LOG_LEVELS:
        raise ValueError("LOG_LEVEL_INVALID")
    if type(query) is not str or len(query) > 100:
        raise ValueError("LOG_QUERY_INVALID")
    candidates = tuple(
        candidate
        for candidate in (
            path.with_name(f"{path.name}.3"),
            path.with_name(f"{path.name}.2"),
            path.with_name(f"{path.name}.1"),
            path,
        )
        if candidate.is_file()
    )
    entries: list[DiagnosticLogEntry] = []
    lowered_query = query.strip().lower()
    for candidate in candidates:
        for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
            entry = _decode_entry(line)
            if entry is None or level is not None and entry.level != level:
                continue
            if lowered_query and lowered_query not in (
                f"{entry.category} {entry.message}".lower()
            ):
                continue
            entries.append(entry)
    return tuple(reversed(entries[-limit:]))


def _decode_entry(line: str) -> DiagnosticLogEntry | None:
    try:
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            return None
        timestamp = datetime.fromisoformat(str(payload["timestamp"]))
        if timestamp.tzinfo is None:
            return None
        return DiagnosticLogEntry(
            timestamp.astimezone(SHANGHAI),
            str(payload["level"]),
            str(payload["category"]),
            safe_diagnostic_text(payload["message"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _response_summary(value: object) -> str:
    if isinstance(value, pd.DataFrame):
        columns = ",".join(map(str, value.columns[:12]))
        return f"dataframe rows={len(value.index)} columns={columns}"
    code = getattr(value, "error_code", None)
    message = getattr(value, "error_msg", None)
    fields = getattr(value, "fields", None)
    if code is not None:
        summary = f"code={safe_diagnostic_text(code, maximum=80)}"
        if message is not None:
            summary += f" message={safe_diagnostic_text(message, maximum=300)}"
        if fields is not None:
            summary += f" fields={safe_diagnostic_text(fields, maximum=300)}"
        return summary
    return f"type={type(value).__name__}"


def _failure_summary(error: Exception) -> str:
    summary = f"exception={type(error).__name__} detail={safe_diagnostic_text(error)}"
    response = getattr(error, "response", None)
    if response is None:
        return summary
    status = getattr(response, "status_code", None)
    request = getattr(response, "request", None)
    url = getattr(request, "url", None) or getattr(response, "url", None)
    body = getattr(response, "text", None)
    if status is not None:
        summary += f" status={safe_diagnostic_text(status, maximum=40)}"
    if url is not None:
        summary += f" response_url={safe_diagnostic_url(str(url))}"
    if body:
        summary += f" body={safe_diagnostic_text(body, maximum=600)}"
    return summary


def diagnostic_request(
    *,
    provider: str,
    transport: str,
    operation: str,
    endpoint: str,
    details: str,
    call: Callable[[], T],
) -> T:
    logger = logging.getLogger(f"{_LOGGER_NAME}.market_data")
    safe_endpoint = (
        safe_diagnostic_url(endpoint) if transport.upper() == "HTTP" else endpoint
    )
    logger.info(
        "request provider=%s transport=%s operation=%s endpoint=%s details=%s",
        provider,
        transport,
        operation,
        safe_endpoint,
        safe_diagnostic_text(details),
    )
    try:
        result = call()
    except Exception as error:
        logger.error(
            "response provider=%s transport=%s operation=%s endpoint=%s outcome=failed %s",
            provider,
            transport,
            operation,
            safe_endpoint,
            _failure_summary(error),
        )
        raise
    logger.info(
        "response provider=%s transport=%s operation=%s endpoint=%s outcome=received %s",
        provider,
        transport,
        operation,
        safe_endpoint,
        _response_summary(result),
    )
    return result
