from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from compass.services.safe_display import safe_identifier
from compass.storage.backtest_result_codec import (
    decode_backtest_result,
    decode_finite_float,
    encode_backtest_result,
    encode_finite_float,
)
from compass.storage.canonical_json import (
    canonical_json,
    content_hash,
    decode_canonical_json,
)
from compass.storage.database import Database
from compass.storage.models import BacktestReportRecord
from compass.storage.run_snapshot_repository import RunSnapshotRepository
from compass.storage.write_order import next_write_order
from compass.ui.components.charts import CurvePoint
from compass.ui.pages.backtests import BacktestReport


Clock = Callable[[], datetime]
_SCHEMA_VERSION = 1


class BacktestReportRepository:
    def __init__(
        self,
        database: Database,
        snapshots: RunSnapshotRepository,
        *,
        clock: Clock,
    ) -> None:
        self._database = database
        self._snapshots = snapshots
        self._clock = clock

    def save(self, report: BacktestReport) -> None:
        if type(report) is not BacktestReport:
            raise TypeError("report must be an exact BacktestReport")
        report.verify_integrity()
        self._snapshots.save_snapshot(report.snapshot)
        payload_json = canonical_json(self._payload(report))
        created_at = self._clock()
        if (
            type(created_at) is not datetime
            or created_at.tzinfo is None
            or created_at.utcoffset() is None
        ):
            raise ValueError("report clock must return a timezone-aware datetime")
        try:
            with self._database.session_factory.begin() as session:
                existing = session.get(BacktestReportRecord, report.run_id)
                if existing is not None:
                    restored = self._decode(existing)
                    if restored != report:
                        raise ValueError("BACKTEST_REPORT_ID_CONFLICT")
                    return
                session.add(
                    BacktestReportRecord(
                        run_id=report.run_id,
                        snapshot_id=report.snapshot.snapshot_id,
                        schema_version=_SCHEMA_VERSION,
                        payload_json=payload_json,
                        content_hash=content_hash(payload_json),
                        created_at=created_at,
                        write_order=next_write_order(session),
                    )
                )
        except IntegrityError:
            raise ValueError("BACKTEST_REPORT_IMMUTABLE") from None

    def list(self) -> tuple[BacktestReport, ...]:
        with self._database.session_factory() as session:
            rows = session.scalars(
                select(BacktestReportRecord).order_by(BacktestReportRecord.run_id)
            ).all()
            return tuple(self._decode(row) for row in rows)

    def history(self) -> tuple[tuple[BacktestReport, datetime], ...]:
        with self._database.session_factory() as session:
            rows = session.scalars(
                select(BacktestReportRecord).order_by(
                    BacktestReportRecord.write_order.desc()
                )
            ).all()
            return tuple((self._decode(row), row.created_at) for row in rows)

    def get(self, run_id: str) -> BacktestReport | None:
        checked = safe_identifier(run_id, label="backtest run id")
        with self._database.session_factory() as session:
            row = session.get(BacktestReportRecord, checked)
            return None if row is None else self._decode(row)

    def latest(self) -> BacktestReport | None:
        with self._database.session_factory() as session:
            row = session.scalars(
                select(BacktestReportRecord)
                .order_by(
                    BacktestReportRecord.write_order.desc(),
                )
                .limit(1)
            ).first()
            return None if row is None else self._decode(row)

    def delete(self, run_id: str) -> bool:
        checked = safe_identifier(run_id, label="backtest run id")
        with self._database.session_factory.begin() as session:
            row = session.get(BacktestReportRecord, checked)
            if row is None:
                return False
            session.delete(row)
            return True

    def clear(self) -> int:
        with self._database.session_factory.begin() as session:
            rows = session.scalars(select(BacktestReportRecord)).all()
            for row in rows:
                session.delete(row)
            return len(rows)

    @staticmethod
    def _payload(report: BacktestReport) -> dict[str, object]:
        return {
            "benchmark_curve": [
                {"day": item.day, "value": encode_finite_float(item.value)}
                for item in report.benchmark_curve
            ],
            "configuration_id": report.configuration_id,
            "export_available": report.export_available,
            "result": encode_backtest_result(report.result),
            "run_id": report.run_id,
            "schema_version": _SCHEMA_VERSION,
            "snapshot_id": report.snapshot.snapshot_id,
            "strategy_instance_ids": list(report.strategy_instance_ids),
        }

    def _decode(self, row: BacktestReportRecord) -> BacktestReport:
        try:
            payload = decode_canonical_json(row.payload_json, row.content_hash)
            expected = {
                "benchmark_curve",
                "configuration_id",
                "export_available",
                "result",
                "run_id",
                "schema_version",
                "snapshot_id",
                "strategy_instance_ids",
            }
            if (
                type(row.schema_version) is not int
                or row.schema_version != _SCHEMA_VERSION
                or type(payload.get("schema_version")) is not int
                or payload["schema_version"] != _SCHEMA_VERSION
                or set(payload) != expected
                or type(payload["run_id"]) is not str
                or type(payload["configuration_id"]) is not str
                or type(payload["snapshot_id"]) is not str
                or type(payload["export_available"]) is not bool
                or type(payload["strategy_instance_ids"]) is not list
                or type(payload["benchmark_curve"]) is not list
                or row.run_id != payload["run_id"]
                or row.snapshot_id != payload["snapshot_id"]
            ):
                raise ValueError
            strategy_ids = tuple(
                item
                for item in cast(list[object], payload["strategy_instance_ids"])
                if type(item) is str
            )
            if len(strategy_ids) != len(cast(list[object], payload["strategy_instance_ids"])):
                raise ValueError
            benchmark = []
            for raw_point in cast(list[object], payload["benchmark_curve"]):
                if not isinstance(raw_point, Mapping):
                    raise ValueError
                benchmark.append(
                    CurvePoint(
                        cast(str, raw_point["day"]),
                        decode_finite_float(raw_point["value"]),
                    )
                )
            snapshot = self._snapshots.load_snapshot(row.snapshot_id)
            result = decode_backtest_result(payload["result"])
            report = BacktestReport.from_result(
                run_id=row.run_id,
                configuration_id=payload["configuration_id"],
                strategy_instance_ids=strategy_ids,
                result=result,
                snapshot=snapshot,
                benchmark_curve=tuple(benchmark),
                export_available=payload["export_available"],
            )
            report.verify_integrity()
            return report
        except (KeyError, TypeError, ValueError):
            raise ValueError("BACKTEST_REPORT_INTEGRITY") from None
