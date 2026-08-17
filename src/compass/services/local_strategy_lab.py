from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from decimal import Decimal
from math import isfinite

import pandas as pd  # type: ignore[import-untyped]

from compass.backtest.engine import (
    BacktestEngine,
    BacktestRequest,
    DecisionTarget,
    DecisionSource,
    ForecastTrace,
)
from compass.backtest.broker import InitialPosition
from compass.backtest.orders import round_money
from compass.backtest.snapshot import RunSnapshot, StrategySnapshot
from compass.data.base import default_instrument_type
from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import TargetIntent
from compass.domain.weights import WEIGHT_SCALE, units_to_weight, weight_to_units
from compass.portfolio.allocator import DeterministicAllocator
from compass.portfolio.models import PortfolioTarget
from compass.portfolio.trace import AllocationPolicy
from compass.services.dataset_provenance import (
    snapshot_data_quality,
    validate_dataset_provenance,
)
from compass.services.instrument_names import common_instrument_name
from compass.services.local_market_configuration import (
    local_instruments,
    local_risk_engine,
    local_rule_book,
)
from compass.services.local_read_gateways import _benchmark_curve
from compass.services.task_manager import TaskStatus
from compass.storage.backtest_report_repository import BacktestReportRepository
from compass.storage.dataset_bundle_repository import DatasetBundleRepository
from compass.strategies.base import StrategyContext
from compass.strategies.dual_ma import DualMaParameters
from compass.strategies.indicators import simple_moving_average
from compass.strategies.kronos_forecast import KronosForecastStrategy
from compass.strategies.rule_dsl import (
    RuleDslParameters,
    compile_rule,
    required_history as dsl_required_history,
)
from compass.ui.pages.backtests import BacktestReport
from compass.ui.pages.strategy_lab import (
    StrategyLabConfiguration,
    StrategyLabHistoryEntry,
    StrategyLabInstrument,
    StrategyLabKind,
    StrategyLabTemplate,
    StrategyLabRebalanceMode,
    StrategyLegConfiguration,
)
from compass.ui.pages.strategies import StrategyGateway


IdFactory = Callable[[str], str]


def _equal_weights(instruments: Sequence[InstrumentId]) -> Mapping[InstrumentId, Decimal]:
    ordered = tuple(sorted(instruments, key=str))
    base, remainder = divmod(WEIGHT_SCALE, len(ordered))
    return {
        instrument: units_to_weight(base + (1 if position < remainder else 0))
        for position, instrument in enumerate(ordered)
    }


def _materialize_initial_positions(
    configuration: StrategyLabConfiguration,
    bars: Mapping[InstrumentId, pd.DataFrame],
    first_session: date,
) -> tuple[Decimal, tuple[InitialPosition, ...]]:
    positions: list[InitialPosition] = []
    invested = Decimal("0")
    for requested in configuration.initial_positions:
        frame = bars[requested.instrument]
        matching = [timestamp for timestamp in frame.index if timestamp.date() == first_session]
        if not matching:
            raise LookupError("BACKTEST_INITIAL_POSITION_PRICE_MISSING")
        raw_price = frame.loc[matching[0], "open"]
        if isinstance(raw_price, bool):
            raise ValueError("BACKTEST_INITIAL_POSITION_PRICE_INVALID")
        price = Decimal(str(raw_price))
        if not price.is_finite() or price <= 0:
            raise ValueError("BACKTEST_INITIAL_POSITION_PRICE_INVALID")
        target_value = configuration.initial_cash * requested.target_weight
        quantity = int(target_value / price) // 100 * 100
        if quantity <= 0:
            raise ValueError("BACKTEST_INITIAL_POSITION_TOO_SMALL")
        market_value = round_money(price * quantity)
        invested += market_value
        positions.append(
            InitialPosition(
                instrument=requested.instrument,
                quantity=quantity,
                available_quantity=quantity,
                average_cost=price,
                mark_price=price,
            )
        )
    cash = round_money(configuration.initial_cash - invested)
    if cash < 0:
        raise ValueError("BACKTEST_INITIAL_POSITION_EXCEEDS_CAPITAL")
    return cash, tuple(positions)


class _CombinedStrategyDecisionSource:
    def __init__(
        self,
        configuration: StrategyLabConfiguration,
    ) -> None:
        self._strategies = tuple(configuration.strategies)
        self._rebalance_mode = configuration.rebalance_mode
        self._rebalance_drift = configuration.rebalance_drift
        self._minimum_trade_amount = configuration.minimum_trade_amount
        self._last_requested_weights: Mapping[str, Decimal] | None = None
        target_instruments = tuple(
            sorted(
                {
                    instrument
                    for strategy in self._strategies
                    for instrument in strategy.instruments
                },
                key=str,
            )
        )
        self._weights = {
            strategy.strategy_id: _equal_weights(strategy.instruments)
            for strategy in self._strategies
        }
        self._parameters = {
            strategy.strategy_id: DualMaParameters(
                short_window=strategy.short_window,
                long_window=strategy.long_window,
                confirmation_days=strategy.confirmation_days,
                target_weight=Decimal("1"),
            )
            for strategy in self._strategies
            if strategy.strategy is StrategyLabKind.DUAL_MA
        }
        self._rule_parameters = {
            strategy.strategy_id: RuleDslParameters(
                buy_expression=strategy.buy_expression,
                sell_expression=strategy.sell_expression,
                variables=strategy.variables,
                target_weight=Decimal("1"),
            )
            for strategy in self._strategies
            if strategy.strategy is StrategyLabKind.RULE_DSL
        }
        self._rule_programs = {
            strategy_id: (
                compile_rule(parameters.buy_expression, tuple(parameters.variable_values)),
                compile_rule(parameters.sell_expression, tuple(parameters.variable_values)),
            )
            for strategy_id, parameters in self._rule_parameters.items()
        }
        self._kronos_strategies = {
            strategy.strategy_id: KronosForecastStrategy(
                strategy.kronos_parameters,
                strategy.strategy_id,
            )
            for strategy in self._strategies
            if strategy.strategy is StrategyLabKind.KRONOS_FORECAST
        }
        self._allocator = DeterministicAllocator()
        self._policy = AllocationPolicy(
            strategy_budgets={
                strategy.strategy_id: strategy.budget for strategy in self._strategies
            },
            asset_class_budgets={AssetType.ETF: Decimal("1")},
            asset_types={instrument: AssetType.ETF for instrument in target_instruments},
            cash_reserve=Decimal("0"),
        )

    def _active(self, strategy: StrategyLegConfiguration, context: StrategyContext) -> bool:
        if strategy.strategy is StrategyLabKind.BUY_AND_HOLD:
            return True
        assert strategy.signal_instrument is not None
        if strategy.strategy is StrategyLabKind.RULE_DSL:
            dsl_parameters = self._rule_parameters[strategy.strategy_id]
            history = context.history(strategy.signal_instrument)
            needed = dsl_required_history(
                (dsl_parameters.buy_expression, dsl_parameters.sell_expression),
                dsl_parameters.variable_values,
            )
            held = any(
                (holding := context.holding(instrument)) is not None and holding.quantity > 0
                for instrument in strategy.instruments
            )
            if len(history) < needed:
                return held
            buy, sell = self._rule_programs[strategy.strategy_id]
            if sell.evaluate(history, dsl_parameters.variable_values):
                return False
            return buy.evaluate(history, dsl_parameters.variable_values) or held
        ma_parameters = self._parameters[strategy.strategy_id]
        close = context.history(strategy.signal_instrument)["close"]
        required_history = ma_parameters.long_window + ma_parameters.confirmation_days - 1
        if len(close) < required_history:
            return False
        short = simple_moving_average(close, ma_parameters.short_window)
        long = simple_moving_average(close, ma_parameters.long_window)
        spread = short - long
        recent = spread.iloc[-ma_parameters.confirmation_days :]
        return (
            not recent.isna().any() and isfinite(float(long.iloc[-1])) and bool((recent > 0).all())
        )

    def _is_rebalance_session(self, context: StrategyContext) -> bool:
        if self._last_requested_weights is None:
            return True
        if self._rebalance_mode in {
            StrategyLabRebalanceMode.DAILY,
            StrategyLabRebalanceMode.SIGNAL_CHANGE,
        }:
            return True
        anchor = self._strategies[0].instruments[0]
        visible = context.history(anchor).index
        if len(visible) < 2:
            return False
        current, previous = visible[-1], visible[-2]
        if self._rebalance_mode is StrategyLabRebalanceMode.WEEKLY:
            return (current.isocalendar().year, current.isocalendar().week) != (
                previous.isocalendar().year,
                previous.isocalendar().week,
            )
        return (current.year, current.month) != (previous.year, previous.month)

    @staticmethod
    def _decision_target(target: PortfolioTarget) -> DecisionTarget:
        weights: dict[InstrumentId, Decimal] = {}
        sleeves: dict[InstrumentId, dict[str, Decimal]] = {}
        for symbol_text, weight in target.weights.items():
            symbol = InstrumentId.parse(symbol_text)
            weights[symbol] = weight
            sleeves[symbol] = {
                strategy_id: sleeve.final_weights[symbol_text]
                for strategy_id, sleeve in target.sleeves.items()
                if symbol_text in sleeve.final_weights and sleeve.final_weights[symbol_text] > 0
            }
        return DecisionTarget(weights, sleeves)

    def _suppress_small_trades(
        self, context: StrategyContext, target: DecisionTarget
    ) -> DecisionTarget:
        adjusted = dict(target.weights)
        for instrument in set(target.weights) | set(context.holdings):
            holding = context.holding(instrument)
            current_value = (
                Decimal("0") if holding is None else holding.mark_price * holding.quantity
            )
            current_units = (
                0
                if context.account_equity <= 0
                else min(
                    WEIGHT_SCALE,
                    int(current_value * WEIGHT_SCALE / context.account_equity),
                )
            )
            current_weight = units_to_weight(current_units)
            requested = target.weights.get(instrument, Decimal("0"))
            trade_amount = abs(requested - current_weight) * context.account_equity
            if (
                abs(requested - current_weight) < self._rebalance_drift
                or trade_amount < self._minimum_trade_amount
            ):
                adjusted[instrument] = current_weight
        if (
            sum((weight_to_units(item, label="adjusted target") for item in adjusted.values()))
            > WEIGHT_SCALE
        ):
            return target
        return DecisionTarget(
            adjusted,
            target.sleeve_weights,
            forecast_traces=target.forecast_traces,
        )

    def targets(self, context: StrategyContext) -> DecisionTarget:
        intents: list[TargetIntent] = []
        forecast_traces: list[ForecastTrace] = []
        for strategy in self._strategies:
            if strategy.strategy is StrategyLabKind.KRONOS_FORECAST:
                assert strategy.kronos_parameters is not None
                kronos_context = StrategyContext(
                    as_of=context.as_of,
                    bars={
                        instrument: context.history(instrument)
                        for instrument in strategy.instruments
                    },
                    instruments=strategy.instruments,
                    account_equity=context.account_equity,
                    cash=context.cash,
                    holdings={
                        instrument: holding
                        for instrument, holding in context.holdings.items()
                        if instrument in strategy.instruments
                    },
                    asset_types={
                        instrument: context.asset_types[instrument]
                        for instrument in strategy.instruments
                    },
                )
                decision = self._kronos_strategies[strategy.strategy_id].generate_targets(
                    kronos_context
                )
                intents.extend(decision.intents)
                forecast_traces.extend(
                    ForecastTrace(
                        decision_date=context.as_of,
                        strategy_id=strategy.strategy_id,
                        instrument=item.instrument,
                        action=item.action,
                        expected_return=item.expected_return,
                        path_positive_ratio=item.path_positive_ratio,
                        rank=item.rank,
                        close=item.close,
                        trend_value=item.trend_value,
                        trend_passed=item.trend_passed,
                        target_weight=item.target_weight,
                        reason_code=item.reason_code,
                        horizon=strategy.kronos_parameters.horizon,
                    )
                    for item in self._kronos_strategies[
                        strategy.strategy_id
                    ].latest_diagnostics
                )
                continue
            active = self._active(strategy, context)
            for instrument, weight in self._weights[strategy.strategy_id].items():
                intents.append(
                    TargetIntent(
                        strategy_id=strategy.strategy_id,
                        instrument=instrument,
                        target_weight=weight if active else Decimal("0"),
                        score=1.0 if active else 0.0,
                        confidence=1.0,
                        reason_code=(
                            "BUY_AND_HOLD_TARGET"
                            if strategy.strategy is StrategyLabKind.BUY_AND_HOLD
                            else "RULE_DSL_RISK_ON"
                            if strategy.strategy is StrategyLabKind.RULE_DSL and active
                            else "RULE_DSL_RISK_OFF"
                            if strategy.strategy is StrategyLabKind.RULE_DSL
                            else "DUAL_MA_RISK_ON"
                            if active
                            else "DUAL_MA_RISK_OFF"
                        ),
                        valid_until=context.as_of,
                    )
                )
        allocated = self._allocator.allocate(intents, self._policy)
        requested = dict(allocated.weights)
        traces = tuple(
            sorted(
                forecast_traces,
                key=lambda item: (
                    item.decision_date,
                    item.strategy_id,
                    str(item.instrument),
                ),
            )
        )
        if (
            self._rebalance_mode is StrategyLabRebalanceMode.SIGNAL_CHANGE
            and self._last_requested_weights == requested
        ):
            return DecisionTarget(
                {}, {}, preserve_unspecified=True, forecast_traces=traces
            )
        if not self._is_rebalance_session(context):
            return DecisionTarget(
                {}, {}, preserve_unspecified=True, forecast_traces=traces
            )
        self._last_requested_weights = requested
        allocated_target = self._decision_target(allocated)
        explained_target = DecisionTarget(
            allocated_target.weights,
            allocated_target.sleeve_weights,
            forecast_traces=traces,
        )
        return self._suppress_small_trades(context, explained_target)


class LocalStrategyLabGateway:
    def __init__(
        self,
        *,
        bundles: DatasetBundleRepository,
        reports: BacktestReportRepository,
        strategies: StrategyGateway,
        app_git_commit: str,
        id_factory: IdFactory,
    ) -> None:
        self._bundles = bundles
        self._reports = reports
        self._strategies = strategies
        self._app_git_commit = app_git_commit
        self._id_factory = id_factory
        self._instrument_cache: tuple[str, tuple[StrategyLabInstrument, ...]] | None = None
        self._comparison_cache: dict[tuple[str, InstrumentId, str], BacktestReport] = {}
        self._history_cache: tuple[StrategyLabHistoryEntry, ...] | None = None

    def templates(self) -> tuple[StrategyLabTemplate, ...]:
        supported = {item.value: item for item in StrategyLabKind}
        pool_reader = getattr(self._strategies, "pool_instruments", None)
        if not callable(pool_reader):
            return ()
        templates = []
        for instance in self._strategies.list():
            kind = supported.get(instance.strategy_type)
            if not instance.enabled or kind is None:
                continue
            templates.append(
                StrategyLabTemplate(
                    instance_id=instance.instance_id,
                    name=instance.name,
                    strategy=kind,
                    strategy_version=instance.strategy_version,
                    instruments=tuple(pool_reader(instance.instance_id)),
                    parameters=instance.parameters,
                )
            )
        return tuple(sorted(templates, key=lambda item: item.instance_id))

    def instruments(self) -> tuple[StrategyLabInstrument, ...]:
        bundle = self._bundles.latest()
        if bundle is None:
            return ()
        cached = self._instrument_cache
        if cached is not None and cached[0] == bundle.bundle_id:
            return cached[1]
        references = self._bundles.references_by_instrument(bundle)
        available: list[StrategyLabInstrument] = []
        for instrument in bundle.instruments:
            reference = references[instrument]
            manifest = self._bundles.load_manifest(reference.manifest_id)
            frame = self._bundles.read_manifest(reference.manifest_id)
            if frame.empty:
                continue
            name = manifest.instrument_name or common_instrument_name(instrument) or instrument.code
            available.append(
                StrategyLabInstrument(
                    instrument=instrument,
                    name=name,
                    asset_type=default_instrument_type(instrument),
                    first_day=frame.index[0].date(),
                    last_day=frame.index[-1].date(),
                    rows=len(frame),
                )
            )
        result = tuple(sorted(available, key=lambda item: str(item.instrument)))
        self._instrument_cache = (bundle.bundle_id, result)
        return result

    def latest_report(self) -> BacktestReport | None:
        return self._reports.latest()

    def report(self, run_id: str) -> BacktestReport | None:
        return self._reports.get(run_id)

    def compare_report(self, run_id: str, benchmark: InstrumentId) -> BacktestReport:
        if type(benchmark) is not InstrumentId:
            raise TypeError("benchmark must be an exact InstrumentId")
        bundle = self._bundles.latest()
        if bundle is None:
            raise LookupError("BACKTEST_DATA_BUNDLE_MISSING")
        cache_key = (run_id, benchmark, bundle.bundle_id)
        cached = self._comparison_cache.get(cache_key)
        if cached is not None:
            return cached
        report = self._reports.get(run_id)
        if report is None:
            raise LookupError("BACKTEST_REPORT_MISSING")
        references = self._bundles.references_by_instrument(bundle)
        reference = references.get(benchmark)
        if reference is None:
            raise LookupError("BACKTEST_BENCHMARK_DATA_MISSING")
        frame = self._bundles.read_manifest(reference.manifest_id)
        sessions = tuple(item.trading_day for item in report.result.ledger)
        if (
            frame.empty
            or sessions[0] < frame.index[0].date()
            or sessions[-1] > frame.index[-1].date()
        ):
            raise LookupError("BACKTEST_BENCHMARK_RANGE_NOT_COVERED")
        compared = BacktestReport.from_result(
            run_id=report.run_id,
            configuration_id=report.configuration_id,
            strategy_instance_ids=report.strategy_instance_ids,
            result=report.result,
            snapshot=report.snapshot,
            benchmark_curve=_benchmark_curve(frame, sessions),
            export_available=report.export_available,
        )
        self._comparison_cache[cache_key] = compared
        return compared

    def history(self) -> tuple[StrategyLabHistoryEntry, ...]:
        if self._history_cache is not None:
            return self._history_cache

        def target_count(report: BacktestReport) -> int:
            raw = report.snapshot.instrument_pool.get("trade_instruments", ())
            return len(raw) if isinstance(raw, (tuple, list)) else 0

        result = tuple(
            StrategyLabHistoryEntry(
                run_id=report.run_id,
                status=TaskStatus.SUCCEEDED,
                submitted_at=created_at,
                completed_at=created_at,
                strategy_count=len(report.strategies),
                target_count=target_count(report),
                has_report=True,
            )
            for report, created_at in self._reports.history()
        )
        self._history_cache = result
        return result

    def delete_report(self, run_id: str) -> bool:
        deleted = self._reports.delete(run_id)
        if deleted:
            self._history_cache = None
            self._comparison_cache = {
                key: value for key, value in self._comparison_cache.items() if key[0] != run_id
            }
        return deleted

    def clear_reports(self) -> int:
        cleared = self._reports.clear()
        self._history_cache = None
        self._comparison_cache.clear()
        return cleared

    def new_run_id(self) -> str:
        return self._id_factory("backtest")

    def run(self, run_id: str, configuration: StrategyLabConfiguration) -> None:
        report = self.evaluate(run_id, configuration)
        self._reports.save(report)
        self._history_cache = None

    def evaluate(
        self,
        run_id: str,
        configuration: StrategyLabConfiguration,
    ) -> BacktestReport:
        bundle = self._bundles.latest()
        if bundle is None:
            raise LookupError("BACKTEST_DATA_BUNDLE_MISSING")
        references = self._bundles.references_by_instrument(bundle)
        trade_instruments = {
            instrument
            for strategy in configuration.strategies
            for instrument in strategy.instruments
        }
        signal_instruments = {
            strategy.signal_instrument
            for strategy in configuration.strategies
            if strategy.signal_instrument is not None
        }
        initial_position_instruments = {item.instrument for item in configuration.initial_positions}
        strategy_instruments = tuple(
            sorted(
                trade_instruments | signal_instruments | initial_position_instruments,
                key=str,
            )
        )
        selected = {*strategy_instruments, configuration.benchmark}
        if not selected.issubset(references):
            raise LookupError("BACKTEST_INSTRUMENT_DATA_MISSING")
        if any(
            default_instrument_type(item) is not AssetType.ETF
            for item in trade_instruments | initial_position_instruments
        ):
            raise ValueError("BACKTEST_ONLY_ETF_TRADABLE")

        selected_references = tuple(
            sorted((references[item] for item in selected), key=lambda item: item.manifest_id)
        )
        selected_manifests = tuple(
            self._bundles.load_manifest(reference.manifest_id) for reference in selected_references
        )
        provenance = validate_dataset_provenance(
            selected_manifests,
            required_through=configuration.end,
            failure_prefix="BACKTEST",
            allow_quality_gaps=bundle.data_quality["mode"] == "degraded",
        )
        sessions = tuple(
            day for day in provenance.sessions if configuration.start <= day <= configuration.end
        )
        if len(sessions) < 2:
            raise LookupError("BACKTEST_SESSIONS_MISSING")

        benchmark_reference = references[configuration.benchmark]
        benchmark_bars = self._bundles.read_manifest(benchmark_reference.manifest_id)
        bars = {
            instrument: self._bundles.read_manifest(references[instrument].manifest_id)
            for instrument in strategy_instruments
        }
        selected_frames = (*bars.values(), benchmark_bars)
        if any(
            sessions[0] < frame.index[0].date() or sessions[-1] > frame.index[-1].date()
            for frame in selected_frames
        ):
            raise LookupError("BACKTEST_RANGE_NOT_COVERED")

        decision_source: DecisionSource = _CombinedStrategyDecisionSource(configuration)
        instruments = local_instruments(strategy_instruments)
        rule_book = local_rule_book(
            instruments,
            fee_confirmed=True,
            commission_rate=configuration.commission_rate,
            minimum_commission=configuration.minimum_commission,
            slippage_bps=configuration.slippage_bps,
        )
        corporate_actions = tuple(
            action
            for action in provenance.corporate_actions
            if action.instrument in trade_instruments | initial_position_instruments
            and action.ex_date in set(sessions)
        )
        initial_cash, initial_positions = _materialize_initial_positions(
            configuration,
            bars,
            sessions[0],
        )
        result = BacktestEngine().run(
            BacktestRequest(
                run_id=run_id,
                sessions=sessions,
                instruments=instruments,
                bars=bars,
                initial_cash=initial_cash,
                initial_positions=initial_positions,
                corporate_actions=corporate_actions,
                decision_source=decision_source,
                risk_engine=local_risk_engine(active=False),
                rule_book=rule_book,
                execution_timing=configuration.execution_timing,
            )
        )
        snapshot = RunSnapshot(
            run_id=run_id,
            schema_version=1,
            market_manifests=selected_references,
            data_quality=snapshot_data_quality(bundle.data_quality, provenance),
            strategies=tuple(
                StrategySnapshot(
                    sleeve_id=strategy.strategy_id,
                    strategy_type=strategy.strategy.value,
                    strategy_version=(
                        "1.1.0"
                        if strategy.strategy is StrategyLabKind.DUAL_MA
                        else "1.0.0"
                        if strategy.strategy is StrategyLabKind.KRONOS_FORECAST
                        else "1.0.0"
                    ),
                    parameters={
                        "budget": strategy.budget,
                        "signal_instrument": (
                            None
                            if strategy.signal_instrument is None
                            else str(strategy.signal_instrument)
                        ),
                        "trade_instruments": tuple(str(item) for item in strategy.instruments),
                        "short_window": strategy.short_window,
                        "long_window": strategy.long_window,
                        "confirmation_days": strategy.confirmation_days,
                        "buy_expression": strategy.buy_expression,
                        "sell_expression": strategy.sell_expression,
                        "variables": tuple(
                            item.model_dump(mode="json") for item in strategy.variables
                        ),
                        "kronos_parameters": (
                            None
                            if strategy.kronos_parameters is None
                            else strategy.kronos_parameters.model_dump(mode="json")
                        ),
                        "template_instance_id": strategy.template_instance_id,
                        "template_name": strategy.template_name,
                    },
                )
                for strategy in configuration.strategies
            ),
            instrument_pool={
                "instruments": tuple(str(item) for item in strategy_instruments),
                "signal_instruments": tuple(
                    str(item) for item in sorted(signal_instruments, key=str)
                ),
                "trade_instruments": tuple(
                    str(item) for item in sorted(trade_instruments, key=str)
                ),
                "initial_positions": tuple(
                    {
                        "instrument": str(item.instrument),
                        "target_weight": item.target_weight,
                    }
                    for item in configuration.initial_positions
                ),
                "initial_cash_weight": configuration.initial_cash_weight,
                "benchmark": str(configuration.benchmark),
                "start": configuration.start,
                "end": configuration.end,
            },
            survivorship_bias={"mode": "static_instrument", "warning": False},
            allocator_configuration={
                "kind": "deterministic_multi_strategy",
                "rebalance_mode": configuration.rebalance_mode.value,
                "rebalance_drift": configuration.rebalance_drift,
                "minimum_trade_amount": configuration.minimum_trade_amount,
                "strategy_budgets": {
                    strategy.strategy_id: strategy.budget for strategy in configuration.strategies
                },
                "asset_class_budgets": {AssetType.ETF.value: Decimal("1")},
                "cash_reserve": Decimal("0"),
                "initial_cash": configuration.initial_cash,
                "materialized_initial_cash": initial_cash,
            },
            risk_configuration={"active": False},
            market_rule_configuration={
                "execution": configuration.execution_timing.value,
                "profile_ids": tuple(profile.profile_id for profile in rule_book.profiles),
            },
            fee_profile_configuration={
                "confirmed": True,
                "commission_rate": configuration.commission_rate,
                "minimum_commission": configuration.minimum_commission,
                "slippage_bps": configuration.slippage_bps,
            },
            app_git_commit=self._app_git_commit,
            random_seed=0,
        )
        report = BacktestReport.from_result(
            run_id=run_id,
            configuration_id=f"config-{run_id.removeprefix('backtest-')}",
            strategy_instance_ids=tuple(
                strategy.strategy_id for strategy in configuration.strategies
            ),
            result=result,
            snapshot=snapshot,
            benchmark_curve=_benchmark_curve(benchmark_bars, sessions),
            export_available=False,
        )
        return report
