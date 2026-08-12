from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal, ROUND_FLOOR, getcontext, setcontext
from random import Random
from types import MappingProxyType

import pytest
from hypothesis import given, settings, strategies as st

from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import TargetIntent
from compass.portfolio import (
    AllocationPolicy,
    AllocationStage,
    DeterministicAllocator,
    PortfolioTarget,
    SleeveTarget,
)


ETF = InstrumentId.parse("SSE.510300")
ETF_2 = InstrumentId.parse("SZSE.159915")
STOCK = InstrumentId.parse("SSE.600000")


def _intent(
    strategy_id: str,
    instrument: InstrumentId,
    weight: Decimal | float,
    *,
    score: float = 1.0,
) -> TargetIntent:
    return TargetIntent(
        strategy_id=strategy_id,
        instrument=instrument,
        target_weight=weight,  # type: ignore[arg-type]
        score=score,
        confidence=1.0,
        reason_code="TEST",
        valid_until=date(2026, 7, 22),
    )


def _policy(
    *,
    strategy_budgets: dict[str, Decimal] | None = None,
    asset_class_budgets: dict[AssetType, Decimal] | None = None,
    asset_types: dict[InstrumentId, AssetType] | None = None,
    cash_reserve: Decimal = Decimal("0.10"),
) -> AllocationPolicy:
    return AllocationPolicy(
        strategy_budgets=strategy_budgets
        or {"rotation": Decimal("0.50"), "trend": Decimal("0.40")},
        asset_class_budgets=asset_class_budgets
        or {AssetType.ETF: Decimal("0.80"), AssetType.STOCK: Decimal("0.20")},
        asset_types=asset_types
        or {ETF: AssetType.ETF, ETF_2: AssetType.ETF, STOCK: AssetType.STOCK},
        cash_reserve=cash_reserve,
    )


def _replay_target() -> PortfolioTarget:
    return DeterministicAllocator().allocate(
        [
            _intent("rotation", ETF, Decimal("0.8")),
            _intent("rotation", STOCK, Decimal("0.4")),
        ],
        _policy(
            strategy_budgets={"rotation": Decimal("0.6")},
            asset_class_budgets={AssetType.ETF: Decimal("0.3"), AssetType.STOCK: Decimal("0.1")},
            asset_types={ETF: AssetType.ETF, STOCK: AssetType.STOCK},
            cash_reserve=Decimal("0.1"),
        ),
    )


def _empty_sleeve(strategy_id: str, budget: Decimal) -> SleeveTarget:
    return SleeveTarget(
        strategy_id=strategy_id,
        strategy_budget=budget,
        requested_weights={},
        normalized_weights={},
        budgeted_weights={},
        asset_limited_weights={},
        final_weights={},
    )


def test_portfolio_rejects_zero_weight_sleeve_with_wrong_policy_budget() -> None:
    policy = _policy(
        strategy_budgets={"rotation": Decimal("0.6")},
        asset_class_budgets={AssetType.ETF: Decimal("1")},
        asset_types={ETF: AssetType.ETF},
        cash_reserve=Decimal("0"),
    )
    result = DeterministicAllocator().allocate(
        [_intent("rotation", ETF, Decimal("0"))], policy
    )
    tampered = replace(
        result.sleeves["rotation"], strategy_budget=Decimal("0.5")
    )

    with pytest.raises(ValueError, match="strategy budget.*policy"):
        replace(result, sleeves={"rotation": tampered})


def test_portfolio_rejects_empty_ghost_sleeve_outside_policy() -> None:
    result = DeterministicAllocator().allocate([], _policy())

    with pytest.raises(ValueError, match="not configured in policy"):
        replace(
            result,
            sleeves={"ghost": _empty_sleeve("ghost", Decimal("0.5"))},
        )


def test_portfolio_rejects_empty_sleeve_for_known_strategy() -> None:
    policy = _policy(strategy_budgets={"rotation": Decimal("0.6")})
    result = DeterministicAllocator().allocate([], policy)

    with pytest.raises(ValueError, match="requested weights must not be empty"):
        replace(
            result,
            sleeves={"rotation": _empty_sleeve("rotation", Decimal("0.6"))},
        )


def test_allocator_preserves_sleeves_but_nets_real_target() -> None:
    policy = _policy(
        strategy_budgets={"rotation": Decimal("0.30"), "trend": Decimal("0.30")},
        asset_class_budgets={AssetType.ETF: Decimal("1")},
        asset_types={ETF: AssetType.ETF},
    )

    result = DeterministicAllocator().allocate(
        [
            _intent("rotation", ETF, Decimal("0.80")),
            _intent("trend", ETF, Decimal("0.20")),
        ],
        policy,
    )

    assert result.sleeves["rotation"].weights["SSE.510300"] == Decimal("0.24")
    assert result.sleeves["trend"].weights["SSE.510300"] == Decimal("0.06")
    assert result.weights["SSE.510300"] == Decimal("0.30")
    assert sum(result.weights.values()) <= Decimal("0.90")
    assert result.cash_weight >= Decimal("0.10")


def test_allocator_applies_normalization_then_strategy_and_asset_budgets() -> None:
    policy = _policy(
        strategy_budgets={"rotation": Decimal("0.60")},
        asset_class_budgets={AssetType.ETF: Decimal("0.30"), AssetType.STOCK: Decimal("0.20")},
        asset_types={ETF: AssetType.ETF, STOCK: AssetType.STOCK},
        cash_reserve=Decimal("0"),
    )

    result = DeterministicAllocator().allocate(
        [
            _intent("rotation", ETF, Decimal("0.80")),
            _intent("rotation", STOCK, Decimal("0.40")),
        ],
        policy,
    )
    sleeve = result.sleeves["rotation"]

    assert sleeve.requested_weights == {
        "SSE.510300": Decimal("0.80"),
        "SSE.600000": Decimal("0.40"),
    }
    assert sleeve.normalized_weights == {
        "SSE.510300": Decimal("0.666666666667"),
        "SSE.600000": Decimal("0.333333333333"),
    }
    assert sleeve.budgeted_weights == {
        "SSE.510300": Decimal("0.400000000000"),
        "SSE.600000": Decimal("0.200000000000"),
    }
    assert sleeve.asset_limited_weights == {
        "SSE.510300": Decimal("0.30"),
        "SSE.600000": Decimal("0.20"),
    }
    assert sleeve.weights == {
        "SSE.510300": Decimal("0.30"),
        "SSE.600000": Decimal("0.20"),
    }
    assert result.asset_weights == {
        AssetType.ETF: Decimal("0.30"),
        AssetType.STOCK: Decimal("0.20"),
    }
    assert result.symbol_asset_types == {
        "SSE.510300": AssetType.ETF,
        "SSE.600000": AssetType.STOCK,
    }
    assert any(
        adjustment.stage is AllocationStage.NORMALIZATION
        and adjustment.before_units == 1_200_000_000_000
        and adjustment.after_units == 1_000_000_000_000
        for adjustment in result.adjustments
    )


def test_cash_reserve_scales_all_sleeve_contributions_proportionally() -> None:
    policy = _policy(
        strategy_budgets={"rotation": Decimal("0.50"), "trend": Decimal("0.50")},
        asset_class_budgets={AssetType.ETF: Decimal("0.50"), AssetType.STOCK: Decimal("0.50")},
        asset_types={ETF: AssetType.ETF, STOCK: AssetType.STOCK},
        cash_reserve=Decimal("0.20"),
    )
    result = DeterministicAllocator().allocate(
        [
            _intent("rotation", ETF, Decimal("1")),
            _intent("trend", STOCK, Decimal("1")),
        ],
        policy,
    )

    assert result.weights == {
        "SSE.510300": Decimal("0.40"),
        "SSE.600000": Decimal("0.40"),
    }
    assert result.cash_weight == Decimal("0.20")
    cash_adjustment = next(
        adjustment
        for adjustment in result.adjustments
        if adjustment.stage is AllocationStage.CASH_RESERVE
    )
    assert cash_adjustment.before_units == 1_000_000_000_000
    assert cash_adjustment.after_units == 800_000_000_000


def test_zero_weight_exit_key_is_retained_for_explanation() -> None:
    result = DeterministicAllocator().allocate(
        [_intent("rotation", ETF, Decimal("0"))],
        _policy(
            strategy_budgets={"rotation": Decimal("0.50")},
            asset_class_budgets={AssetType.ETF: Decimal("0.80")},
            asset_types={ETF: AssetType.ETF},
        ),
    )

    sleeve = result.sleeves["rotation"]
    assert sleeve.requested_weights == {"SSE.510300": Decimal("0")}
    assert sleeve.budgeted_weights == {"SSE.510300": Decimal("0")}
    assert sleeve.weights == {"SSE.510300": Decimal("0")}
    assert result.weights == {"SSE.510300": Decimal("0")}
    assert result.asset_weights == {AssetType.ETF: Decimal("0")}
    assert result.cash_weight == Decimal("1")


def test_empty_intents_produce_all_cash_and_no_sleeves() -> None:
    result = DeterministicAllocator().allocate([], _policy())

    assert result.sleeves == {}
    assert result.weights == {}
    assert result.asset_weights == {}
    assert result.cash_weight == Decimal("1")


def test_zero_strategy_budget_retains_an_explained_zero_target() -> None:
    result = DeterministicAllocator().allocate(
        [_intent("rotation", ETF, Decimal("1"))],
        _policy(
            strategy_budgets={"rotation": Decimal("0")},
            asset_class_budgets={AssetType.ETF: Decimal("1")},
            asset_types={ETF: AssetType.ETF},
            cash_reserve=Decimal("0"),
        ),
    )

    assert result.sleeves["rotation"].requested_weights["SSE.510300"] == Decimal("1")
    assert result.sleeves["rotation"].weights["SSE.510300"] == Decimal("0")
    assert result.weights["SSE.510300"] == Decimal("0")
    assert result.cash_weight == Decimal("1")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("strategy", Decimal("-0.01"), "strategy budget"),
        ("strategy", Decimal("NaN"), "strategy budget"),
        ("strategy", Decimal("Infinity"), "strategy budget"),
        ("strategy", True, "strategy budget"),
        ("strategy", 0.5, "strategy budget"),
        ("strategy", "0.5", "strategy budget"),
        ("asset", Decimal("1.01"), "asset-class budget"),
        ("asset", False, "asset-class budget"),
        ("cash", Decimal("NaN"), "cash reserve"),
        ("cash", 0.1, "cash reserve"),
    ],
)
def test_policy_rejects_non_exact_or_out_of_range_decimal_values(
    field: str, value: object, error: str
) -> None:
    kwargs: dict[str, object] = {
        "strategy_budgets": {"rotation": Decimal("0.5")},
        "asset_class_budgets": {AssetType.ETF: Decimal("0.8")},
        "asset_types": {ETF: AssetType.ETF},
        "cash_reserve": Decimal("0.1"),
    }
    if field == "strategy":
        kwargs["strategy_budgets"] = {"rotation": value}
    elif field == "asset":
        kwargs["asset_class_budgets"] = {AssetType.ETF: value}
    else:
        kwargs["cash_reserve"] = value

    with pytest.raises((TypeError, ValueError), match=error):
        AllocationPolicy(**kwargs)  # type: ignore[arg-type]


def test_policy_rejects_budget_totals_above_one() -> None:
    with pytest.raises(ValueError, match="strategy budgets.*one"):
        _policy(
            strategy_budgets={"rotation": Decimal("0.6"), "trend": Decimal("0.5")}
        )
    with pytest.raises(ValueError, match="asset-class budgets.*one"):
        _policy(
            asset_class_budgets={AssetType.ETF: Decimal("0.9"), AssetType.STOCK: Decimal("0.2")}
        )


def test_policy_requires_explicit_stable_keys_and_asset_type_values() -> None:
    with pytest.raises(ValueError, match="strategy id"):
        _policy(strategy_budgets={" rotation ": Decimal("0.5")})
    with pytest.raises(TypeError, match="asset-class budget keys"):
        AllocationPolicy(
            strategy_budgets={"rotation": Decimal("0.5")},
            asset_class_budgets={"etf": Decimal("0.8")},  # type: ignore[dict-item]
            asset_types={ETF: AssetType.ETF},
            cash_reserve=Decimal("0.1"),
        )


def test_policy_validates_mixed_key_types_before_canonical_sorting() -> None:
    with pytest.raises(TypeError, match="strategy id must be an exact string"):
        AllocationPolicy(
            strategy_budgets={1: Decimal("0.2"), "rotation": Decimal("0.3")},  # type: ignore[dict-item]
            asset_class_budgets={AssetType.ETF: Decimal("0.8")},
            asset_types={ETF: AssetType.ETF},
            cash_reserve=Decimal("0.1"),
        )
    with pytest.raises(TypeError, match="asset type values"):
        AllocationPolicy(
            strategy_budgets={"rotation": Decimal("0.5")},
            asset_class_budgets={AssetType.ETF: Decimal("0.8")},
            asset_types={ETF: "etf"},  # type: ignore[dict-item]
            cash_reserve=Decimal("0.1"),
        )


def test_policy_is_immutable_and_stably_sorted() -> None:
    policy = AllocationPolicy(
        strategy_budgets={"zeta": Decimal("0.4"), "alpha": Decimal("0.3")},
        asset_class_budgets={AssetType.STOCK: Decimal("0.2"), AssetType.ETF: Decimal("0.7")},
        asset_types={STOCK: AssetType.STOCK, ETF: AssetType.ETF},
        cash_reserve=Decimal("0.1"),
    )

    assert isinstance(policy.strategy_budgets, MappingProxyType)
    assert list(policy.strategy_budgets) == ["alpha", "zeta"]
    assert list(policy.asset_class_budgets) == [AssetType.ETF, AssetType.STOCK]
    assert [str(key) for key in policy.asset_types] == ["SSE.510300", "SSE.600000"]
    with pytest.raises(TypeError):
        policy.strategy_budgets["alpha"] = Decimal("0")  # type: ignore[index]


@pytest.mark.parametrize(
    ("intents", "policy", "error"),
    [
        (
            [_intent("unknown", ETF, Decimal("1"))],
            _policy(asset_types={ETF: AssetType.ETF}),
            "unknown strategy",
        ),
        (
            [_intent("rotation", ETF_2, Decimal("1"))],
            _policy(asset_types={ETF: AssetType.ETF}),
            "unknown instrument asset type",
        ),
        (
            [_intent("rotation", ETF, Decimal("1")), _intent("rotation", ETF, Decimal("0"))],
            _policy(asset_types={ETF: AssetType.ETF}),
            "duplicate strategy/instrument",
        ),
        (
            [_intent(" rotation ", ETF, Decimal("1"))],
            _policy(asset_types={ETF: AssetType.ETF}),
            "stable strategy id",
        ),
        (
            [_intent("rotation", ETF, 0.5)],
            _policy(asset_types={ETF: AssetType.ETF}),
            "target weight.*Decimal",
        ),
    ],
)
def test_allocator_rejects_ambiguous_or_unknown_intents(
    intents: list[TargetIntent], policy: AllocationPolicy, error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        DeterministicAllocator().allocate(intents, policy)


def test_allocator_rejects_an_asset_without_an_explicit_budget() -> None:
    with pytest.raises(ValueError, match="missing asset-class budget for stock"):
        _policy(
            strategy_budgets={"rotation": Decimal("0.5")},
            asset_class_budgets={AssetType.ETF: Decimal("0.8")},
            asset_types={STOCK: AssetType.STOCK},
        )


def test_score_does_not_affect_allocation_math() -> None:
    policy = _policy(
        strategy_budgets={"rotation": Decimal("0.5")},
        asset_class_budgets={AssetType.ETF: Decimal("0.8")},
        asset_types={ETF: AssetType.ETF},
    )

    ordinary = DeterministicAllocator().allocate(
        [_intent("rotation", ETF, Decimal("1"), score=1.0)], policy
    )
    malicious = DeterministicAllocator().allocate(
        [_intent("rotation", ETF, Decimal("1"), score=float("nan"))], policy
    )

    assert malicious == ordinary


def test_output_mappings_are_immutable_and_stably_sorted() -> None:
    result = DeterministicAllocator().allocate(
        [
            _intent("trend", STOCK, Decimal("0.5")),
            _intent("rotation", ETF_2, Decimal("0.5")),
            _intent("rotation", ETF, Decimal("0.5")),
        ],
        _policy(),
    )

    assert list(result.sleeves) == ["rotation", "trend"]
    assert list(result.weights) == ["SSE.510300", "SSE.600000", "SZSE.159915"]
    assert list(result.symbol_asset_types) == ["SSE.510300", "SSE.600000", "SZSE.159915"]
    assert list(result.sleeves["rotation"].weights) == ["SSE.510300", "SZSE.159915"]
    with pytest.raises(TypeError):
        result.weights["SSE.510300"] = Decimal("0")  # type: ignore[index]
    with pytest.raises(TypeError):
        result.sleeves["rotation"].weights["SSE.510300"] = Decimal("0")  # type: ignore[index]
    with pytest.raises(TypeError):
        result.symbol_asset_types["SSE.510300"] = AssetType.STOCK  # type: ignore[index]


def test_sleeve_contributions_sum_exactly_to_each_portfolio_weight() -> None:
    result = DeterministicAllocator().allocate(
        [
            _intent("rotation", ETF, Decimal("0.7")),
            _intent("rotation", STOCK, Decimal("0.3")),
            _intent("trend", ETF, Decimal("0.333333333333")),
        ],
        _policy(cash_reserve=Decimal("0.25")),
    )

    for symbol, target in result.weights.items():
        assert sum(
            (sleeve.weights.get(symbol, Decimal("0")) for sleeve in result.sleeves.values()),
            Decimal("0"),
        ) == target
    assert result.cash_weight == Decimal("1") - sum(
        (weight for weight in result.weights.values() if weight > 0), Decimal("0")
    )


def test_allocation_is_independent_of_global_decimal_context() -> None:
    policy = _policy(cash_reserve=Decimal("0.23456789"))
    intents = [
        _intent("rotation", ETF, Decimal("0.987654321012")),
        _intent("rotation", STOCK, Decimal("0.876543210123")),
        _intent("trend", ETF_2, Decimal("0.765432101234")),
    ]
    original = getcontext().copy()
    try:
        baseline = DeterministicAllocator().allocate(intents, policy)
        getcontext().prec = 2
        getcontext().rounding = ROUND_FLOOR
        constrained = DeterministicAllocator().allocate(intents, policy)
    finally:
        setcontext(original)

    assert constrained == baseline


def test_huge_decimal_exponents_are_rejected_without_decimal_arithmetic() -> None:
    tiny = Decimal("1e-1000000")
    with pytest.raises(ValueError, match="12 decimal places"):
        DeterministicAllocator().allocate(
            [
                _intent("rotation", ETF, Decimal("1")),
                _intent("rotation", ETF_2, tiny),
            ],
            _policy(
                strategy_budgets={"rotation": Decimal("1")},
                asset_class_budgets={AssetType.ETF: Decimal("0.9")},
                asset_types={ETF: AssetType.ETF, ETF_2: AssetType.ETF},
                cash_reserve=Decimal("0.1"),
            ),
        )


def test_allocator_rejects_inputs_that_previously_created_a_negative_last_item() -> None:
    policy = _policy(
        strategy_budgets={"rotation": Decimal("1")},
        asset_class_budgets={AssetType.ETF: Decimal("0.9")},
        asset_types={ETF: AssetType.ETF, ETF_2: AssetType.ETF, STOCK: AssetType.ETF},
        cash_reserve=Decimal("0.1"),
    )
    intents = [
        _intent("rotation", ETF, Decimal("0.8673257233673126292192545022")),
        _intent("rotation", ETF_2, Decimal("0.4572783003697858148922256442")),
        _intent("rotation", STOCK, Decimal("1E-72")),
    ]

    with pytest.raises(ValueError, match="12 decimal places"):
        DeterministicAllocator().allocate(intents, policy)


def test_policy_rejects_sub_unit_budget_overflow_instead_of_rounding_the_sum() -> None:
    with pytest.raises(ValueError, match="12 decimal places"):
        AllocationPolicy(
            strategy_budgets={
                "rotation": Decimal("0.50000000000000000000000000001"),
                "trend": Decimal("0.50000000000000000000000000001"),
            },
            asset_class_budgets={AssetType.ETF: Decimal("1")},
            asset_types={ETF: AssetType.ETF},
            cash_reserve=Decimal("0"),
        )


def test_extreme_exponent_is_rejected_with_a_stable_validation_error() -> None:
    with pytest.raises(ValueError, match="12 decimal places"):
        DeterministicAllocator().allocate(
            [_intent("rotation", ETF, Decimal("1E-1000000"))],
            _policy(
                strategy_budgets={"rotation": Decimal("1")},
                asset_class_budgets={AssetType.ETF: Decimal("1")},
                asset_types={ETF: AssetType.ETF},
                cash_reserve=Decimal("0"),
            ),
        )


def test_asset_summary_is_exactly_derived_from_final_symbol_weights() -> None:
    result = DeterministicAllocator().allocate(
        [
            _intent("rotation", ETF, Decimal("0.01851078")),
            _intent("rotation", ETF_2, Decimal("0.99999998")),
            _intent("rotation", STOCK, Decimal("0.00000001")),
            _intent("trend", ETF, Decimal("0")),
            _intent("trend", ETF_2, Decimal("0")),
            _intent("trend", STOCK, Decimal("0")),
        ],
        _policy(
            strategy_budgets={"rotation": Decimal("0.55"), "trend": Decimal("0.35")},
            asset_class_budgets={AssetType.ETF: Decimal("0.65"), AssetType.STOCK: Decimal("0.25")},
            asset_types={ETF: AssetType.ETF, ETF_2: AssetType.ETF, STOCK: AssetType.STOCK},
            cash_reserve=Decimal("0.15"),
        ),
    )

    assert result.asset_weights[AssetType.STOCK] == result.weights["SSE.600000"]
    assert result.asset_weights[AssetType.ETF] == (
        result.weights["SSE.510300"] + result.weights["SZSE.159915"]
    )


def test_cash_limit_is_applied_after_symbol_net_then_reattributed_to_sleeves() -> None:
    two_units = Decimal("0.000000000002")
    result = DeterministicAllocator().allocate(
        [
            _intent("rotation", ETF, two_units),
            _intent("rotation", ETF_2, two_units),
            _intent("trend", ETF, two_units),
        ],
        _policy(
            strategy_budgets={"rotation": Decimal("0.5"), "trend": Decimal("0.5")},
            asset_class_budgets={AssetType.ETF: Decimal("1")},
            asset_types={ETF: AssetType.ETF, ETF_2: AssetType.ETF},
            cash_reserve=Decimal("0.999999999998"),
        ),
    )

    assert result.weights == {
        "SSE.510300": Decimal("0.000000000001"),
        "SZSE.159915": Decimal("0.000000000001"),
    }
    assert result.sleeves["rotation"].weights == {
        "SSE.510300": Decimal("0.000000000001"),
        "SZSE.159915": Decimal("0.000000000001"),
    }
    assert result.sleeves["trend"].weights == {"SSE.510300": Decimal("0")}
    cash_adjustment = next(
        adjustment
        for adjustment in result.adjustments
        if adjustment.stage is AllocationStage.CASH_RESERVE
    )
    assert cash_adjustment.residue_recipients == ("SZSE.159915",)
    etf_attribution = next(
        adjustment
        for adjustment in result.adjustments
        if adjustment.stage is AllocationStage.SLEEVE_ATTRIBUTION
        and adjustment.group == "symbol:SSE.510300"
    )
    assert etf_attribution.before_units == 2
    assert etf_attribution.after_units == 1
    assert etf_attribution.residue_recipients == ("rotation/SSE.510300",)


def test_sleeve_rejects_replayable_normalization_and_budget_tampering() -> None:
    result = _replay_target()
    sleeve = result.sleeves["rotation"]

    with pytest.raises(ValueError, match="normalized.*replay"):
        replace(
            sleeve,
            normalized_weights={
                "SSE.510300": Decimal("0.7"),
                "SSE.600000": Decimal("0.3"),
            },
        )
    with pytest.raises(ValueError, match="budgeted.*replay"):
        replace(
            sleeve,
            budgeted_weights={
                "SSE.510300": Decimal("0.39"),
                "SSE.600000": Decimal("0.21"),
            },
        )


def test_portfolio_rejects_missing_or_reordered_adjustment_trace() -> None:
    result = _replay_target()

    with pytest.raises(ValueError, match="adjustment trace"):
        replace(result, adjustments=())
    with pytest.raises(ValueError, match="adjustment trace"):
        replace(result, adjustments=tuple(reversed(result.adjustments)))


def test_portfolio_rejects_policy_and_adjustment_limit_tampering() -> None:
    result = _replay_target()

    with pytest.raises(ValueError, match="replay"):
        replace(
            result,
            policy=replace(result.policy, cash_reserve=Decimal("0.2")),
        )
    cash_index = next(
        index
        for index, adjustment in enumerate(result.adjustments)
        if adjustment.stage is AllocationStage.CASH_RESERVE
    )
    tampered = list(result.adjustments)
    tampered[cash_index] = replace(
        tampered[cash_index],
        limit_units=tampered[cash_index].limit_units - 1,
    )
    with pytest.raises(ValueError, match="adjustment trace"):
        replace(result, adjustments=tuple(tampered))


def test_adjustment_recipients_require_exact_ordered_tuple_and_replay() -> None:
    result = _replay_target()
    adjustment_index = next(
        index
        for index, adjustment in enumerate(result.adjustments)
        if len(adjustment.residue_recipients) >= 1
    )
    adjustment = result.adjustments[adjustment_index]

    with pytest.raises(TypeError, match="exact tuple"):
        replace(adjustment, residue_recipients=adjustment.residue_recipients[0])
    with pytest.raises(ValueError, match="unique"):
        replace(
            adjustment,
            residue_recipients=(
                adjustment.residue_recipients[0],
                adjustment.residue_recipients[0],
            ),
        )
    tampered = list(result.adjustments)
    tampered[adjustment_index] = replace(
        adjustment,
        residue_recipients=tuple(reversed(adjustment.input_keys)),
    )
    with pytest.raises(ValueError, match="adjustment trace"):
        replace(result, adjustments=tuple(tampered))


def test_final_weights_are_explicit_with_read_only_legacy_aliases() -> None:
    sleeve = _replay_target().sleeves["rotation"]

    assert sleeve.final_weights == sleeve.weights == sleeve.pre_net_weights


def test_trace_input_identifiers_are_canonical_with_prefix_strategy_ids() -> None:
    result = DeterministicAllocator().allocate(
        [
            _intent("a", ETF, Decimal("0.4")),
            _intent("a-b", ETF, Decimal("0.4")),
        ],
        _policy(
            strategy_budgets={"a": Decimal("0.5"), "a-b": Decimal("0.5")},
            asset_class_budgets={AssetType.ETF: Decimal("1")},
            asset_types={ETF: AssetType.ETF},
            cash_reserve=Decimal("0"),
        ),
    )

    assert all(
        adjustment.input_keys == tuple(sorted(adjustment.input_keys))
        for adjustment in result.adjustments
    )


@settings(max_examples=80, deadline=None)
@given(
    weights=st.lists(
        st.decimals(
            min_value=Decimal("0"),
            max_value=Decimal("1"),
            allow_nan=False,
            allow_infinity=False,
            places=12,
        ),
        min_size=6,
        max_size=6,
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_property_invariants_and_shuffle_independence(
    weights: list[Decimal], seed: int
) -> None:
    instruments = [ETF, ETF_2, STOCK]
    intents = [
        _intent(strategy, instrument, weights[strategy_index * 3 + instrument_index])
        for strategy_index, strategy in enumerate(("rotation", "trend"))
        for instrument_index, instrument in enumerate(instruments)
    ]
    policy = _policy(
        strategy_budgets={"rotation": Decimal("0.55"), "trend": Decimal("0.35")},
        asset_class_budgets={AssetType.ETF: Decimal("0.65"), AssetType.STOCK: Decimal("0.25")},
        asset_types={ETF: AssetType.ETF, ETF_2: AssetType.ETF, STOCK: AssetType.STOCK},
        cash_reserve=Decimal("0.15"),
    )
    expected = DeterministicAllocator().allocate(intents, policy)
    shuffled = list(intents)
    Random(seed).shuffle(shuffled)
    actual = DeterministicAllocator().allocate(shuffled, policy)

    assert actual == expected
    assert all(Decimal("0") <= weight <= Decimal("1") for weight in actual.weights.values())
    assert sum(actual.weights.values()) <= Decimal("1")
    assert actual.asset_weights[AssetType.STOCK] <= Decimal("0.25")
    assert actual.asset_weights[AssetType.ETF] <= Decimal("0.65")
    assert actual.cash_weight >= Decimal("0.15")
    for symbol, target in actual.weights.items():
        assert sum(
            (sleeve.weights.get(symbol, Decimal("0")) for sleeve in actual.sleeves.values()),
            Decimal("0"),
        ) == target
    for asset_type, target in actual.asset_weights.items():
        assert sum(
            (
                weight
                for symbol, weight in actual.weights.items()
                if actual.symbol_asset_types[symbol] is asset_type
            ),
            Decimal("0"),
        ) == target
    assert all(
        weight >= 0
        for sleeve in actual.sleeves.values()
        for stage in (
            sleeve.requested_weights,
            sleeve.normalized_weights,
            sleeve.budgeted_weights,
            sleeve.asset_limited_weights,
            sleeve.pre_net_weights,
        )
        for weight in stage.values()
    )
