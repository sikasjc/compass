from compass.domain.runs import DecisionStage


def test_decision_stages_describe_the_four_step_decision_pipeline() -> None:
    assert tuple(stage.name for stage in DecisionStage) == (
        "RAW",
        "ALLOCATED",
        "RISK_ADJUSTED",
        "FINAL",
    )
    assert tuple(stage.value for stage in DecisionStage) == (
        "RAW",
        "ALLOCATED",
        "RISK_ADJUSTED",
        "FINAL",
    )
