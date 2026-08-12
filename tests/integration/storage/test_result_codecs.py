from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from compass.storage.backtest_result_codec import (
    decode_backtest_result,
    encode_backtest_result,
)
from compass.storage.decision_result_codec import (
    decode_decision_result,
    encode_decision_result,
)
from tests.integration.services.test_decision_service import _audited_result
from tests.support.sleeve_results import multi_reallocation_result


def test_backtest_codec_rejects_unknown_nested_record_keys() -> None:
    encoded = encode_backtest_result(multi_reallocation_result())
    mutations = (
        lambda payload: payload["orders"][0].__setitem__("unexpected", True),
        lambda payload: payload["fills"][0].__setitem__("unexpected", True),
        lambda payload: payload["ledger"][0].__setitem__("unexpected", True),
        lambda payload: payload["ledger"][1]["positions"][0].__setitem__("unexpected", True),
    )
    for mutate in mutations:
        tampered = deepcopy(encoded)
        mutate(tampered)
        with pytest.raises(ValueError, match="BACKTEST_RESULT_INTEGRITY"):
            decode_backtest_result(tampered)


def test_decision_codec_rejects_unknown_nested_record_keys(tmp_path: Path) -> None:
    encoded = encode_decision_result(_audited_result(tmp_path))
    recommendation = encoded["recommendations"][0]
    mutations = (
        lambda payload: payload["strategy_decisions"][0].__setitem__("unexpected", True),
        lambda payload: payload["recommendations"][0]["raw_intents"][0].__setitem__(
            "unexpected", True
        ),
        lambda payload: payload["recommendations"][0]["allocation_trace"][0].__setitem__(
            "unexpected", True
        ),
        lambda payload: payload["recommendations"][0]["risk_adjustments"][0].__setitem__(
            "unexpected", True
        ),
        lambda payload: payload["recommendations"][0]["costs"].__setitem__("unexpected", True),
    )
    assert recommendation["allocation_trace"]
    assert recommendation["risk_adjustments"]
    for mutate in mutations:
        tampered = deepcopy(encoded)
        mutate(tampered)
        with pytest.raises(ValueError, match="DECISION_RESULT_INTEGRITY"):
            decode_decision_result(tampered)
