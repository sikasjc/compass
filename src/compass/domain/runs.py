from enum import StrEnum


class DecisionStage(StrEnum):
    RAW = "RAW"
    ALLOCATED = "ALLOCATED"
    RISK_ADJUSTED = "RISK_ADJUSTED"
    FINAL = "FINAL"
