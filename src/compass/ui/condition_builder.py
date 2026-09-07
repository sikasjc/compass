from decimal import Decimal, InvalidOperation


INDICATORS = {
    "close": "收盘价",
    "open": "开盘价",
    "volume": "成交量",
    "sma(close, 20)": "20 日均线",
    "sma(close, 60)": "60 日均线",
    "rsi(close, 14)": "14 日 RSI",
    "position_return": "持仓收益率",
    "holding_days": "持仓天数",
}
OPERATORS = {
    ">": "大于",
    "<": "小于",
    ">=": "不低于",
    "<=": "不高于",
    "cross_above": "向上穿越",
    "cross_below": "向下穿越",
}


def build_condition(left: str, operator: str, right: str) -> str:
    if left not in INDICATORS or operator not in OPERATORS:
        raise ValueError("请选择有效的指标和比较方式。")
    if right not in INDICATORS:
        try:
            number = Decimal(right)
        except InvalidOperation:
            raise ValueError("比较数值必须是数字。") from None
        if not number.is_finite() or abs(number) > Decimal("1e12"):
            raise ValueError("比较数值须为有限数字，绝对值不超过一万亿。")
        right = format(number, "f")
    if operator.startswith("cross_"):
        return f"{operator}({left}, {right})"
    return f"{left} {operator} {right}"
