from urllib.parse import urlencode


def account_url(route: str, account_id: str) -> str:
    return f"{route}?{urlencode({'account_id': account_id})}"


def draft_url(route: str, draft_id: str) -> str:
    return f"{route}?{urlencode({'draft_id': draft_id})}"
