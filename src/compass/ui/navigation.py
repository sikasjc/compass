from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl
from pathlib import Path
import json
from nicegui import ui


def account_url(route: str, account_id: str) -> str:
    parts = urlsplit(route)
    query = dict(parse_qsl(parts.query))
    query["account_id"] = account_id
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def draft_url(route: str, draft_id: str) -> str:
    return f"{route}?{urlencode({'draft_id': draft_id})}"


def account_storage_key(root: Path | None) -> str:
    return "compass.account:" + (str(root.resolve()) if root else "default")


def redirect_to_account(route: str, default_id: str, root: Path | None) -> None:
    ui.run_javascript("""(() => {
        const key = """ + json.dumps(account_storage_key(root)) + """;
        const reset = new URL(location.href).searchParams.has('reset_account');
        let selected = """ + json.dumps(default_id) + """;
        try {
            if (reset) sessionStorage.removeItem(key);
            else selected = sessionStorage.getItem(key) || selected;
        } catch (_) {}
        const target = new URL(""" + json.dumps(route) + """, location.origin);
        target.searchParams.set('account_id', selected);
        location.replace(target.href);
    })()""")
