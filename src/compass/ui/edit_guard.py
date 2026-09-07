from nicegui import ui
from nicegui.awaitable_response import AwaitableResponse


class EditGuard:
    """Warn on browser navigation and on application navigation with unsaved edits."""

    def __init__(self) -> None:
        ui.add_body_html("""<script>
        window.compassDirty = false;
        document.addEventListener('input', () => { window.compassDirty = true; });
        document.addEventListener('change', () => { window.compassDirty = true; });
        window.addEventListener('beforeunload', event => {
            if (window.compassDirty) { event.preventDefault(); event.returnValue = ''; }
        });
        document.addEventListener('click', event => {
            if (event.target.closest('a[href]') && window.compassDirty) {
                if (!confirm('还有未保存的修改，确定离开吗？')) {
                    event.preventDefault(); event.stopImmediatePropagation();
                } else { window.compassDirty = false; }
            }
        }, true);
        </script>""")

    def clear(self) -> AwaitableResponse:
        return ui.run_javascript("window.compassDirty = false")

    def mark(self) -> None:
        ui.run_javascript("window.compassDirty = true")

    async def navigate(self, target: str) -> None:
        if await ui.run_javascript(
            "!window.compassDirty || confirm('还有未保存的修改，确定离开吗？')"
        ):
            await self.clear()
            ui.navigate.to(target)
