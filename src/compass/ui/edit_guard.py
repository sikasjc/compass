import json

from nicegui import ui
from nicegui.awaitable_response import AwaitableResponse


class EditGuard:
    """Track edits in a form and ask before discarding them."""

    def __init__(self, *, scope: str | None = None) -> None:
        selector = scope or f"#c{ui.context.slot.parent.id}"
        self._checking = False
        ui.on("compass-navigate", lambda event: self.navigate(str(event.args["target"])))
        ui.add_body_html("""<script>
        window.compassDirty = false;
        const editScope = """ + json.dumps(selector) + """;
        const markEdit = event => {
            if (event.target.closest(editScope) &&
                !event.target.closest('.compass-edit-ignore')) window.compassDirty = true;
        };
        document.addEventListener('input', markEdit);
        document.addEventListener('change', markEdit);
        window.addEventListener('beforeunload', event => {
            if (window.compassDirty) { event.preventDefault(); event.returnValue = ''; }
        });
        document.addEventListener('click', event => {
            const link = event.target.closest('a[href]');
            if (link && window.compassDirty && !event.ctrlKey && !event.metaKey &&
                !event.shiftKey && link.target !== '_blank' && !link.hasAttribute('download')) {
                event.preventDefault(); event.stopImmediatePropagation();
                emitEvent('compass-navigate', {target: link.href});
            }
        }, true);
        </script>""")

    def clear(self) -> AwaitableResponse:
        return ui.run_javascript("window.compassDirty = false", timeout=10)

    def mark(self) -> None:
        ui.run_javascript("window.compassDirty = true")

    async def confirm_leave(self) -> bool:
        if self._checking:
            return False
        self._checking = True
        try:
            if not await ui.run_javascript("Boolean(window.compassDirty)", timeout=10):
                return True
            with ui.dialog().props("persistent") as dialog, ui.card():
                ui.label("还有未保存的修改，确定放弃并继续吗？")
                with ui.row():
                    ui.button("继续编辑", on_click=lambda: dialog.submit(False)).props("flat")
                    ui.button("放弃修改并继续", on_click=lambda: dialog.submit(True))
            try:
                return bool(await dialog)
            finally:
                dialog.delete()
        except TimeoutError:
            ui.notify("页面连接暂时不可用，修改尚未丢弃，请重试。", type="warning")
            return False
        finally:
            self._checking = False

    async def navigate(self, target: str) -> bool:
        if not await self.confirm_leave():
            return False
        await self.clear()
        ui.navigate.to(target)
        return True
