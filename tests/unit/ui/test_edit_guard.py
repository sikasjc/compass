import asyncio

import pytest
from nicegui import ui

from compass.ui.edit_guard import EditGuard


@pytest.mark.parametrize("confirmed", [False, True])
def test_slow_confirmation_preserves_cancel_and_navigates_only_when_accepted(monkeypatch, confirmed):
    scripts = []
    destinations = []
    dialogs = []
    original_dialog = ui.dialog

    async def javascript(code, **kwargs):
        scripts.append(code)
        return True if code == "Boolean(window.compassDirty)" else None

    def dialog_factory(*args, **kwargs):
        dialog = original_dialog(*args, **kwargs)
        dialogs.append(dialog)
        return dialog

    monkeypatch.setattr(ui, "run_javascript", javascript)
    monkeypatch.setattr(ui, "dialog", dialog_factory)
    monkeypatch.setattr(ui.navigate, "to", destinations.append)

    with ui.column() as container:
        guard = EditGuard()

    async def exercise():
        async def navigate_with_slot():
            # NiceGUI keeps the active UI slot per asyncio task. Real event
            # handlers receive it automatically; this synthetic task must
            # enter the container explicitly before the dialog is created.
            with container:
                return await guard.navigate("/signals?account_id=b")

        navigation = asyncio.create_task(navigate_with_slot())
        await asyncio.sleep(1.2)
        assert not navigation.done()
        dialogs[-1].submit(confirmed)
        assert await navigation is confirmed

    asyncio.run(exercise())
    container.delete()
    assert destinations == (["/signals?account_id=b"] if confirmed else [])
    assert ("window.compassDirty = false" in scripts) is confirmed
    assert all("confirm(" not in code for code in scripts)
