from pathlib import Path
from uuid import uuid4
import asyncio

from nicegui import ui
from nicegui.events import UploadEventArguments

from compass.services.backup_service import BackupService


def render_backup_panel(root: Path) -> None:
    service = BackupService(root)
    with ui.expansion("备份与恢复", icon="backup").classes("w-full border rounded"):
        ui.label("完整备份包括数据库、行情、账户、研究配置、报告与日志。").classes("text-sm")
        feedback = ui.label("").classes("text-sm text-slate-600")

        async def create() -> None:
            backup_button.disable()
            feedback.set_text("正在制作并校验备份……")
            try:
                path = await asyncio.to_thread(service.create)
                await asyncio.to_thread(service.inspect, path)
                ui.download(path)
                feedback.set_text("备份已生成，浏览器开始下载。")
            except Exception as error:
                feedback.set_text(f"备份未完成：{error}")
            finally:
                backup_button.enable()

        backup_button = ui.button("下载完整备份", icon="download", on_click=create)
        ui.label("恢复会在下次启动时应用；恢复前的数据会保存在运行目录的 .recovery 中。").classes(
            "text-sm"
        )

        async def upload(event: UploadEventArguments) -> None:
            incoming = root / ".recovery" / f"upload-{uuid4().hex}.zip"
            incoming.parent.mkdir(parents=True, exist_ok=True)
            try:
                await event.file.save(incoming)
                summary = await asyncio.to_thread(service.inspect, incoming)
                with ui.dialog() as dialog, ui.card().classes("max-w-xl"):
                    ui.label("备份已通过校验").classes("font-semibold")
                    ui.label(f"创建时间：{summary['created_at']} · 共 {summary['files']} 个文件")
                    ui.label("确认后将安排下次启动恢复。请先保存当前工作，再手动重启应用。")

                    async def stage() -> None:
                        apply_button.disable()
                        try:
                            await asyncio.to_thread(service.stage, incoming)
                            feedback.set_text("已安排恢复，请停止并重新启动 Compass。")
                            dialog.close()
                        except Exception as error:
                            feedback.set_text(f"恢复未安排：{error}")
                        finally:
                            apply_button.enable()

                    with ui.row():
                        ui.button("取消", on_click=dialog.close).props("flat")
                        apply_button = ui.button("下次启动时恢复此备份", on_click=stage)
                dialog.open()
            except Exception as error:
                feedback.set_text(f"备份无法使用：{error}")

        ui.upload(
            label="选择 Compass 备份文件",
            on_upload=upload,
            auto_upload=True,
            max_file_size=512 * 1024**2,
        ).props("accept=.zip max-files=1").classes("w-full")

        def cancel() -> None:
            service.cancel_pending()
            feedback.set_text("已取消待恢复任务，当前数据保持不变。")

        if (service.recovery / "pending.json").exists():
            feedback.set_text("有一份备份等待下次启动恢复。")
        ui.button("取消待恢复任务", on_click=cancel).props("flat")
