from __future__ import annotations

from compass.services.task_manager import TaskStatus


def task_status_label(status: TaskStatus) -> str:
    if type(status) is not TaskStatus:
        raise TypeError("status must be an exact TaskStatus")
    return {
        TaskStatus.QUEUED: "排队中",
        TaskStatus.RUNNING: "运行中",
        TaskStatus.CANCELLATION_REQUESTED: "请求取消",
        TaskStatus.CANCELLED: "已取消",
        TaskStatus.SUCCEEDED: "已完成",
        TaskStatus.FAILED: "失败",
    }[status]
