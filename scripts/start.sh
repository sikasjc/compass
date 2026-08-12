#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PORT=${COMPASS_PORT:-8080}
UV_COMMAND=${UV_COMMAND:-uv}
ENVIRONMENT_FILE=${COMPASS_ENV_FILE:-$REPOSITORY_ROOT/.env}

case "$PORT" in
    ''|*[!0-9]*)
        printf '%s\n' '启动失败：端口必须是 1 到 65535 之间的整数。' >&2
        exit 17
        ;;
esac
if [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    printf '%s\n' '启动失败：端口必须是 1 到 65535 之间的整数。' >&2
    exit 17
fi
if ! command -v "$UV_COMMAND" >/dev/null 2>&1; then
    printf '%s\n' '启动失败：未找到 uv，请先安装 uv 并加入 PATH。' >&2
    exit 13
fi

cd "$REPOSITORY_ROOT"
"$UV_COMMAND" sync

if [ -f "$ENVIRONMENT_FILE" ]; then
    exec "$UV_COMMAND" run --env-file "$ENVIRONMENT_FILE" \
        python -m compass.ui.app --port "$PORT"
fi
if [ -n "${COMPASS_ENV_FILE:-}" ]; then
    printf '%s\n' '启动失败：指定的环境文件不存在。' >&2
    exit 16
fi
exec "$UV_COMMAND" run python -m compass.ui.app --port "$PORT"
