#!/bin/sh
set -eu

APP_ROOT="${APP_PATH:-/home/site/wwwroot}"
cd "$APP_ROOT"

python -m uvicorn main:app --host 0.0.0.0 --port 8000
