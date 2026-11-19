#!/usr/bin/env bash
# 실행 스크립트 (선택 산출물). .env 는 app.py 에서 python-dotenv 로 읽습니다.
set -e
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
