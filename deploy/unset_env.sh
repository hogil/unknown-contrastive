#!/usr/bin/env bash
# deploy 관련 환경변수 해제 (Ubuntu / bash)
#
#   source deploy/unset_env.sh
#
# ★ source 로 실행해야 한다. `bash unset_env.sh` 는 자식 셸에서만 지워져 소용없다.
#
# 왜 필요한가: 예전에는 환경변수가 config.py 를 항상 이겨서, 한 번 export 해두면
# config 를 고쳐도 조용히 무시됐다. 지금은 SITE_ENV_OVERRIDE=1 일 때만 환경변수가
# 이기지만, 깨끗하게 지우고 싶을 때 쓴다.

for v in $(env | grep -oE '^(SITE_|UC_|REPRO_)[A-Z0-9_]*'); do unset "$v"; done

echo "남은 것:"
env | grep -E '^(SITE_|UC_|REPRO_)' || echo "  (없음)"
