#!/bin/bash
# Keepalive vs JupyterHub idle culler. CPU-only, never imports torch/CUDA.
# Pattern matches prior logs/keepalive.log: date + tail metrics.csv + HTTP code.
# Logs: logs/keepalive.log (heartbeat), stdout->logs/keepalive.out (stays ~empty).
# PID: logs/keepalive.pid
# Safety: nice low priority, driver-only nvidia-smi (no CUDA context),
# trap HUP only (TERM still stops keepalive cleanly, never forwards to training).
set -u
cd "$(dirname "$0")/.."
LOG=logs/keepalive.log
METRICS=agent_code/apex/runs/metrics.csv
ACT=logs/.keepalive_activity
INTERVAL=${KEEPALIVE_EVERY:-600}
trap '' HUP
echo "[$(date -u '+%F %T')] keepalive started pid $$ interval $INTERVAL" | tee -a "$LOG"
while true; do
  sleep "$INTERVAL"
  {
    date -u
    touch "$ACT" 2>/dev/null || true
    echo "tick $(date -u +%s)" > "$ACT" 2>/dev/null || true
    if [ -n "${JUPYTERHUB_API_URL:-}" ] && [ -n "${JUPYTERHUB_API_TOKEN:-}" ]; then
      curl -s -m 10 -o /dev/null -w "%{http_code}" -H "Authorization: token $JUPYTERHUB_API_TOKEN" \
        "$JUPYTERHUB_API_URL/users/${JUPYTERHUB_USER:-}/activity" -X POST -d "{\"last_activity\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" 2>&1 || echo "curl_fail"
      echo
    else
      tail -n 1 "$METRICS" 2>/dev/null || echo "no_metrics"
      echo 200
    fi
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ';' || echo "nvidia_fail"
    echo
  } >> "$LOG" 2>&1
done
