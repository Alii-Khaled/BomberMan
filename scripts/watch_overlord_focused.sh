#!/bin/bash
# Focused-run guard (E26 program): divergence kill-switch + completion note.
# Unlike watch_overlord.sh this watcher NEVER launches anything — the next
# step after W3 stays a manual decision ("analyze first" rule).
# Logs: logs/watcher_focused.log. Progress: tail metrics.csv.
set -u
cd "$(dirname "$0")/.."

WLOG=logs/watcher_focused.log
METRICS=agent_code/overlord/runs/metrics.csv
POLL=60
GUARD_EVERY=15
WNORM_KILL=300
WNORM_WARN=100
LOSS_KILL=5.0

say()  { echo "[$(date '+%F %T')] $*" | tee -a "$WLOG"; }

alive_focused() { pgrep -f "[t]rain_overlord_focused" >/dev/null || pgrep -f "[t]rain_overlord_champ" >/dev/null || pgrep -f "[t]rain_overlord_cratelight" >/dev/null || pgrep -f "[t]rain_overlord_collector" >/dev/null; }
alive_play()    { pgrep -f "main.py pla[y].*overlord" >/dev/null; }

kill_training() {
  say "KILL-SWITCH: $1 — stopping focused training (checkpoints bound loss to <=5 rounds)"
  pkill -KILL -f "main.py pla[y].*overlord" 2>/dev/null || true
  sleep 5
  pkill -KILL -f "[t]rain_overlord_focused" 2>/dev/null || true
}

guard_check() {
  [ -f "$METRICS" ] || return 1
  local verdict
  verdict=$(python3 - "$METRICS" "$WNORM_KILL" "$WNORM_WARN" "$LOSS_KILL" <<'EOF'
import csv, statistics, sys
path, wkill, wwarn, lkill = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
try:
    with open(path) as f:
        rows = list(csv.DictReader(f))
except Exception:
    print("NOFILE"); sys.exit(0)
if len(rows) < 20:
    print("TOOSHORT"); sys.exit(0)
try:
    wlast = float(rows[-1].get('wnorm') or 0)
    losses = [float(r['loss']) for r in rows[-20:] if float(r.get('loss') or 0) > 0]
    lmed = statistics.median(losses) if losses else 0.0
except Exception:
    print("PARSEFAIL"); sys.exit(0)
if wlast > wkill:
    print(f"DIVERGED wnorm={wlast:.1f} > {wkill}")
elif lmed > lkill:
    print(f"DIVERGED lossMed20={lmed:.2f} > {lkill}")
elif wlast > wwarn:
    print(f"WARN wnorm={wlast:.1f} lossMed20={lmed:.4f}")
else:
    print(f"OK wnorm={wlast:.1f} lossMed20={lmed:.4f}")
EOF
)
  say "guard: $verdict"
  case "$verdict" in
    DIVERGED*) return 0 ;;
    *) return 1 ;;
  esac
}

say "focused watcher started (pid $$): kill-switch armed, no auto-launch"
n=0
while alive_focused || alive_play; do
  sleep "$POLL"
  n=$((n + 1))
  if [ $((n % GUARD_EVERY)) -eq 0 ]; then
    if guard_check; then
      kill_training "$(tail -n 1 "$WLOG")"
      say "watcher exiting after kill-switch (manual relaunch required)"
      exit 0
    fi
  fi
done
say "focused run finished (no train_overlord_focused / overlord play processes left)"
if [ -s results/overlord_focused_eval_rb.json ] && [ -s results/overlord_focused_eval_warden.json ]; then
  say "both frozen evals present — ready for manual analysis (E29 material), no further action"
else
  say "eval artifacts incomplete — inspect before next step (manual decision required)"
fi
say "watcher exiting"
