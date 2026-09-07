#!/bin/bash
# Overlord unattended watcher: guards the validation run, then gated-launches
# the full curriculum. Fully detached (own session). Never edits training code;
# only starts/stops processes and writes logs.
#
# Phases:
#   1. Guard validation run (train_overlord_validation) + kill-switch checks.
#   2. On validation exit: gate analysis -> launch full curriculum or HANDOFF.
#   3. Guard full run (train_overlord_curriculum) + kill-switch until done.
#
# Logs: logs/watcher.log (heartbeat), logs/watcher_decision.log (decisions).
# Progress: tail -n 2 logs/watcher.log; tail -n 3 agent_code/overlord/runs/metrics.csv
set -u
cd "$(dirname "$0")/.."

WLOG=logs/watcher.log
DLOG=logs/watcher_decision.log
METRICS=agent_code/overlord/runs/metrics.csv
EVAL_JSON=results/overlord_val_eval.json
POLL=60
GUARD_EVERY=15   # kill-switch check every 15th poll (~15 min)
WNORM_KILL=300
WNORM_WARN=100
LOSS_KILL=5.0

say()  { echo "[$(date '+%F %T')] $*" | tee -a "$WLOG"; }
decide() { echo "[$(date '+%F %T')] $*" | tee -a "$WLOG" >> "$DLOG"; }

alive_val()  { pgrep -f "[t]rain_overlord_validation" >/dev/null; }
alive_full() { pgrep -f "[t]rain_overlord_curriculum" >/dev/null; }
alive_play() { pgrep -f "main.py pla[y].*overlord" >/dev/null; }

kill_training() {
  local why="$1"
  decide "KILL-SWITCH: $why — stopping overlord training (checkpoints bound loss to <=1 round)"
  pkill -KILL -f "main.py pla[y].*overlord" 2>/dev/null || true
  sleep 5
  pkill -KILL -f "[t]rain_overlord_validation" 2>/dev/null || true
  pkill -KILL -f "[t]rain_overlord_curriculum" 2>/dev/null || true
}

# Returns 0 if metrics look diverged (caller kills), 1 if healthy.
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

# Gate analysis after validation. Returns 0 if full run may launch.
gate_analysis() {
  decide "validation process exited — running gate analysis"
  if [ ! -s "$EVAL_JSON" ]; then
    decide "GATE FAIL: frozen eval artifact $EVAL_JSON missing/empty — NOT launching full run"
    return 1
  fi
  local verdict
  verdict=$(python3 - "$METRICS" <<'EOF'
import csv, statistics, sys
try:
    with open(sys.argv[1]) as f:
        rows = list(csv.DictReader(f))
except Exception as ex:
    print(f"GATE FAIL: metrics unreadable ({ex})"); sys.exit(0)
def col(name):
    out = []
    for r in rows:
        try: out.append(float(r.get(name) or 0))
        except ValueError: pass
    return out
o1 = rows[:200]
o1coins = [float(r['coins']) for r in o1 if r.get('coins') not in (None, '')]
o4 = [r for r in rows if 800 < int(float(r.get('episode') or 0)) <= 1300]
o4sui = [float(r['suicides']) for r in o4] or [9]
loss = [float(r['loss']) for r in rows if float(r.get('loss') or 0) > 0]
wlast = float(rows[-1].get('wnorm') or 0)
checks = []
checks.append(("loss falling (tailMed < headMed)", (statistics.median(loss[-100:]) < statistics.median(loss[:100])) if len(loss) >= 200 else (len(loss) > 0 and statistics.median(loss[-50:]) <= statistics.median(loss[:50]))))
checks.append((f"O1 coins>=25 (got {statistics.mean(o1coins):.1f})", bool(o1coins) and statistics.mean(o1coins) >= 25))
checks.append((f"O4 suicide<0.6 (got {statistics.mean(o4sui):.2f})", statistics.mean(o4sui) < 0.6))
checks.append((f"wnorm<100 (got {wlast:.1f})", wlast < 100))
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
print("GATE: " + ("GO" if all(ok for _, ok in checks) else "NOGO"))
EOF
)
  echo "$verdict" | tee -a "$WLOG" >> "$DLOG"
  echo "$verdict" | grep -q "^GATE: GO"
}

launch_full() {
  decide "GATE GO — launching full curriculum (O1-O4 750+1500+1000+2000) in its own detached session, resuming from validation weights per stage-reset convention"
  decide "NOTE: stage re-warm step 39474 ~= eps 0.63 on the 100k decay (safe direction: more exploration); env identical to validation otherwise"
  setsid nohup bash -c 'export OVERLORD_OPT=adam OVERLORD_TUNED=1 OVERLORD_BATCH=1024 OVERLORD_UTD=2 OVERLORD_EOR_UPDATES=12 OVERLORD_EPS_DECAY=100000 OVERLORD_SAVE_EVERY=5 OVERLORD_CHANNELS_LAST=1 OVERLORD_COMPILE=0 OVERLORD_BASE=96 OVERLORD_FC=512 OVERLORD_NORM=bn OVERLORD_DEEP=0; bash scripts/train_overlord_curriculum.sh' \
    > logs/overlord_curriculum.log 2>&1 < /dev/null &
  decide "full curriculum launcher pid: $!"
}

write_handoff() {
  local reason="$1"
  decide "HANDOFF: $reason — writing results/HANDOFF.md, no full-run launch"
  python3 - results/HANDOFF.md "$METRICS" "$reason" <<'EOF'
import csv, statistics, sys
out, mpath, reason = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(mpath) as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    summary = (f"Episodes: {len(rows)}, last ep={last.get('episode')}, "
               f"ema={last.get('ema_reward')}, wnorm={last.get('wnorm')}, loss={last.get('loss')}")
except Exception as ex:
    summary = f"metrics unreadable: {ex}"
with open(out, 'w') as f:
    f.write(f"# Overlord handoff (watcher)\n\n- {reason}\n- {summary}\n\n"
            "Resume: `bash scripts/train_overlord_validation.sh` (auto-resumes last.pt; "
            "set STAGE_O*_N=0 to skip finished stages) or launch full run:\n"
            "`bash scripts/train_overlord_curriculum.sh` (same OVERLORD_* env as validation).\n")
EOF
}

say "watcher started (pid $$): guarding validation run, gated full-launch armed, kill-switch armed"

# ---- Phase 1: guard validation ----
n=0
while alive_val || alive_play; do
  sleep "$POLL"
  n=$((n + 1))
  if [ $((n % GUARD_EVERY)) -eq 0 ]; then
    if guard_check; then
      kill_training "$(tail -n 1 "$WLOG")"
      write_handoff "divergence kill-switch fired during validation"
      say "watcher exiting after kill-switch"
      exit 0
    fi
  fi
done
say "validation run finished (no train_overlord_validation / overlord play processes left)"

# ---- Phase 2: gate + launch ----
if gate_analysis; then
  launch_full
else
  write_handoff "validation gate did not pass"
  say "watcher exiting (no launch)"
  exit 0
fi

# ---- Phase 3: guard full run ----
sleep 120  # let the curriculum process appear
n=0
while alive_full || alive_play; do
  sleep "$POLL"
  n=$((n + 1))
  if [ $((n % GUARD_EVERY)) -eq 0 ]; then
    if guard_check; then
      kill_training "$(tail -n 1 "$WLOG")"
      write_handoff "divergence kill-switch fired during full curriculum"
      say "watcher exiting after kill-switch"
      exit 0
    fi
  fi
done
if [ -s results/overlord_eval.json ]; then
  decide "full curriculum finished, results/overlord_eval.json present — DONE, no further action"
else
  write_handoff "full curriculum process ended without results/overlord_eval.json (interrupted?)"
fi
say "watcher exiting"
