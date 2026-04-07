#!/usr/bin/env bash
# DDP benchmark suite — runs all modes back to back.
# Usage: ./bench_ddp.sh [EPOCHS]    (default: 5)
#        ./bench_ddp.sh 50           (overnight run)
set -euo pipefail

EPOCHS="${1:-5}"
BATCH=52
BENCH_DIR="runs/ddp_bench"          # relative to Docker CWD (letter/)
HOST_BENCH="letter/$BENCH_DIR"      # same dir, relative to repo root
MONITOR=3000

# All modes and their directory names.
MODES=(
    "solo_5060ti"
    "solo_1060"
    "sync"
    "async-nccl-sync"
    "async-nccl-cadence"
    "async-nccl-async"
    "async-cpu-sync"
    "async-cpu-cadence"
    "async-cpu-async"
)

# Docker-relative dirs (passed to make)
DIRS=(
    "$BENCH_DIR/solo_5060ti"
    "$BENCH_DIR/solo_1060"
    "$BENCH_DIR/sync"
    "$BENCH_DIR/async_nccl_sync"
    "$BENCH_DIR/async_nccl_cadence"
    "$BENCH_DIR/async_nccl_async"
    "$BENCH_DIR/async_cpu_sync"
    "$BENCH_DIR/async_cpu_cadence"
    "$BENCH_DIR/async_cpu_async"
)

# Host-relative dirs (for mkdir, cp, reading results)
HOST_DIRS=()
for d in "${DIRS[@]}"; do
    HOST_DIRS+=("letter/$d")
done

TOTAL=${#MODES[@]}

echo "=== DDP Benchmark Suite ==="
echo "Epochs: $EPOCHS  Batch: $BATCH  Modes: $TOTAL"
echo ""

# Ensure template config exists at bench root
TEMPLATE="$HOST_BENCH/gen_config.json"
if [ ! -f "$TEMPLATE" ]; then
    # Bootstrap from any existing mode's config
    src=$(find "$HOST_BENCH" -name gen_config.json -print -quit 2>/dev/null)
    if [ -n "$src" ]; then
        cp "$src" "$TEMPLATE"
    else
        echo "ERROR: no gen_config.json template found in $HOST_BENCH/" >&2
        exit 1
    fi
fi

for i in "${!MODES[@]}"; do
    mode="${MODES[$i]}"
    dir="${DIRS[$i]}"
    host_dir="${HOST_DIRS[$i]}"
    mkdir -p "$host_dir"
    cp -n "$TEMPLATE" "$host_dir/gen_config.json" 2>/dev/null || true

    echo "--- [$((i+1))/$TOTAL] $mode ---"

    case "$mode" in
        solo_5060ti)
            make train-letter GEN="$dir/gen_config.json" SAVE="$dir" EPOCHS="$EPOCHS" BATCH="$BATCH" GPU=0 MONITOR="$MONITOR"
            ;;
        solo_1060)
            make train-letter GEN="$dir/gen_config.json" SAVE="$dir" EPOCHS="$EPOCHS" BATCH="$BATCH" GPU=1 MONITOR="$MONITOR"
            ;;
        sync)
            make train-letter GEN="$dir/gen_config.json" SAVE="$dir" EPOCHS="$EPOCHS" BATCH="$BATCH" DDP=sync MONITOR="$MONITOR"
            ;;
        *)
            make train-letter GEN="$dir/gen_config.json" SAVE="$dir" EPOCHS="$EPOCHS" BATCH="$BATCH" DDP="$mode" MONITOR="$MONITOR"
            ;;
    esac
    echo ""
done

# --- Summary ---
echo "=== Results ==="
printf "%-25s %12s %12s %12s %12s\n" "Mode" "Avg epoch" "Total" "letter_acc" "case_acc"
echo "------------------------------------------------------------------------------------"
for i in "${!MODES[@]}"; do
    host_dir="${HOST_DIRS[$i]}"
    mode="${MODES[$i]}"
    if [ -f "$host_dir/benchmark.json" ] && [ -f "$host_dir/training.csv" ]; then
        stats=$(python3 -c "
import json, csv
d=json.load(open('$host_dir/benchmark.json'))
# Read last row of training.csv for final accuracy
with open('$host_dir/training.csv') as f:
    rows=list(csv.DictReader(f))
    if rows:
        last=rows[-1]
        la=float(last.get('letter_acc','0'))
        ca=float(last.get('case_acc','0'))
    else:
        la=ca=0
print(f'{d[\"avg_epoch_s\"]:>10.1f}s {d[\"total_time_s\"]:>10.1f}s {la:>10.1%} {ca:>10.1%}')
" 2>/dev/null || echo "         parse error")
        printf "%-25s %s\n" "$mode" "$stats"
    fi
done
echo ""
echo "Done."
