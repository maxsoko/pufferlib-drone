#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/build/eval_drone_hover_edge}"
WEIGHTS="${WEIGHTS:-$ROOT/resources/drone/drone_weights.bin}"
STEPS="${STEPS:-1024}"
AGENTS="${AGENTS:-64}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
LAYERS="${LAYERS:-3}"
CC="${CC:-clang}"
CFLAGS="${CFLAGS:--O3 -DNDEBUG}"
COMPILE_ONLY="${COMPILE_ONLY:-0}"

mkdir -p "$(dirname "$OUT")"
"$CC" $CFLAGS -I"$ROOT" -I"$ROOT/src" -I"$ROOT/ocean/drone" -I"$ROOT/vendor" \
    "$ROOT/tests/eval_drone_hover_edge.c" -lm -o "$OUT"

if [ "$COMPILE_ONLY" = "1" ]; then
    echo "built $OUT"
    exit 0
fi

"$OUT" "$WEIGHTS" "$STEPS" "$AGENTS" "$HIDDEN_SIZE" "$LAYERS"
