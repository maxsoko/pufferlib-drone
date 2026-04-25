#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/build/bench_puffernet_edge}"
BATCH="${BATCH:-64}"
INPUT_DIM="${INPUT_DIM:-23}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LAYERS="${LAYERS:-3}"
ITERS="${ITERS:-1000}"

mkdir -p "$(dirname "$OUT")"

CC="${CC:-clang}"
CFLAGS="${CFLAGS:--O3 -DNDEBUG}"

"$CC" $CFLAGS -I"$ROOT" "$ROOT/tests/bench_puffernet_edge.c" -lm -o "$OUT"
"$OUT" "$BATCH" "$INPUT_DIM" "$HIDDEN_DIM" "$LAYERS" "$ITERS"
