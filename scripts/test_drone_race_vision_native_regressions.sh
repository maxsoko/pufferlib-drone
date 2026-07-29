#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/build/test_drone_race_vision_native}"
CC="${CC:-clang}"
CFLAGS="${CFLAGS:--O2 -DNDEBUG}"

mkdir -p "$(dirname "$OUT")"
"$CC" $CFLAGS -I"$ROOT" -I"$ROOT/src" -I"$ROOT/vendor" \
    "$ROOT/tests/test_drone_race_vision_native.c" -lm -o "$OUT"
"$OUT"
