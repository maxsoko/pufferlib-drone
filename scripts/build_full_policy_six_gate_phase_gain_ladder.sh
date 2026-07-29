#!/usr/bin/env bash
# N105: deterministic amplification of N104's learned Gate-4/5/6 encoder
# direction. Native-only; never connects to or resets official FlightSim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n104_phase_adapter/checkpoints/drone_race_full_policy_six_gate_bootstrap/1784295732298/0000000001048576.bin}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/n105_phase_gain_ladder}"

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    echo "missing source checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 2
fi

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/work"
for gain in 4 8 16 32; do
    label="$(printf '%02d' "$gain")"
    step27="$RUN_ROOT/work/gain${label}_col27.bin"
    step28="$RUN_ROOT/work/gain${label}_col27_28.bin"
    output="$RUN_ROOT/checkpoints/gain${label}.bin"
    if [[ ! -f "$output" ]]; then
        python scripts/scale_policy_encoder_feature.py \
            "$SOURCE_CHECKPOINT" "$step27" \
            --input-dim 32 --layout-precision-bytes 4 \
            --feature-index 27 --scale "$gain"
        python scripts/scale_policy_encoder_feature.py \
            "$step27" "$step28" \
            --input-dim 32 --layout-precision-bytes 4 \
            --feature-index 28 --scale "$gain"
        python scripts/scale_policy_encoder_feature.py \
            "$step28" "$output" \
            --input-dim 32 --layout-precision-bytes 4 \
            --feature-index 29 --scale "$gain"
    fi
    sha256sum "$output"
done
