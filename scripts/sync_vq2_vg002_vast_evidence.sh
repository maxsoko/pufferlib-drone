#!/usr/bin/env bash
# Copy immutable count reports, mutable state, and the runner log from Vast.
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 INSTANCE_ID [LOCAL_DESTINATION]" >&2
    exit 2
fi

VQ2_INSTANCE_ID="$1"
VQ2_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VQ2_LOCAL_DESTINATION="${2:-$VQ2_REPO_ROOT/logs/remote_sync/vast_$VQ2_INSTANCE_ID/vq2_vg002}"
VQ2_VASTAI_BIN="${VQ2_VASTAI_BIN:-$HOME/.local/bin/vastai}"
VQ2_REMOTE_OUTPUT="/workspace/pufferlib-drone/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg002_variable_gate_oracle_admission"

if [ ! -x "$VQ2_VASTAI_BIN" ]; then
    echo "vastai CLI is not executable: $VQ2_VASTAI_BIN" >&2
    exit 2
fi
mkdir -p "$VQ2_LOCAL_DESTINATION"

"$VQ2_VASTAI_BIN" copy \
    "C.$VQ2_INSTANCE_ID:$VQ2_REMOTE_OUTPUT/" \
    "local:$VQ2_LOCAL_DESTINATION/admission/"
"$VQ2_VASTAI_BIN" copy \
    "C.$VQ2_INSTANCE_ID:/workspace/vq2_vg002_runner.log" \
    "local:$VQ2_LOCAL_DESTINATION/vq2_vg002_runner.log"

echo "synced VG002 evidence to $VQ2_LOCAL_DESTINATION"
