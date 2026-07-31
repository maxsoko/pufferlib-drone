#!/usr/bin/env bash
# Source-locked Vast bootstrap and resumable VG053 trust-region refit.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG053 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg053_late_gate_trust_refit_001}"
VQ2_STATE="$VQ2_OUTPUT/training_state.pt"
VQ2_VG052="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg052_late_gate_local_dagger_vg033_visited_512"
VQ2_VG052_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg052_late_gate_local_dagger_admission_2026-07-31.json"
VQ2_PARENT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg033_variable_gate_eight_source_refit_001"
VQ2_PARENT_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
VQ2_GOAL="$VQ2_WORKSPACE_PATH/docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"

run_training() {
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/train_vq2_vg053_late_gate_trust_refit.py \
            --output "$VQ2_OUTPUT" --device cuda "$@"
}

cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
test -x .venv/bin/python
source .venv/bin/activate

if [ -f "$VQ2_STATE" ]; then
    run_training --resume
    exit $?
fi

for command_name in git nvidia-smi python; do
    command -v "$command_name" >/dev/null
done
test "$(nproc)" -ge 32
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
test "$VQ2_RAM_GIB" -ge 60
test "$VQ2_DISK_FREE_GIB" -ge 20

test "$(sha256sum "$VQ2_VG052/report.json" | cut -d" " -f1)" = \
    687141acf2ac95449772666ecda6a9deed2f59a77a23d666c17330c39070fa91
test "$(sha256sum "$VQ2_VG052/metadata.json" | cut -d" " -f1)" = \
    03b71103b956a660f32888ed8a9290c1a2251f214f736497e1d53b209ede1895
test "$(sha256sum "$VQ2_VG052_ADMISSION" | cut -d" " -f1)" = \
    95c3f6e6b9b838bf7fec40503136136098780eb5719b2971d579c72ecf63016e
test "$(sha256sum "$VQ2_PARENT/policy_best.pt" | cut -d" " -f1)" = \
    56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26
test "$(sha256sum "$VQ2_PARENT/report.json" | cut -d" " -f1)" = \
    033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60
test "$(sha256sum "$VQ2_PARENT_ADMISSION" | cut -d" " -f1)" = \
    dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44
test "$(sha256sum "$VQ2_GOAL" | cut -d" " -f1)" = \
    03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY

OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_train_vq2_vg053_late_gate_trust_refit.py \
    tests/test_train_vq2_variable_gate_nine_source_refit.py \
    tests/test_train_vq2_variable_gate_seven_source_refit.py \
    tests/test_train_vq2_variable_gate_recurrent_bc.py \
    tests/test_vq2_recurrent_phase.py

run_training
