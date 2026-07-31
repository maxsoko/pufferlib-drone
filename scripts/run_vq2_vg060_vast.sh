#!/usr/bin/env bash
# Source-locked Vast bootstrap and resumable VG060 direct indexed-head fit.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG060 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg060_direct_phase4_indexed_residual_001}"
VQ2_PARENT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg033_variable_gate_eight_source_refit_001"
VQ2_DATASET="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg039_variable_gate_dagger_round9_vg033_visited_512"
VQ2_PARENT_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
VQ2_DATASET_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg039_variable_gate_dagger_round9_admission_2026-07-31.json"
VQ2_REJECTION="$VQ2_WORKSPACE_PATH/docs/vq2_vg059_phase4_indexed_residual_bracket_rejection_2026-07-31.json"
VQ2_GOAL="$VQ2_WORKSPACE_PATH/docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"

run_training() {
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/train_vq2_vg060_direct_phase4_indexed_residual.py \
            --output "$VQ2_OUTPUT" --device cuda "$@"
}

cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
test -x .venv/bin/python
source .venv/bin/activate

if [ -f "$VQ2_OUTPUT/training_state.pt" ]; then
    run_training --resume
    exit $?
fi

for command_name in git nvidia-smi python; do
    command -v "$command_name" >/dev/null
done
test "$(sha256sum "$VQ2_PARENT/policy_best.pt" | cut -d" " -f1)" = \
    56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26
test "$(sha256sum "$VQ2_PARENT/report.json" | cut -d" " -f1)" = \
    033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60
test "$(sha256sum "$VQ2_PARENT_ADMISSION" | cut -d" " -f1)" = \
    dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44
test "$(sha256sum "$VQ2_DATASET/report.json" | cut -d" " -f1)" = \
    a0bdcd0d60f66f7de55323a08b3c4695a0333b5bf88a93454d56d75e61eb21e1
test "$(sha256sum "$VQ2_DATASET/metadata.json" | cut -d" " -f1)" = \
    89134752057b0dd7759031aaf524a93f0a85096bd1b8973ed6daebb5715c74ac
test "$(sha256sum "$VQ2_DATASET_ADMISSION" | cut -d" " -f1)" = \
    e01eb2e28c65f827cfb39916c28d56320c57a4c23513ee61553447bd236b11cc
test "$(sha256sum "$VQ2_REJECTION" | cut -d" " -f1)" = \
    a181066247c73ab7222323a945a0c777c6ba72f3d25ab2992e0255f7755e858b
test "$(sha256sum "$VQ2_GOAL" | cut -d" " -f1)" = \
    03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1
for name in mask.npy tail.npy action.npy terminal.npy valid.npy; do
    test -f "$VQ2_DATASET/$name"
done

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) >= (8, 0)
print(torch.cuda.get_device_name(0))
PY

OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_train_vq2_vg060_direct_phase4_indexed_residual.py \
    tests/test_train_vq2_vg057_warmed_phase_residual.py \
    tests/test_vq2_recurrent_phase_residual.py \
    tests/test_train_vq2_variable_gate_recurrent_bc.py

run_training
