# VQ2-VG001B float32 binding smoke preregistration — 2026-07-29

Tag: `vq2_vg001b_variable_gate_float_binding_smoke`

VG001's compiled binding distribution passed, but the subsequent collector
audit found that the extension had been built in the repository's default bf16
mode. That artifact is rejected as legal-actor ABI evidence because the VQ2
PyTorch/collector path requires float32. Preserve its report as superseded; do
not relabel or overwrite it.

VG001B changes only the build mode to `build.sh drone_race_vision --float` and
adds an explicit `precision_bytes == 4` admission gate. Repeat the exact 64
one-drone vector environments, two forced one-step timeout episodes per
environment, seed `429001`, and fixed-per-instance `5..12` assignment. Require
128 logged episodes, exactly `0.125` episode mass at each count 5 through 12,
zero mass at all other counts, zero success under the forced timeout, and the
same distribution after reset.

This is an offline binding smoke only. It sends no FlightSim packet, accesses
no sealed test, performs no student update, and cannot authorize Submission.
