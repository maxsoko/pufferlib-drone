# VQ2 C004 phase-corrected measured handoff preregistration — 2026-07-28

C003R proved the fixture reset phase was incorrectly `1/2` because the native
segment declared two gates. Its invalid result is frozen at
`docs/vq2_c003r_handoff_phase_result_2026-07-28.md`, SHA-256
`e7de77552cb0d88f2ced4080679642f95415b015042d365bfec79f3aea334536`.

C004 changes only the offline phase-feature scale:

- keep `num_gates=6` so the training-only phase authority begins at `1/6`;
- explicitly set observable progress denominator `6` and the six-way one-hot;
- continue to stop at the next ordered `2/6` transition; and
- never interpret six as the official VQ2 finish gate count.

The focused native test now asserts the reset phase is exactly `1/6` for every
agent. All focused tests pass `17/17`.

Frozen corrected sources:

- evaluator SHA-256
  `47d2fbd00879c6231ed558c1c563a598a64c8e07ce933dbb1dd2a0f95475dcac`;
- tests SHA-256
  `6e04668d7e8e01b5a5b862b3445adc7ed65edf56209eb0fbf72fb4de7f504ad8`.

Run exactly once each with `128` agents, seed `43003`, CUDA:

1. `vq2_c004a_measured_handoff_true_range_exact128`: zoom `1`, hold `0`;
2. `vq2_c004b_measured_handoff_live_alias_exact128`: zoom
   `1.741775393486023`, hold `16`.

Every recurrent prefix, measured state/clock/action, plant, policy, mask,
action selector, fixture-valid bound, policy-admission criterion, and safety
condition remains exactly as frozen in C003/C003R. A valid failing fixture
authorizes one suffix-only policy-state DAgger collection from these two fixed
distributions. Another fixture correction or calibration loop is forbidden.

This remains command-free: no VQ2 Training, Submission, or FlightSim packet.
