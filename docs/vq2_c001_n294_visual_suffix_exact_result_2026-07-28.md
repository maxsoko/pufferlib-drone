# VQ2 C001 N294/visual-suffix exact result — 2026-07-28

## Decision

Reject C001 as a deployment-matched training environment. Do not rerun it
unchanged and do not collect suffix labels from its transition.

C001 did validate the legal composite implementation: every one of 128 exact
episodes passed Gate 1 under N294; both recurrent policies advanced on every
public observation; the held public phase selected one complete policy action;
and the native plant received that selected action with zero numerical error.
No teacher action, student update, FlightSim packet, or Submission action
occurred.

The rejection is solely the now-measured prefix dynamics mismatch. C001 first
observed Gate 1 at step 432 (`6.750 s`) with native exit `vx=2.078475237 m/s`.
N399 passed official Gate 1 at `3.194960355 s`, and the retained legal
reconstruction uses approximately `4.677 m/s` at the transition. Training the
suffix on C001 would therefore warm it for more than twice as many recurrent
steps and hand it a materially slower vehicle.

This is not an adapter regression. The frozen pre-C001 N294 exact report
records `6.700715542 s` and `vx=2.124841690 m/s`; C001 reproduced that old
native regime closely. The old plant—not the new selector—is the rejected
deployment assumption.

## Frozen evidence

- C001 report: `logs/drone_race_full_policy_six_gate_bootstrap/vq2_c001_n294_visual_suffix_exact_128/report.json`
  - SHA-256 `997ffefaec812a22e9c61e15a232204ac7d2f454b89e9db6018a36737cee4d44`
- C001 trace: `logs/drone_race_full_policy_six_gate_bootstrap/vq2_c001_n294_visual_suffix_exact_128/trace_agent0.npz`
  - SHA-256 `2301cf881265793c2aaa9ba342d34508f6228fa501bba37aec9724cd2fd8f675`
- Frozen N294 exact report: `logs/drone_race_full_policy_six_gate_bootstrap/vq2_n294_full_blend_refine/alpha_0p60_fixed_exact512.json`
  - SHA-256 `8954c859b1fc4b21b572ea1dba2a36c73b623ae35ac9bea5b1d9d70ca843c127`

The C001 report's `harness_admitted=true` means only that its preregistered
selector and Gate-1 checks passed. It is superseded by this cross-runtime
deployment check and is not policy-training admission.

## Single next action

Calibrate one coupled training-only longitudinal plant gain against the N399
Gate-1 time and transition speed. Keep N294, the visual suffix, observation
contracts, camera model, cadence, course geometry, and every policy parameter
frozen. Suffix training remains forbidden until a separate exact composite
run matches the transition.
