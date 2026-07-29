# VQ2 C005 final measured handoff preregistration — 2026-07-28

C004's only failure was immediate native `out_of_order` from unset future gate
planes. The result is frozen at
`docs/vq2_c004_measured_handoff_result_2026-07-28.md`, SHA-256
`e8484adea1fedf3ddddf013e460c03e5c91a7e32ca576c0c99e3de4c270bde11`.

C005 explicitly parks the phase-scale-only gates 2–5 at forward positions
`60/90/120/150 m`, with fixed `y=8.70`, `z=1.095`. The measured active Gate-2
plane remains `[14.74, 8.70, 1.095] m`. The evaluator stops at its ordered
transition, so the inert gates cannot become targets or affect an admitted
action. A native focused test now proves the reset phase is `1/6` and one
zero-action step is nonterminal.

Frozen final fixture sources:

- evaluator SHA-256
  `095efa1b5621e6024ab9b8772f778426c28915bfb4d71dc633ddbe90247417a3`;
- tests SHA-256
  `ec5d358c972b342ad0cfa2b72c32fad3ca05c455a15f1bc5f12a90ca20db9bd5`;
- focused tests `17/17`, selected sources compile.

Run once each with `128` agents, seed `43003`, CUDA:

1. `vq2_c005a_measured_handoff_true_range_exact128`: zoom `1`, hold `0`;
2. `vq2_c005b_measured_handoff_live_alias_exact128`: zoom
   `1.741775393486023`, hold `16`.

Every other contract and decision remains as preregistered for C003/C004. This
is the final fixture execution. If fixture-valid and SF066 fails either policy
screen, proceed directly to the one authorized suffix-only policy-state DAgger
collection; do not revise the fixture again.

Zero FlightSim/Training/Submission packets are authorized.
