# VQ2 C010 C009 measured handoff screen preregistration — 2026-07-28

## Objective

Test whether the deployment-matched recurrent-context correction produces
closed-loop Gate-2 completion. Reuse C005 unchanged and perform no update,
teacher query, or simulator action.

## Frozen implementation and candidate

- evaluator: `scripts/eval_vq2_measured_visual_suffix_handoff.py`
  - SHA-256 `c05b344121705754768c68a8bc7ac9e99d016260b51ea45851da8ed1b086aa92`;
- evaluator tests: `tests/test_eval_vq2_measured_visual_suffix_handoff.py`
  - SHA-256 `9f3dd316d5f7302e0b8ec85b90438ab447035cc81f87f4b9fe8c769cc9cf959a`;
- focused evaluator/warmed-fit/composite suite: `25/25` passing;
- C009 checkpoint/report SHA-256:
  - `f037e56ec07ae134a87b3cf86bc9e1f3312d2310918cdf4f6fb84a9e96b259f2`;
  - `296adbfdb8c6e2c8360e872a47f604b54ac7265f7628944c4d8c3b7b426baf34`.

N294 owns every index-0 action while C009 warms stepwise on the exact `208`
legal observations. C009 owns every complete handoff action. No blend,
override, analytic fallback, or privileged actor input is allowed.

## Fixed screen and admission

Run up to this fixed pair, stopping immediately if the first is rejected:

1. `vq2_c010_c009_true_range_exact_128`, seed `43010`, alias zoom `1`, hold
   `0`;
2. `vq2_c010_c009_live_alias_exact_128`, seed `43011`, alias zoom
   `1.741775393486023`, hold `16`.

Each executed screen must have `128/128` ordered next-gate transitions, zero
failure/crash/timeout/miss/out-of-order event, recurrent warm and action
delivery error `<=5e-5`, zero action-envelope/nonfinite fault, and zero
FlightSim/teacher/update/Submission action.

Failure rejects C009 and ends the pair. Passing both authorizes only the larger
exact/perturbed offline admission; it does not authorize FlightSim.
