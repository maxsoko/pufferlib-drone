# VQ2 C011 C009 measured visual suffix DAgger preregistration — 2026-07-28

## Objective

Collect one legal, deployment-warmed, C009-on-policy DAgger dataset. This is a
single correction for the closed-loop distribution shift proved by C010, not a
parameter sweep.

## Frozen implementation and sources

- collector: `scripts/collect_vq2_c009_measured_visual_suffix_dagger.py`
  - SHA-256 `d40f2e99ea985797bab302d86e6ea5267886800739ba23ad31c31b989aaeaf81`;
- tests: `tests/test_collect_vq2_c009_measured_visual_suffix_dagger.py`
  - SHA-256 `361c3df4145f2fc2425da05a21983ea64d933db44f08761bca5fd7a4ca9287f0`;
- focused collection/evaluator/composite suite: `21/21` passing;
- C009 checkpoint/report SHA-256:
  - `f037e56ec07ae134a87b3cf86bc9e1f3312d2310918cdf4f6fb84a9e96b259f2`;
  - `296adbfdb8c6e2c8360e872a47f604b54ac7265f7628944c4d8c3b7b426baf34`;
- C010 failed report/trace SHA-256:
  - `b341b49f6053110508553e805b7b1f12c5edf4ac6e9b3635fd262b1b546296c8`;
  - `b387175e6a708cb98162ef1d8f72b241748350098443e8ee31ec62233f53b62d`.

## Fixed collection

Run tag `vq2_c011_c009_measured_visual_suffix_dagger_128` exactly once on CUDA,
seed `43011`, with `128` agents:

- warm C009 stepwise on the exact `208` legal prefix observations;
- C009 deterministic mean emits every complete plant action;
- agents `0..63` use true-range masks;
- agents `64..127` use the fixed `1.741775393486023` zoom for the first `16`
  suffix steps;
- teacher blend remains exactly zero;
- query the admitted alignment oracle for complete four-action labels only;
- store only legal mask, sensor/action/status tail, label, valid, and terminal
  fields.

## Admission

Require one final terminal per episode, no sequence splice, exact recurrent
warm and plant action delivery (`<=5e-5`), zero phase decrease,
action-envelope/nonfinite fault, and zero teacher action, student update,
FlightSim, or Submission action. Source-lock all output arrays and metadata.

Passing authorizes one aggregate warmed-context fit from C009 using the frozen
C006 and C011 visitation distributions. It does not authorize FlightSim.
