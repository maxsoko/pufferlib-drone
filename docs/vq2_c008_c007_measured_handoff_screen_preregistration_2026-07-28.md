# VQ2 C008 C007 measured handoff screen preregistration — 2026-07-28

## Objective

Make the first and decisive closed-loop admission test of C007. Reuse the
admitted C005 fixture unchanged; do not fit, tune, or query a teacher.

## Frozen implementation and candidate

- evaluator: `scripts/eval_vq2_measured_visual_suffix_handoff.py`
  - corrected C008R SHA-256
    `80721eb56adf1ddfc41ae034ab7dfbf604229ccd75552e72ddbd6e36e6d3dc57`;
- focused evaluator tests:
  `tests/test_eval_vq2_measured_visual_suffix_handoff.py`
  - corrected C008R SHA-256
    `9804eb912d0423f528f1fbb2fe9bfc54a8ecdd560808dd059c5f181affeb923e`;
- focused composite suite: `20/20` passing;
- C007 checkpoint SHA-256
  `393f5de6f9dd97ae81d855b23d77c96e97926507e7c2b89611714df3d60a24c5`;
- C007 report SHA-256
  `bfc99b3d5177c5e83d9d46653480c26c66e05eebbc8d1d1bbc960bacd77c6e49`.

The evaluator warms C007 stepwise on all `208` legal measured-prefix
observations while N294 owns every index-0 action. At the measured N399
handoff, C007 owns every complete four-action vector. No blend, override,
analytic fallback, teacher action, or FlightSim packet is allowed.

## Fixed paired screen

Run each exactly once on CUDA with `128` agents:

1. `vq2_c008_c007_true_range_exact_128`, seed `43008`, alias zoom `1`, hold
   `0` steps;
2. `vq2_c008_c007_live_alias_exact_128`, seed `43009`, alias zoom
   `1.741775393486023`, hold `16` steps.

## Admission

Both reports must show:

- diagnostic valid;
- `128/128` ordered next-gate transitions;
- `0` premature failure, crash, timeout, miss, or out-of-order event;
- recurrent warm determinism and plant action-delivery maximum error
  `<=5e-5`;
- zero nonfinite/action-envelope fault; and
- zero FlightSim, teacher, update, or Submission action.

Any failure rejects C007 and ends this fit branch. Passing both authorizes the
larger exact/perturbed offline screen only; it does not authorize FlightSim.

## Invalid first execution and corrected tags

The first true-range execution completed its offline rollout but failed before
writing `report.json`: the evaluator reused `suffix_candidate` for a recurrent
state tensor and JSON serialization rejected it. Preserve
`vq2_c008_c007_true_range_exact_128` as invalid partial evidence; never
overwrite or count it. The rollout sent zero FlightSim/Submission packets and
made zero update.

The only correction renames that local recurrent-state variable. Re-run the
same fixed pair once under:

1. `vq2_c008r_c007_true_range_exact_128`;
2. `vq2_c008r_c007_live_alias_exact_128`.

All other settings and the admission predicate remain unchanged.
