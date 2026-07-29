# VQ2 C014 C009-C012 interpolation bracket preregistration — 2026-07-28

## Objective

Test the smallest whole-Puffer scalar bracket supported by the opposing C009
and C012 closed-loop behaviors. This is not an action blend or a training run.

## Frozen implementation

- generator: `scripts/interpolate_vq2_phase_checkpoints.py`
  - SHA-256 `a4741142f75f03ccb87ccd524db42342377bd6719c6c791e002f2bf8f3ecafeb`;
- bracket evaluator: `scripts/eval_vq2_c009_c012_interpolation_bracket.py`
  - SHA-256 `f5e3087f74e17da6d6ff6b4d4a850631502bf59b38afdbf36adff5785eba2601`;
- bracket tests: `tests/test_eval_vq2_c009_c012_interpolation_bracket.py`
  - SHA-256 `bea3eeed10bbe50b2eed723d29ac70c86c24d715abeebeea6400ecce6268e1a6`;
- combined interpolation/evaluator suite: `20/20` passing;
- base C009 checkpoint SHA-256
  `f037e56ec07ae134a87b3cf86bc9e1f3312d2310918cdf4f6fb84a9e96b259f2`;
- endpoint C012 checkpoint SHA-256
  `2362ed2250ce41743d9153c64d7ccd3760171a468733c1b62cf34ebb2694f9f0`.

## Fixed bracket

Generate exactly three full-state-dictionary interpolants
`C009 + alpha*(C012-C009)` for alpha `0.25`, `0.50`, and `0.75`. Verify every
saved tensor by exact recomputation before evaluation.

Evaluate all three once on the unchanged true-range C005 handoff, `128` agents,
seeds `43014..43016`. Every action must be the complete deterministic output of
one interpolated recurrent Puffer checkpoint. Teacher blend and analytic
fallback remain absent.

Promotion requires `128/128` ordered next-gate completion with zero failure,
crash, timeout, miss, out-of-order, recurrent/action-delivery, envelope, or
nonfinite fault. If multiple pass, retain the lowest alpha for the subsequent
alias screen. If none pass, end the interpolation route. FlightSim remains
forbidden.
