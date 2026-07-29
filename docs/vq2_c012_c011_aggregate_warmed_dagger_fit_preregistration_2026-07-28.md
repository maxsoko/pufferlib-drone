# VQ2 C012 C011 aggregate warmed DAgger fit preregistration — 2026-07-28

## Objective

Fit one child that covers both the original SF066 visitation distribution and
C009's actual closed-loop failure distribution, with deployment-matched warm
context on both. Do not collect more data or alter N294.

## Frozen implementation and data

- trainer: `scripts/train_vq2_c011_aggregate_warmed_dagger.py`
  - SHA-256 `46c3ce8473be9c6b501a45b11ee945818326f8b574516606d7ddd01089351cbe`;
- tests: `tests/test_train_vq2_c011_aggregate_warmed_dagger.py`
  - SHA-256 `fafe5b96f9a769cc253ac3f3dadf62d202b5add909e90b0f47382498aa7c45b2`;
- combined aggregate/warmed/collection suite: `14/14` passing;
- C009 parent checkpoint/report SHA-256:
  - `f037e56ec07ae134a87b3cf86bc9e1f3312d2310918cdf4f6fb84a9e96b259f2`;
  - `296adbfdb8c6e2c8360e872a47f604b54ac7265f7628944c4d8c3b7b426baf34`;
- C011 report/metadata SHA-256:
  - `4264a4eef0644149d54a30e4e541478968ff84a6581fe55ea53a489a9b63d0e3`;
  - `e5910896a5cfaf50d50601b5f4a4a05020dbd4ab7eecb6bf9706a180987f3fbc`;
- C006 and retained anchor hashes remain the values enforced by the trainer.

## Fixed fit

Run tag `vq2_c012_c011_aggregate_warmed_dagger_fit_001` exactly once on CUDA:

- seed `43012`;
- parent C009;
- seven intact, equally weighted 64-agent groups:
  SF049 clean, SF062 prior-crash, SF065 under-turn, warmed C006 true/alias,
  and warmed C011 true/alias;
- last eight local agents of every group held out (`56` total);
- prepend the same exact `208` legal observations and frozen SF066 unselected
  Puffer outputs to every C006/C011 history;
- `16` epochs, learning rate `2e-5`, phase-zero/Gate-2 weights `1/2`, and the
  unchanged AdamW, gradient, chunk, action-weight, full actor, and frozen
  log-standard-deviation settings.

## Admission

Require aggregate numerical admission plus separate audits for all seven
sources. Every phase-zero/Gate-2 weighted MSE must be `<=0.02`, every
per-action MSE `<=0.05`, and every metric finite. Require zero stored actor
privilege and zero FlightSim/Submission packet.

Failure ends C012 without rollout. Passing authorizes one paired true/alias
closed-loop screen of the child and nothing live.
