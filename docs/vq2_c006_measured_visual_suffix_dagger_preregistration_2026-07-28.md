# VQ2 C006 measured visual suffix DAgger preregistration — 2026-07-28

## Purpose

Perform the one suffix-only policy-state DAgger collection authorized by C005.
SF066 remains the complete plant-action source; the native alignment oracle is
queried only for offline labels at those visited suffix states.

## Frozen inputs

- collector: `scripts/collect_vq2_measured_visual_suffix_dagger.py`
  - SHA-256 `99c6970340f33a7e3f284fb09d0610c1b1d5274fc47534cbceb6156944a882cb`
- collector tests:
  `tests/test_collect_vq2_measured_visual_suffix_dagger.py`
  - SHA-256 `08d063b4cfa9b65ed2bc443e42a332929786e95323a2d5bb84ee6608c63e286f`
- final fixture evaluator SHA-256
  `095efa1b5621e6024ab9b8772f778426c28915bfb4d71dc633ddbe90247417a3`
- C005 result SHA-256
  `b343e801f8ff1da7e2e33b731e55102487a609ba9d8e19d4ebfdfc56c9cdfff0`
- C005 true/alias report SHA-256:
  - `f8d34cd8d7f11bea624061f7dbe9c6e4e962090bc9fa8c0e176e7d0886c153c3`
  - `ce537f45509defd4a48295c5b7e68225bdff937f9aeb11487363a5668a8b4428`
- admitted SF009/SF011 oracle reports:
  - `c76c4f12045cc5782c6b24f4e5d39eee52eeb1855e77f3272526b5d48a100f33`
  - `f9b39ba627a3c4c1daf79672175ec35989e29acb8f6ae7ef5ecff212fe7c2a1e`
- oracle query source SHA-256
  `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`.

Focused tests pass `20/20`; selected Python sources compile.

## Fixed collection

Run tag `vq2_c006_measured_visual_suffix_dagger_128` exactly once with `128`
agents, seed `43006`, CUDA:

- agents `0..63`: measured true-range handoff;
- agents `64..127`: source-locked zoom `1.741775393486023` for the first `16`
  suffix steps, then true-range imagery;
- exact 208-record warm prefix and measured N399 transition;
- held `4 Hz` public progress and continuous selected-action history;
- SF066 deterministic mean owns every complete plant action;
- teacher blend exactly zero.

For each active state, save only the `4096`-value legal soft mask, `23` legal
sensor/progress/history values, one complete four-action oracle label, valid
bit, and terminal bit. Privileged native state may enter the training-only
oracle query but must never enter a saved observation or recurrent actor.

End each dataset episode on native failure or the first raw ordered `2/6`
transition. Do not continue into another gate distribution.

## Admission

Finalize the dataset only if:

- all `128` episodes contain a nonempty contiguous valid prefix;
- every episode has exactly one terminal and it is its final record;
- record count equals the sum of episode lengths;
- N294/SF066 warm replay and plant action delivery are each within `5e-5`;
- phase never decreases;
- there is no nonfinite or action-envelope fault; and
- all source hashes match.

On admission, train one suffix child from SF066 using C006 plus the retained
clean SF012 and accepted measured datasets as anchors. No second collection is
authorized unless that child yields a new source-locked failure and satisfies
the execution prompt's two-collection bound.

This collection sends zero FlightSim/Training/Submission packets and performs
zero student update.
