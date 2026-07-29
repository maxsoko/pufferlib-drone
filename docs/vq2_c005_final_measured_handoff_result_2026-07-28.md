# VQ2 C005 final measured handoff result — 2026-07-28

C005 admits the measured visual handoff fixture and rejects unchanged SF066 on
both fixed distributions. Proceed to one suffix-only policy-state DAgger
collection; do not revise the fixture or rerun SF066 unchanged.

Common accepted contract:

- `208` timing-matched held-index-0 warm observations;
- N294 and SF066 prefix replay maximum error `0.0`;
- exact N399 clock/state/previous-action handoff;
- SF066 owns every complete handoff action while N294 continues advancing;
- plant delivery maximum error `0.0`;
- zero action-envelope, nonfinite, or out-of-order fault;
- zero FlightSim, teacher action, student update, or Submission action.

Results:

| Screen | Gate-2 passes | Terminal behavior | Inference |
|---|---:|---|---:|
| true range | `0/128` | all fail; aggregate native `crash=1.0` from the `45 m` position bound after runaway | `52.07 steps/s` |
| live alias | `0/128` | all timeout, zero native crash | `55.01 steps/s` |

The earlier user update saying the true-range screen had zero collision/fault
was imprecise: the report is authoritative and records a late native boundary
crash. It is an offline failure state, not a FlightSim collision. Both screens
still provide valid policy-state DAgger distributions.

Frozen artifacts:

- true-range report SHA-256
  `f8d34cd8d7f11bea624061f7dbe9c6e4e962090bc9fa8c0e176e7d0886c153c3`;
- true-range trace SHA-256
  `0716f77df6d58ab17bc1916806e2a5fd6fff4f089e2225848106467943718b41`;
- live-alias report SHA-256
  `ce537f45509defd4a48295c5b7e68225bdff937f9aeb11487363a5668a8b4428`;
- live-alias trace SHA-256
  `89596c5f3d469eb5591a332353606e412f6da3d45b320e54607bc1ee3143da91`.

The authorized next action is one C006 collection with half the agents on
true-range imagery and half on the fixed 16-step live alias. The plant must
remain SF066 deterministic mean with teacher blend zero. Query the admitted
alignment oracle only for complete four-action labels at the suffix states
actually visited. Store only legal mask/sensor/progress/selected-action input,
labels, validity, and terminal flags.
