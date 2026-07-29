# VQ2 C008R C007 measured handoff result — 2026-07-28

C007 is rejected. The first valid true-range report is decisive, so the alias
half was not run.

- ordered Gate-2 transitions: `0/128`;
- terminal behavior: `128/128` misses, zero native crash or timeout;
- terminal radial error: `6.71708869934082 m`;
- recurrent warm/action-delivery maximum errors: `0.0/0.0`;
- action-envelope/nonfinite/out-of-order faults: `0/0/0`;
- FlightSim/teacher/update/Submission actions: `0/0/0/0`.

Frozen artifacts:

- report SHA-256
  `86b6772ca8265ab89eaab4526ee082a10a5c6e282d6564bcfb323bfcf41fa7a2`;
- trace SHA-256
  `eee894c0ec391b601f1e5eba0758913c820404ecf97440203b58022b356b5e6b`.

The cause is a training/deployment recurrent-context mismatch. C006 stores only
handoff-and-later observations, so C007 training and validation initialized the
GRU at zero there. Deployment instead warms it through `208` legal index-0
observations. For the first C006 record, C007 predicts from zero with absolute
action error `[0.0092, 0.1017, 0.0308, 0.0142]`; the exact warm state changes
that to `[0.2032, 0.0109, 0.4075, 0.0108]`.

Do not collect another oracle dataset. Prepend the already frozen legal prefix
and SF066's unselected whole-Puffer prefix outputs to both C006 groups, then
run one bounded warmed-context fit. This changes training context only and
matches the existing deployment evaluator exactly.
