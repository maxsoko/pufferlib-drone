# VQ2 C013 C012 measured handoff result — 2026-07-28

C012 is rejected; the alias half was not run.

- ordered Gate-2 transitions: `0/128`;
- terminal behavior: `128/128` timeouts, zero crash/miss/out-of-order event;
- final mean position: `[-20.5847, 15.6795, -0.7802] m`;
- warm/action-delivery error and action faults: `0/0`;
- FlightSim/teacher/update/Submission actions: `0/0/0/0`.

Frozen report/trace SHA-256:

- `2f04b2c333a1714b81531bc5539702961212717daa65dbd29642dda962e90640`;
- `51a4c0bfc55a3d4d1e1941bf7ea815a5daf7fe51096582c28a8433bad6dbfc9c`.

C009 under-turns and reaches the gate plane near `y=2.4`; C012 over-corrects,
reverses through negative X, and finishes near `y=15.7`. This authorizes one
small whole-checkpoint interpolation bracket at alphas `0.25/0.50/0.75`.
There is no runtime action blend. If none completes Gate 2, end interpolation.
