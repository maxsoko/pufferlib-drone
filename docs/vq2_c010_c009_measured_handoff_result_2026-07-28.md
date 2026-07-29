# VQ2 C010 C009 measured handoff result — 2026-07-28

C009 is rejected. The true-range screen was decisive; the alias half was not
run.

- ordered Gate-2 transitions: `0/128`;
- terminal behavior: `128/128` misses, zero crash or timeout;
- terminal radial/right/vertical error:
  `6.9637566/-6.2987976/-2.9696791 m`;
- recurrent warm and plant delivery maximum errors: `0.0/0.0`;
- action-envelope/nonfinite/out-of-order faults: `0/0/0`;
- FlightSim/teacher/update/Submission actions: `0/0/0/0`.

Frozen artifacts:

- report SHA-256
  `b341b49f6053110508553e805b7b1f12c5edf4ac6e9b3635fd262b1b546296c8`;
- trace SHA-256
  `b387175e6a708cb98162ef1d8f72b241748350098443e8ee31ec62233f53b62d`.

C009 repaired the recurrent warm contract and fits C006's SF066 visitation
distribution, but its own closed-loop states leave that distribution. The
next bounded action is one genuine policy-state DAgger collection: C009 emits
every plant action from the exact warmed handoff, while the admitted alignment
oracle supplies offline labels only. Use half true-range and half fixed-alias
agents as in C006. Do not change the plant or query FlightSim.
