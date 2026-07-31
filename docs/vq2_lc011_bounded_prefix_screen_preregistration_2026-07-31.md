# VQ2 LC011 bounded prefix screen — 2026-07-31

Run one cheap teacher-free prefix diagnostic of LC010S: 32 full-start 24-gate
episodes, seed `431110`, 32 native threads, deterministic CUDA mean actions,
and a hard `12,000`-step / `187.5 s` simulated horizon. The recurrent state
starts at zero and advances continuously; every plant action is the complete
LC010S Puffer output using held 4 Hz `active_gate_index / 6` progress.

Record mean gates passed, maximum raw/held index distribution, crash, miss,
timeout, action delivery, and phase transport. This rung localizes whether the
LC008 label fit improves the weak early-course region for roughly one-third the
wall time of a full 20/24-gate timeout. It does not require or claim a finish.
Only a transport-clean progress improvement can authorize a full-course screen.
LC011 sends zero FlightSim packets and grants no replay, shadow, live, or
Submission authority.
