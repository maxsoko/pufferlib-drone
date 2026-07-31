# VQ2 LC009 long-course Puffer residual fit — 2026-07-31

Run one six-epoch offline fit tagged
`vq2_lc009_long_course_residual_fit_001` on all `3,631,517` admitted LC008
records. Convert frozen VG071 to the unsaturated `active_gate_index / 6`
contract, freeze its legal visual encoder, phase embedding, GRU, action trunk,
and log standard deviation, and train only the nonlinear recurrent Puffer
residual tensors for public indices 1 through 23. Heads 0 and 24 through 32
must retain exact-zero outputs.

The stored LC008 pre-tanh action is used only as the frozen-parent validation
baseline. Candidate predictions are recomputed from the stored recurrent hidden
state through the frozen action trunk plus the candidate's indexed nonlinear
residual, preventing the retained VG071 residual from being counted twice.

Use seed `431090`, six epochs, AdamW at `2e-3`, chunks of `65,536`, the
source-locked group validation split, and per-phase balanced loss. Require all
non-residual parameters exact, non-target outputs zero, finite L2 at most 512,
at least 2x aggregate validation improvement, at least 2x improvement at
indices 1--3, and no phase validation regression. A numerical pass authorizes
only phase-local teacher-free screening. No checkpoint from LC009 has replay,
shadow, live, or Submission authority. Send zero FlightSim packets.
