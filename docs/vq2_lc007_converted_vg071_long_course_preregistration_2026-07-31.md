# VQ2 LC007 converted VG071 long-course baseline — 2026-07-31

Run exactly one offline teacher-free diagnostic tagged
`vq2_lc007_converted_vg071_long_course_001` on fixed 20- and 24-gate native
courses. Use 32 agents/episodes per count, 32 native threads, CUDA deterministic
mean inference, and the public 4 Hz sample-and-hold progress feature
`active_gate_index / 6` without saturation.

Convert frozen VG071 by rescaling its phase embedding so indices 0 through 16
retain their actions and recurrent transitions, copying its existing residual
heads, zeroing new output heads 17 through 32, and retaining an exact-zero
residual beyond 32. Every plant action must be the complete recurrent Puffer
output. Teacher blend, teacher rollout, spline, alignment, and action-label
reward paths are zero. The native ordered-gate index is used only as the
training fixture's mirror of public official status. No privileged pose or gate
geometry enters the actor.

This is a rejection-localization screen, not a promotion screen. Record the
maximum raw and held public indices, success/crash/miss/timeout rates, mean gates
passed, transport integrity, conversion identity, and source hashes. Use the
first weak phases to target the next gate-local teacher corpus. The converted
checkpoint has no replay, shadow, live, or Submission authority regardless of
the result. Send zero FlightSim packets and write no new checkpoint.
