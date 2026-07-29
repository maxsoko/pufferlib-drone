# VQ2-SF034 fresh teacher-free SF033 screen — 2026-07-28

Tag: `vq2_sf034_recurrent_roll_priority_teacher_free_512`

SF033 is the first compact recurrent successor after SF028 to pass every
frozen SF012/SF021/SF030 numerical gate. Its selected epoch-4 checkpoint and
report SHA-256 values are
`91088ca96f432f57f1b85bbc521ddab29ba422fadf2b79b8d642ee426f09bea7` /
`f54bf8512490a4629c08bf35ceae7c7fe07514e3424f89433f680536d96a0279`.

Run exactly one fresh deterministic native screen with seed `42034`, `512`
full-start randomized six-gate episodes, the true `0.75 m` aperture, the
existing `180 s` episode bound, and deterministic recurrent actor means. Set
every teacher/blend/reward-label path to zero. The actor sees exactly the
4,118 legal values and emits all four plant actions; native privilege is
excluded from its input and recurrent state.

Record ordered completion, gate counts, collision classes, misses,
out-of-order events, timeouts, action/rate/thrust envelopes, finite actions,
and inference throughput. A valid solve requires `512/512` six-gate finishes
with zero crash or other invalid terminal. Any lesser result is diagnostic
only and authorizes at most a new DAgger collection on the source-locked SF034
state distribution.

This is native/offline only: send zero FlightSim packets, do not access N712,
and do not shadow, reset, arm, setpoint, run a bounded attempt, or select
Submission.

