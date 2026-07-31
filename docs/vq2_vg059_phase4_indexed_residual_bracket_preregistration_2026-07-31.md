# VQ2 VG059 phase-4-indexed learned residual bracket — 2026-07-31

Map VG057's learned shared residual into an indexed Puffer residual table. All
17 heads initialize to zero; public phase index 4 alone receives
`(4/16) * VG057.phase_action_residual.weight`. At scale 1 this exactly matches
VG057's learned correction when phase 4 is active, while phases 0--3 and 5--16
remain exactly VG033. No analytic action is introduced.

Screen physical residual scales `0/0.1/0.25/0.5/0.75/1/1.5/2/3/4` on fresh
six-gate offsets `128/136/144/152`, 64 episodes per offset, four threads, and
3,072 steps. Every scale sees the same 256 episodes. Qualification requires
hard transport, exactly equal Gate-1--4 reach, no crash regression, and new
Gate-5 reach. Selection prioritizes finishes/Gate-6/Gate-5/mean, then crashes
and the smallest scale.

Every action is the deterministic complete output of one recurrent Puffer
policy using only the legal observation and public phase. Teacher action/blend,
updates, FlightSim, shadow, Training, and Submission are forbidden. A pass is
offline-only pending an independent confirmation. Submission requires explicit
user authorization.

Source lock before the run:

- VG057 checkpoint/admission and VG058 rejection: `cf3825ed...`/`24edfd5c...`/`3ffd88cf...`
- indexed actor/wrapper/runner/focused test/actor test: `f66116b6...`/`6ef13f27...`/`3ca3b058...`/`a2e9c547...`/`190b0919...`
