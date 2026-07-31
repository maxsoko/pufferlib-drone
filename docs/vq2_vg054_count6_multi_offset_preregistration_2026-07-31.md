# VQ2 VG054 VG053 six-gate multi-offset screen — 2026-07-31

Run VG053 exactly once on the fixed six-gate native proxy at episode offsets
`0/8/16/24`, 64 episodes per offset, four threads, 3,072 maximum steps, and
the official `0.75 m` aperture. Use the deterministic recurrent Puffer mean
over the unchanged legal 4,119-value ABI. Teacher action/blend and runtime
native state are forbidden.

Compare the 256-episode aggregate to frozen VG051/VG033. Qualification requires
all hard transport predicates, no Gate-1 or Gate-2 reach regression, no crash
regression from `17`, and strict lexicographic improvement in finishes,
Gate-6/5/4/3 reach, then mean gates. A full-update rejection authorizes only a
separately preregistered VG033->VG053 checkpoint-fraction bracket. A pass
authorizes only a larger independent offline six-gate admission screen.

VG054 is offline-only. Replay, shadow, FlightSim Training, and Submission remain
unauthorized; Submission requires explicit user authorization.

Source lock before the run:

- VG053 checkpoint/report/admission: `f0a9812f...`/`ec8b81b3...`/`3838729e...`
- manifest/evaluator/runner: `001e385a...`/`c0fb4710...`/`8a68ff33...`
- generic/focused tests: `c1cf8ffb...`/`952754b8...`
