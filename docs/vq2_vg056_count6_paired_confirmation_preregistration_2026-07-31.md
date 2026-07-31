# VQ2 VG056 larger independent six-gate confirmation — 2026-07-31

Compare frozen VG033 and the VG055 alpha-`0.75` checkpoint on new episode
offsets `64/72/80/88`, 64 episodes per offset, four threads, 3,072 steps, and
the official `0.75 m` aperture. Both actors see the same 256 episodes.

Qualification requires exact hard transport, no Gate-1 or Gate-2 reach
regression, no crash regression, strict downstream improvement, and nonzero
Gate-5 reach beyond the parent. The last predicate directly targets the
established Gate-4-to-5 bottleneck and prevents sparse Gate-4 progress from
granting further promotion authority.

Every action is the deterministic recurrent Puffer mean over the unchanged
legal ABI. Teacher action/blend, updates, FlightSim, shadow, Training, and
Submission are forbidden. A pass remains offline-only and requires another
source-locked promotion decision; Submission requires explicit user
authorization.

Source lock before the run:

- VG055 checkpoint/report/admission: `d4076edd...`/`bb026980...`/`ae892d09...`
- evaluator/runner/test: `e354477c...`/`8b35810b...`/`e7f6c8b2...`
