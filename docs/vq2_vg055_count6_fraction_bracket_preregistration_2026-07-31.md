# VQ2 VG055 six-gate fraction bracket — 2026-07-31

Screen VG033→VG053 checkpoint fractions
`0/0.025/0.05/0.10/0.20/0.35/0.50/0.75/1.0` on new six-gate episode offsets
`32/40/48/56`, with 32 episodes per offset, four threads, 3,072 steps, and the
official `0.75 m` aperture. Every fraction sees the same 128 episodes.

Qualification requires exact hard transport, no aggregate Gate-1 or Gate-2
reach regression, no crash regression, and strict lexicographic improvement in
finishes, Gate-6/5/4/3 reach, then mean gates over the alpha-zero parent.
Selection prioritizes that downstream tuple, then Gate-2 reach, mean progress,
crashes, and the smallest fraction. Any selected policy remains offline-only
and requires a larger independent multi-offset six-gate screen.

Every action is the deterministic recurrent Puffer mean over the unchanged
legal ABI. Teacher action/blend, updates, FlightSim, shadow, Training, and
Submission are forbidden. Submission requires explicit user authorization.

Source lock before the run:

- parent/update checkpoints: `56a8e3b8...`/`f0a9812f...`
- VG054 rejection: `bb78f928...`
- evaluator/runner/test: `0a876b34...`/`8998f64e...`/`625930c2...`
