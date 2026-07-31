# VQ2 VG058 learned residual-scale bracket — 2026-07-31

Screen VG057 residual scales
`0/0.01/0.025/0.05/0.10/0.20/0.35/0.50/0.75/1.0` on fresh six-gate episode
offsets `96/104/112/120`, 32 episodes per offset, four threads, and 3,072
steps. Every scale sees the same 128 episodes. Alpha zero is VG033 represented
exactly by the residual actor with a zero residual; all other frozen weights
are identical.

Qualification requires exact hard transport, no Gate-1/Gate-2 reach or crash
regression, strict downstream improvement, and Gate-5 reach above the parent.
Selection prioritizes finishes and Gate-6/5/4/3 reach, then mean progress,
crashes, and the smallest scale. A selection remains offline-only pending a
larger independent screen.

Every action is the deterministic full output of one recurrent Puffer policy
over the unchanged legal ABI. Teacher action/blend, student updates, FlightSim,
shadow, Training, and Submission are forbidden. Submission requires explicit
user authorization.

Source lock before the run:

- VG057 checkpoint/report/admission: `cf3825ed...`/`843f44a9...`/`24edfd5c...`
- generic bracket/VG058 wrapper/runner/test: `4b27e3cc...`/`4772eff8...`/`8acbc4a8...`/`eb56a779...`
