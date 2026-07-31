# VQ2 VG061 direct phase-4 indexed residual bracket — 2026-07-31

Screen VG060 head-4 scales `0/0.1/0.25/0.5/0.75/1/1.5/2/3` on fresh
six-gate offsets `160/168/176/184`, 64 episodes per offset, four threads, and
3,072 steps. Every scale sees the same 256 episodes. The zero-scale parent and
all candidates share bit-identical base parameters and zero non-target heads.

Qualification requires every hard transport predicate, exactly equal Gate-1
through Gate-4 reach, no crash regression, and Gate-5 reach above the parent.
Selection prioritizes finishes, Gate-6/Gate-5 reach and mean progress, then
crashes and the smallest scale. A selection remains offline-only pending an
independent confirmation.

Every action is the deterministic complete output of one recurrent Puffer
policy over the legal ABI and official public progress. Teacher action/blend,
updates, FlightSim, shadow, Training, and Submission are forbidden. Submission
requires explicit user authorization.

Source lock before the run:

- VG060 checkpoint/report/admission: `32aba158...`/`68f52953...`/`13f6c301...`
- wrapper/runner/test: `290cc9be...`/`3839657d...`/`d9daa1bf...`
