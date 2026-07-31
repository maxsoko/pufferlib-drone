# VQ2 VG069 phase-independent early-stop replay — 2026-07-31

Run one epoch-resumable offline replay tagged
`vq2_vg069_phase_independent_early_stop_001`. VG068 is rejected: its single
aggregate-best epoch improves held phase-balanced action MSE only `1.5144x`,
misses the early-phase threshold, and regresses phases 6, 8, 10, and 11.
However, its independently parameterized heads reach their lowest held losses
at different epochs. A non-executable history-only composition predicts
`2.117763x` improvement with no regression.

Repeat VG068 exactly: same admitted VG063 features, seed `429203`, ten epochs,
chunk order, 256->64 tanh->4 head architecture, AdamW `2e-3`, weight decay
`1e-5`, gradient clip 1, and group-stratified validation agents. Retain each
phase 1--11 head at its own lowest held action MSE. Epoch zero is the frozen
exact parent and must be retained whenever every trained epoch is worse. The
entire VG068 per-epoch validation history must replay within `1e-12` maximum
absolute MSE error.

Numerical admission remains unchanged: at least `2x` phase-balanced held
improvement, phases 1--3 each at least `2x`, no phase regression, exact base
parameters, zero inactive outputs, finite weights, and trainable L2 at most
256. This selection uses only development validation; no rollout corpus or
sealed test is read. A pass authorizes one separately source-locked,
teacher-free count-5 scale diagnostic only.

Bind the pushed commit, VG033, VG063 features/report/admission, VG067 rejection,
VG068 report/rejection and original sources, the goal, nonlinear model, trainer,
runner, tests, runtime, and safety fields. Frozen new hashes are filled before
launch:

- trainer: `7b003d02cc4e13364b6d7afa490e7c2fc2bc5829a42916cde8994056ad388cda`
- runner: `ac17084e9992be20c620dd903176fc8e87d181cdf55faab883b869d85c62be00`
- tests: `6c21fd13471f8daa97c616f404674f68a8fa1a7c5806affb55b1fd68552a0e98`
- VG068 rejection: `1f8273e954f68f828306423b6e5ec567470c7abcbd65c0323d3f53af83327b18`

FlightSim, shadow, Training, Submission, teacher runtime actions, and sealed
test access are zero. VQ2 Submission remains forbidden.
