# VQ2 variable-gate solve-first goal prompt — 2026-07-28

This is the authoritative execution prompt for the next agent session. It
supersedes the fixed-six-gate assumption embedded in the SF001–SF068 and
C001–C014 lineages. Read `AGENTS.md` and the tail of `PRD.md` before acting.
Repository state at handoff is commit `508286e` ("Checkpoint July VQ1
promotion and complete VQ2 solve-first lineage"); the tree was clean.

## Mission

Produce a single recurrent PufferLib policy that completes the official
`AI-GP Virtual Qualifier R2` course — camera/IMU runtime only — with a valid,
collision-free, ordered full-course finish. The competition scores fastest
valid lap; a lap that finishes slowly beats any lap that does not finish.
Deadline: the submission window closes in approximately five days
(target: offline-admitted checkpoint within 3 days, Windows shadow and one
bounded Training attempt within 5).

## Two corrections that drive this revision

1. **The VQ2 course is not six gates.** The operator observed approximately
   **11 gates** from the start line of the VQ2 simulator UI. VQ2 publishes no
   track transfer, so the true count is unknown at runtime. Every SF-series
   training run so far used `num_gates = 6`
   (`config/drone_race_full_policy_six_gate_bootstrap.ini`). The native
   environment already supports variable counts: `DRONE_RACE_MAX_GATES` is
   `16` in `pufferlib/ocean/drone_race/drone_race.h`, and `binding.c` accepts
   `num_gates` as a kwarg (default 4, capped at 16). `num_gates` is fixed per
   environment instance, so gate-count randomization must be done across
   vectorized instances, not within one instance.
2. **The blind-gap hypothesis is dead; the problem is transition ambiguity.**
   A corrected diagnostic over the SF012 corpus (after fixing `step_dt`
   normalization, true rate 64 Hz) measured mask-empty gaps of p50 62 ms,
   p99 250 ms, max **359 ms** (~23 steps). The policy is almost never blind.
   The actual failure mode, proven repeatedly at Gate 2, is **re-targeting
   ambiguity**: after crossing a gate, multiple red apertures are visible and
   the policy must commit to the correct next one and turn hard enough.
   BPTT windows of 256 steps comfortably span every real gap; do not build
   long-horizon memory machinery for multi-second blindness.

## Verified frontier at handoff (do not re-derive)

- **Oracle (SF009/SF011)**: training-only alignment governor passes
  `4096/4096` randomized six-gate courses, mean completion `98.08 s`, worst
  crossing error `0.042 m`. Admitted for offline label generation only.
- **Legal dataset (SF012)**: 64 episodes, `403,722` causal transitions,
  actor data = `mask[4096] uint8 + legal_tail[22] float32 +
  executed_action[4] float32`, zero privileged values. The measured two-gate
  lineage later extended the ABI to **4,119** inputs (4,096 mask + 22 legal
  tail + one held 4 Hz public phase scalar).
- **SF063/SF064**: `512/512` exact Gate-1 passes, **zero crashes**, but
  `0/512` Gate 2 — a safe, consistent under-turn with mean terminal
  radial/right/vertical error `8.19/-8.16/-0.65 m`. This is the best
  teacher-free student behavior so far.
- **SF059/SF060**: `452/512` (88.3%) Gate 1, `0/512` Gate 2, 96.7% low
  crashes. SF057: `361/512` (70.5%) Gate 1, `0` Gate 2.
- **SF068 (exact two-gate PPO)**: update-8 checkpoint passed Gate 1
  `128/128`; deterministic Gate-2 closest range `8.54 m`. One *sampled*
  trajectory reached the Gate-2 plane at `0.054 m` radial error, but the
  deterministic mean never consolidated it, and whole-actor PPO progressively
  damaged the solved Gate-1 pathway. Conclusion on record: use a
  **phase-local update boundary** (protect Gate-1 behavior while optimizing
  the transition).
- **C014**: checkpoint interpolation/line-search is closed. Its recorded
  recommendation: optimize closed-loop return directly on the measured
  true/alias fixture with recurrent PPO, legal actor input, training-only
  critic.
- **Live anchors**: N294 passes official Gate 1 live (`3.23 s`); every
  Gate-2 continuation died live at the transition. The only live-measured
  transition evidence is the N295/N399/N483 public traces. FlightSim has been
  frozen since N522.

## Non-negotiable constraints (unchanged)

- Deployed control is **one recurrent full-output Puffer policy**: official
  JPEG → intrinsic normalization → causal soft all-red mask (`64x64`,
  `scripts/vq2_soft_red_mask.py`, SF001 contract) → CNN → one GRU → Gaussian
  actor deterministic mean → fixed CTBR wire conversion. No selected contour,
  reconstructed pose, course coordinates, gate identity, planner, classical
  blend, channel override, phase-switched checkpoint, or fallback.
- Public official race progress (the 4 Hz status scalar) is legal actor
  input; detector geometry and privileged native state are not.
- The classical teacher/oracle, privileged state, and critics are
  **training-only**. Zero teacher blend in any admission screen.
- **No FlightSim packet of any kind** until the offline gates below pass and
  a zero-command Windows shadow is separately preregistered. **VQ2 Submission
  is forbidden.**
- Evidence discipline: every run gets a unique tag, preregistration doc,
  JSON report, and SHA-256 hashes recorded in `PRD.md`/`AGENTS.md`. Rejected
  configurations are never retried unchanged. The consumed N712 sealed test
  is never reopened.

## Required strategy

### Stage 0 — environment: variable gate counts (blocking everything)

Extend the native course generator so vectorized training/screens sample
`num_gates` in `[5, 12]` per instance (uniform, with the six-gate case
retained as a regression anchor). Requirements:

- Course geometry randomization must produce ordered, reachable, non-
  overlapping gates for every count; reuse the existing randomized-course
  machinery (SF010/SF011 style) and extend, do not rewrite.
- The public phase scalar must become **count-agnostic**. Do not normalize by
  6. Preregister one encoding — raw official index divided by a fixed
  constant (16, the engine cap) is the default choice — and use it
  identically in training, screens, export, and deployment.
- Add a regression test proving observation layout, mask geometry, and legal
  tail are byte-identical across gate counts, and that only the phase scalar
  and episode length differ.
- Acceptance: oracle (SF009 governor, unchanged) finishes `>= 99%` of
  `512` randomized courses at each of counts `{5, 8, 11, 12}` at the true
  `0.75 m` aperture, zero collision. If the oracle itself fails on longer
  courses, fix course generation, not the oracle.

### Stage 1 — corpus: variable-gate legal BC dataset

Re-run the SF012 collector (`collect_vq2_oracle_bc_dataset.py` lineage) on
randomized counts `5-12`, ~`256` episodes, storing only the legal actor
fields plus executed actions. Verify the SF012 invariants: zero privileged
values persisted, exact action-history alignment, one contiguous valid
prefix per episode. Target `>= 1.5M` transitions.

### Stage 2 — student: recurrent BC with transition oversampling

Train the 4,119-input single-actor architecture (CNN + one GRU, ~256 wide as
in the SF-series) with:

- full-episode time-major BPTT, window `>= 256` steps (spans every measured
  mask gap and the 4 Hz phase-scalar hold);
- **transition oversampling**: weight the loss so windows containing an
  official index increment get `>= 3x` exposure — the measured failure is
  the post-crossing re-target, so the corpus exposure must match;
- action-history boundary exactness tests as in SF012/SF057.

Acceptance screen (teacher-free, deterministic, exact): `>= 90%` full-course
finishes on `256` held-out randomized courses across counts `{5, 8, 11, 12}`,
zero crash. If full-course fails but all failures are at one specific
transition, proceed to Stage 3 anyway with that transition as the fixture.

### Stage 3 — DAgger on visited states (the proven repair loop)

Iterate the existing DAgger machinery
(`collect_vq2_measured_two_gate_dagger*.py`,
`train_vq2_measured_two_gate_*.py` lineage, generalized to full course):
roll out the current student, query the oracle on visited states, aggregate
with all prior anchors (never replace a failure distribution — SF064's
explicit instruction), refit, rescreen. Keep source-balanced record counts
per SF060's audit finding (clean sequences must not dilute failure
distributions). Budget 3–5 rounds.

### Stage 4 — only if DAgger plateaus: phase-local recurrent PPO

Follow C014's recorded recommendation: recurrent PPO on the measured
true/alias transition fixture with legal actor input and a training-only
privileged critic. Respect SF068's finding with a **phase-local update
boundary**: anchor pre-transition behavior with a BC/KL term on clean Gate-1
data so PPO cannot erode the solved approach, and evaluate the deterministic
mean (a sampled success that the mean does not consolidate is not progress —
SF068 already proved that trap).

### Stage 5 — admission and handback

- Exact + perturbed (start/gate/plant jitter) full-course screens at the
  `0.75 m` aperture, teacher blend zero, across gate counts, zero crash.
- Composite replay parity and export parity (Linux vs. Windows controller
  Python), per the N388/N520 precedent.
- Hand back: checkpoint(s) + SHA-256, all screen reports, dataset manifests,
  and updated `PRD.md`/`AGENTS.md` sections. The operator runs the Windows
  zero-command shadow and any bounded Training attempt locally; do not
  attempt them from the remote box.

## Compute

Local hardware (RTX 3070 8 GB, 4 cores, 7.7 GB RAM) is the bottleneck, not
the method. The offline pipeline is FlightSim-independent: rent one Vast.ai
Linux instance — RTX 4090 (or A100/A6000), `>= 32` vCPU, `>= 64 GB` RAM,
`>= 100 GB` disk. Clone this repository at commit `508286e`; regenerate
datasets remotely rather than uploading `logs/` (they are gitignored and
reproducible from seeds). Use the CPU cores for vectorized native rollouts
(the C env is CPU-bound) and the GPU for BPTT/PPO updates. Keep every run
resumable; sync reports and checkpoints back continuously.

## Do not repeat (closed branches)

Checkpoint interpolation/line search (C014); one-step supervised decoder
fits and ridge probes; whole-actor unanchored PPO (SF068); the pose-vector
detector/predictor composites (N480–N522); Dreamer/imagination training as a
prerequisite (N523–N735 stands as evidence only); any 32-input secondary-
policy trainer on the 4,119-input ABI (SF064's explicit warning); blind-gap
long-memory architectures (measurement closed it); guessing the live gate
count.

## Reporting

Append every experiment to `PRD.md` with its tag, seeds, hashes, and verdict
the moment it completes. If a stage acceptance fails twice with distinct
causes, stop and write a diagnosis section instead of a third variation —
the deadline does not permit unbounded retries. The single most valuable
deliverable is a checkpoint that finishes variable-count courses offline;
everything else is secondary.
