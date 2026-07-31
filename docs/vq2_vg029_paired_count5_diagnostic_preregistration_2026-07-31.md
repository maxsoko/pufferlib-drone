# VQ2 VG029 paired count-5 diagnostic preregistration — 2026-07-31

## One-run decision

Run exactly one paired teacher-free diagnostic tagged
`vq2_vg029_vg028_paired_count5_diagnostic_001`. Compare admitted VG025 epoch 5
against admitted VG028 epoch 6 on the identical fresh count-5 fixture before
spending the full 256-course screen.

VG025 checkpoint/report/admission SHA-256 values are
`85671c31...`/`b9d22d28...`/`633a34d5...`. VG028 checkpoint/report/admission
SHA-256 values are `ec1206b5...`/`b51b7a9b...`/`3282c98f...`.

This is offline native evidence only. It authorizes no shadow, simulator
packet, bounded Training run, or Submission action.

## Paired fixture

- Run exactly 32 count-5 episodes per actor, seed `429121`, four native vector
  threads, `2,560` maximum environment steps, and one terminal episode per
  agent.
- The seed is fresh and the complete course, plant, and initial-state fixture
  is identical between the two components. Run the components concurrently to
  reduce wall time without changing either deterministic vector contract.
- Four threads retain the proven historical execution path. VG026 measured
  only `1.019054x` speedup at 32 threads; using four threads per concurrent
  component avoids oversubscribing the host for negligible benefit.
- Each actor emits its complete deterministic recurrent four-action mean.
  Public phase, previous action, and recurrent state advance normally. Teacher
  blend/action emission, sampling, clipping, analytic fallback, and optimizer
  updates are zero.
- The official aperture remains `0.75 m`. The stricter `0.50 m` crossing
  margin is recorded as a diagnostic, not an admission predicate.
- Each component has atomic source-bound state and completed resume support.
  Its generic per-component `admitted` field is not the paired decision
  authority; only the terminal comparator report is authoritative.

Parent component manifest SHA-256:
`2a5d5dc1f744f5f28e73812b1c95b20c789e52728c6fcbed91a696b5d646d251`.
Candidate component manifest SHA-256:
`473bc93b31060b99712bb4e1b1999224b90af7ed7b74c8a407bbef911762c3a1`.

## Promotion gate

Qualify VG028 for a separately preregistered full counts-5/8/11/12 screen only
if all of the following hold on the paired fixture:

1. Both components complete the same seed and episode count with zero
   teacher, non-finite, action-envelope, wire-rate, thrust-envelope,
   out-of-order, public-phase, action-history, or source-identity fault.
2. VG028 Gate-1 reach is no lower than VG025.
3. VG028 crash count is no higher than VG025.
4. VG028 records at least one ordered Gate-3-or-later reach or full finish.
5. VG028 strictly improves either ordered Gate-3 reach or full finishes over
   VG025.

Failure is a terminal diagnostic rejection. It forbids a full screen of VG028
and authorizes only a causally distinct offline repair loop using the new
visited-state frontier. It does not authorize an unchanged diagnostic retry.

## Remote and source gate

Require the pushed Git commit, clean tracked worktree, retained Vast
environment, CUDA, Clang/OpenMP, ccache, at least 32 CPUs, about 64 GiB RAM,
15 GiB free disk, both native regression suites, a fresh SM89 float32 native
build, and the complete focused diagnostic/evaluator test shard.

Bind the compiled extension, both manifests, checkpoint/report/admission
triples, this preregistration, runner, component, comparator, goal prompt,
generic evaluator, native environment, recurrent ABI, and public-phase code
before the first component step.

Frozen new source surface:

- `scripts/eval_vq2_staged_count5_component.py` — SHA-256
  `f6ae14ae567dd9c7073e3c0dae32bc809a9155f7ef05013bfae6f173f2d19955`
- `scripts/compare_vq2_staged_count5_diagnostic.py` — SHA-256
  `42e9a4d713ecdd3fb0423bbbf34130192827e8ac8cedc3f711cb1309e0401040`
- `scripts/run_vq2_vg029_vast.sh` — SHA-256
  `0bb86a7ebb304648629b2aa06bafd1856cdedcf6ad1e920d3560442a808ee249`
- `tests/test_vq2_staged_count5_diagnostic.py` — SHA-256
  `7048db0c8cb3d07d3cf1d55664bfd8777e0b754e077455a8c4715b2567b818e4`
- `docs/vq2_vg028_seven_source_refit_admission_2026-07-31.json` — SHA-256
  `3282c98fab562a9593261538f452226190bc51b10e5e64495957cd348403755b`
- `docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md` — SHA-256
  `03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1`

## Safety

Privileged actor inputs, teacher plant actions, optimizer updates, simulator
packets, sealed-test accesses, shadow authority, Training authority, and
Submission authority are all zero. VQ2 Submission remains forbidden without a
new explicit user instruction.
