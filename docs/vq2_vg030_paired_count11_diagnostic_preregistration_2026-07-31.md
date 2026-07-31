# VQ2 VG030 paired count-11 diagnostic preregistration — 2026-07-31

## One-run decision

Run exactly one paired teacher-free diagnostic tagged
`vq2_vg030_vg028_paired_count11_diagnostic_001`. Compare admitted VG025 epoch
5 against admitted VG028 epoch 6 on the identical fresh count-11 fixture
before spending the full 256-course screen.

VG029 already qualifies VG028 on its source-locked count-5 pair: candidate
versus parent Gate-1/Gate-2/Gate-3/Gate-4 reach was `32/25/3/1` versus
`32/13/0/0`, while crashes improved from `5` to `2`. Its admission evidence
SHA-256 is
`42ac6e99c15c326bfef6ebac4dffd65f9c32a992e3fc28c4dae8fe3b55072b6e`.

VG025 checkpoint/report/admission SHA-256 values are
`85671c31...`/`b9d22d28...`/`633a34d5...`. VG028 checkpoint/report/admission
SHA-256 values are `ec1206b5...`/`b51b7a9b...`/`3282c98f...`.

This is offline native evidence only. It authorizes no shadow, simulator
packet, bounded Training run, or Submission action.

## Paired fixture

- Run exactly 32 count-11 episodes per actor, seed `429122`, four native
  vector threads, `2,560` maximum environment steps, and one terminal episode
  per agent.
- The seed is fresh and the complete course, plant, and initial-state fixture
  is identical between the two components. Run the components concurrently.
- Each actor emits its complete deterministic recurrent four-action mean.
  Public phase, previous action, and recurrent state advance normally. Teacher
  blend/action emission, sampling, clipping, analytic fallback, and optimizer
  updates are zero.
- The official aperture remains `0.75 m`. The stricter `0.50 m` crossing
  margin is diagnostic only.
- Each component has atomic source-bound state and completed resume support.
  Its generic per-component `admitted` field is not the paired decision
  authority; only the terminal comparator report is authoritative.

Parent component manifest SHA-256:
`929f10635e663f0f2b2e88bb6be87df9198b7e6690af452b7e738bd0fe3c22a9`.
Candidate component manifest SHA-256:
`d89d641b6619b13b35762639e0c721f5e1292453460040b603229c8f63eedfa0`.

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
triples, this preregistration, runner, component, comparator, VG029 admission,
goal prompt, generic evaluator, native environment, recurrent ABI, and
public-phase code before the first component step.

Frozen new source surface:

- `scripts/eval_vq2_staged_count5_component.py` — SHA-256
  `925ee589351318ffa8d93449c9fc5a5f2f184f16503fc59c71b38158a7bd3de8`
- `scripts/compare_vq2_staged_count5_diagnostic.py` — SHA-256
  `53d30a1e6a93f9432c075a075b54180a00edf22df6473e26d63a6699d706113a`
- `scripts/run_vq2_vg030_vast.sh` — SHA-256
  `8a4b19ed0aff4883d8ff6a757f66cacbc43451e114e6195e4d6422b7b87337f7`
- `tests/test_vq2_staged_count5_diagnostic.py` — SHA-256
  `b337cbea8dceb02961550bc55866caeb7810296abf60015ffdf2e87bb39dd0e1`
- `docs/vq2_vg029_paired_count5_diagnostic_admission_2026-07-31.json` —
  SHA-256
  `42ac6e99c15c326bfef6ebac4dffd65f9c32a992e3fc28c4dae8fe3b55072b6e`
- `docs/vq2_vg028_seven_source_refit_admission_2026-07-31.json` — SHA-256
  `3282c98fab562a9593261538f452226190bc51b10e5e64495957cd348403755b`
- `docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md` — SHA-256
  `03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1`

## Safety

Privileged actor inputs, teacher plant actions, optimizer updates, simulator
packets, sealed-test accesses, shadow authority, Training authority, and
Submission authority are all zero. VQ2 Submission remains forbidden without a
new explicit user instruction.
