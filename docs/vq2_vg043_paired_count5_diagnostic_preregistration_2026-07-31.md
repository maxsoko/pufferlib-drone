# VQ2 VG043 paired count-5 teacher-free diagnostic — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable same-fixture comparison tagged
`vq2_vg043_vg042_paired_count5_diagnostic_001`. Compare numerically admitted
VG042 epoch 2 against retained VG033. Checkpoint/report/admission SHA-256
values are:

- VG033: `56a8e3b8...`/`033a7f1f...`/`dbf0fbb4...`
- VG042: `876b0ecc...`/`2f4c1c9b...`/`4b18462a...`

VG042 improves fixed balanced validation `0.042006633060 ->
0.032940475596`, phase-balanced VG039 validation `0.105610140617 ->
0.062242848395`, held-out phase 3 `0.047932099463 -> 0.033795586811`,
and held-out phase 4 `0.123386528242 -> 0.088277508252`, while all nine
preservation caps pass. Numerical admission is not rollout evidence; this
fresh paired diagnostic is the next required teacher-free rung.

## Fixed paired contract

- Run 32 terminal count-5 episodes for each actor on the exact same fresh
  seed `429157`, with four native threads, 2,560 maximum steps, the true
  `0.75 m` aperture, and the unchanged randomized course/plant/start
  distribution.
- Launch both source-locked components concurrently on the same worker.
  Preserve separate state, count report, component log, and exit artifact.
- Each actor is a single full-output recurrent PufferLib policy using the
  unchanged legal 4,119-value observation, causal action history, and held
  public phase. Use only its deterministic mean.
- Teacher blend/action, sampling, clipping, analytic override, fallback,
  checkpoint switching, privileged actor input, optimizer updates, FlightSim
  packets, and sealed-test access are zero.

## Qualification

VG042 qualifies for a separately preregistered count-11 diagnostic only if:

- both components pass every action, action-history, public-phase, ordering,
  envelope, finite-value, and transport predicate;
- VG042 Gate-1 reach is no worse than VG033;
- VG042 crash count is no worse than VG033;
- VG042 has nonzero Gate-3-or-finish signal; and
- VG042 strictly improves Gate-3 reach or full finishes over VG033.

A completed comparison is immutable. Failure rejects this screen path and
requires a causally distinct offline repair; it cannot be retried unchanged.

## Source identity and remote gate

Parent/candidate manifest SHA-256 values are
`ecb6dc3601dafe13eb1007f84c5dd517ecce8672df3186ed4e7e5999c56990be`
and
`c74dd150391f9efe0f8097860659c75d2cf2572aa13b7c08d1681cb4628c6584`.
Before rollout, bind the exact pushed commit, runtime, compiled extension,
both manifests, checkpoints/reports/admissions, component, comparator,
runner, this preregistration, goal prompt, native sources/config, seeds,
course count, horizon, and zero-authority fields.

Require the retained Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 15 GiB free disk, Clang/OpenMP, ccache, both native suites, a
fresh SM89 float32 vision build, and the focused evaluator test shard.

Frozen reusable source surface:

- `scripts/eval_vq2_staged_count5_component.py` — SHA-256
  `925ee589351318ffa8d93449c9fc5a5f2f184f16503fc59c71b38158a7bd3de8`
- `scripts/compare_vq2_staged_count5_diagnostic.py` — SHA-256
  `53d30a1e6a93f9432c075a075b54180a00edf22df6473e26d63a6699d706113a`
- `scripts/run_vq2_vg043_vast.sh` — SHA-256
  `197006cdab4245181b90d67cc6789261ed66fb7ecdf9239e8373f8901d05aa6a`
- `tests/test_vq2_staged_count5_diagnostic.py` — SHA-256
  `b337cbea8dceb02961550bc55866caeb7810296abf60015ffdf2e87bb39dd0e1`
- `tests/test_vq2_vg043_staged_count5_diagnostic.py` — SHA-256
  `931631e2d7c93944ca49edbebdf1963ca77dab22faae921504be3682cbaf1e02`

## Safety

This is offline native evaluation only. Shadow, VQ2 Training, and Submission
authority are zero. VQ2 Submission remains forbidden without explicit user
authorization.
