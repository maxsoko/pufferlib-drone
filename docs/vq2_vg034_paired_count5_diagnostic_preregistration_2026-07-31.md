# VQ2 VG034 paired count-5 teacher-free diagnostic — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable same-fixture comparison tagged
`vq2_vg034_vg033_paired_count5_diagnostic_001`. Compare VG033 epoch 6
against its VG028 parent. Checkpoint/report/admission SHA-256 values are:

- VG028: `ec1206b5...`/`b51b7a9b...`/`3282c98f...`
- VG033: `56a8e3b8...`/`033a7f1f...`/`dbf0fbb4...`

VG033 improves fixed eight-source validation `0.057872192702 ->
0.029448763167` and VG032 validation `0.122915181429 ->
0.042820791151`, with every preservation cap, source-weight audit, and
transition exposure passing. Numerical admission is not a rollout result;
this diagnostic is the next required teacher-free rung.

## Fixed paired contract

- Run 32 terminal count-5 episodes for each actor on the exact same fresh
  seed `429133`, with four native threads, 2,560 maximum steps, the true
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

VG033 qualifies for a separately preregistered count-11 diagnostic only if:

- both components pass every action, action-history, public-phase, ordering,
  envelope, finite-value, and transport predicate;
- VG033 Gate-1 reach is no worse than VG028;
- VG033 crash count is no worse than VG028;
- VG033 has nonzero Gate-3-or-finish signal; and
- VG033 strictly improves Gate-3 reach or full finishes over VG028.

A completed comparison is immutable. Failure rejects this screen path and
requires a causally distinct offline repair; it cannot be retried unchanged.

## Source identity and remote gate

Parent/candidate manifest SHA-256 values are
`601a170676f332ed5e1a78d2a6faa2bbe635afe290a8ae815980846318ced163`
and
`3dc5a6d7ca807d0e7535a38c5ea4e2d96e03e92336bd8deb53b43740d07bb9c1`.
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
- `scripts/run_vq2_vg034_vast.sh` — SHA-256
  `a941ec38ee42fce274de214303d339df5b4735bd6b0bdabda2173ec0f6a8bcae`
- `tests/test_vq2_staged_count5_diagnostic.py` — SHA-256
  `b337cbea8dceb02961550bc55866caeb7810296abf60015ffdf2e87bb39dd0e1`
- `tests/test_vq2_vg034_staged_count5_diagnostic.py` — SHA-256
  `13f33e5bf2fa8e791e3633cf4d1eeb55f63995d2b878878706fffdae13e933ab`

## Safety

This is offline native evaluation only. Shadow, VQ2 Training, and Submission
authority are zero. VQ2 Submission remains forbidden without explicit user
authorization.
