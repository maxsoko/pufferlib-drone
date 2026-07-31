# VQ2 VG041 paired count-5 teacher-free diagnostic — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable same-fixture comparison tagged
`vq2_vg041_vg040_paired_count5_diagnostic_001`. Compare numerically admitted
VG040 epoch 3 against its VG033 parent. Checkpoint/report/admission SHA-256
values are:

- VG033: `56a8e3b8...`/`033a7f1f...`/`dbf0fbb4...`
- VG040: `c2a015f3...`/`0f380319...`/`53110ead...`

VG040 improves nine-source balanced validation `0.052488560618 ->
0.028067853488` and VG039 actor-visited validation `0.093892966977 ->
0.033389153715`, with all nine preservation caps, every source-weight audit,
and transition exposure passing. Numerical admission is not rollout evidence;
this fresh paired diagnostic is the next required teacher-free rung.

## Fixed paired contract

- Run 32 terminal count-5 episodes for each actor on the exact same fresh
  seed `429154`, with four native threads, 2,560 maximum steps, the true
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

VG040 qualifies for a separately preregistered count-11 diagnostic only if:

- both components pass every action, action-history, public-phase, ordering,
  envelope, finite-value, and transport predicate;
- VG040 Gate-1 reach is no worse than VG033;
- VG040 crash count is no worse than VG033;
- VG040 has nonzero Gate-3-or-finish signal; and
- VG040 strictly improves Gate-3 reach or full finishes over VG033.

A completed comparison is immutable. Failure rejects this screen path and
requires a causally distinct offline repair; it cannot be retried unchanged.

## Source identity and remote gate

Parent/candidate manifest SHA-256 values are
`51364059c7914f7b1a99231335921ae0a14d2e6fe8097c50892693ce1c873460`
and
`b2169bc6296833dc04de762cb6012ed0c47b2c24ee96e693a87b77f077f60e89`.
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
- `scripts/run_vq2_vg041_vast.sh` — SHA-256
  `190ae65592943363779892b353dc0f6226367cdf3653d7f1f20286251402cce2`
- `tests/test_vq2_staged_count5_diagnostic.py` — SHA-256
  `b337cbea8dceb02961550bc55866caeb7810296abf60015ffdf2e87bb39dd0e1`
- `tests/test_vq2_vg041_staged_count5_diagnostic.py` — SHA-256
  `832ad024640f040c697f16e4b7abf28c8e6e904ef2acbbf79255a72b0e568551`

## Safety

This is offline native evaluation only. Shadow, VQ2 Training, and Submission
authority are zero. VQ2 Submission remains forbidden without explicit user
authorization.
