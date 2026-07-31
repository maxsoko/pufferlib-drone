# VQ2 VG035 paired count-11 teacher-free diagnostic — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable same-fixture comparison tagged
`vq2_vg035_vg033_paired_count11_diagnostic_001`. Compare VG033 epoch 6
against VG028 on a fresh count-11 fixture. Actor checkpoint/report/admission
SHA-256 values remain:

- VG028: `ec1206b5...`/`b51b7a9b...`/`3282c98f...`
- VG033: `56a8e3b8...`/`033a7f1f...`/`dbf0fbb4...`

VG034 passed the count-5 rung. VG033 versus VG028 Gate-1/Gate-2/Gate-3 reach
was `32/31/8` versus `32/25/3`, mean gates `2.21875` versus `1.90625`, and
crashes `1` versus `2`. Every hard predicate passed. VG034 admission evidence
SHA-256 is `858adb9e...`.

## Fixed paired contract

- Run 32 terminal count-11 episodes per actor on the exact same fresh seed
  `429134`, four native threads, 2,560 maximum steps, true `0.75 m`
  aperture, and unchanged randomized course/plant/start distribution.
- Launch both source-locked components concurrently with separate state,
  count report, component log, and exit artifact.
- Each actor is one deterministic full-output recurrent PufferLib policy on
  the unchanged legal 4,119-value ABI with causal action history and held
  public phase.
- Teacher blend/action, sampling, clipping, analytic override, fallback,
  checkpoint switching, privileged actor input, optimizer work, FlightSim
  packets, and sealed-test access are zero.

## Qualification

VG033 qualifies for the fresh full counts-5/8/11/12 teacher-free screen only
if both components pass every hard predicate and VG033:

- preserves Gate-1 reach;
- does not increase crash count;
- has nonzero Gate-3-or-finish signal; and
- strictly improves Gate-3 reach or full finishes over VG028.

A completed comparison is immutable. Failure rejects the full-screen path and
requires causally distinct offline repair.

## Source identity and remote gate

Parent/candidate manifest SHA-256 values are
`de4dcb1f25c39e338d2ea8bfe455ab93991fb761e8c57119a2e726dfb76e922c`
and
`479cdc9c8b9af13ec07802e21a105b4b559841ae56f2364e187571ba73eebc68`.
Before rollout, bind the exact pushed commit, runtime, extension, manifests,
both actors and admissions, VG034 admission, component, comparator, runner,
this preregistration, goal prompt, native sources/config, fixture, and
zero-authority fields.

Require the retained Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 15 GiB free disk, Clang/OpenMP, ccache, both native suites, a
fresh SM89 float32 build, and the focused evaluator shard.

Frozen source surface:

- `scripts/eval_vq2_staged_count5_component.py` — SHA-256
  `925ee589351318ffa8d93449c9fc5a5f2f184f16503fc59c71b38158a7bd3de8`
- `scripts/compare_vq2_staged_count5_diagnostic.py` — SHA-256
  `53d30a1e6a93f9432c075a075b54180a00edf22df6473e26d63a6699d706113a`
- `scripts/run_vq2_vg035_vast.sh` — SHA-256
  `c9a17edf1b3edcda43ead19b1db9a003c4c2b1e434073c6e9145efb243f77f24`
- `tests/test_vq2_staged_count5_diagnostic.py` — SHA-256
  `b337cbea8dceb02961550bc55866caeb7810296abf60015ffdf2e87bb39dd0e1`
- `tests/test_vq2_vg035_staged_count11_diagnostic.py` — SHA-256
  `d4e22f003622d68420bb778646250d6fbdf37eb51797c08be82ab64d738d9ec3`

## Safety

This is offline native evaluation only. Shadow, VQ2 Training, and Submission
authority remain zero. VQ2 Submission is forbidden without explicit user
authorization.
