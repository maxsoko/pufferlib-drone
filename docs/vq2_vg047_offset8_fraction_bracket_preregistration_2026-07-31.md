# VQ2 VG047 offset-8 smaller-fraction bracket — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable offline bracket tagged
`vq2_vg047_vg033_to_vg042_offset8_fraction_bracket_001`. VG046 proves that
alpha `0.10` is behaviorally distinct from offset zero and consistently
improves Gate-2/3 reach, but it regresses crashes `6 -> 7`. VG044 separately
shows alpha `0.05` improves Gate-3 at offset zero without Gate-1/2 or crash
regression. Treat update magnitude as the remaining causal variable.

## Fixed rollout contract

- Linearly interpolate all floating actor tensors in float64 and cast to their
  endpoint dtype at alphas `0/0.025/0.04/0.05/0.06/0.075/0.085/0.10`.
  Nonfloating tensors must be endpoint-identical.
- Evaluate exactly 64 terminal count-5 episodes per alpha on the identical
  offset-8 fixture, declared label seed `429161`, four threads, 2,560 steps,
  true `0.75 m` aperture, and unchanged randomized course/plant/start profile.
- Every action is one deterministic full-output recurrent PufferLib mean over
  the legal 4,119-value ABI and held public phase.
- Teacher blend/action, sampled action, clipping, analytic fallback,
  optimizer updates, FlightSim packets, and sealed-test access are zero.

## Selection

An alpha other than zero qualifies only if both reports pass every hard
predicate, Gate-1 and Gate-2 reach do not regress, crashes do not regress, and
Gate-3 reach or finishes strictly improve versus alpha zero. Rank qualified
fractions by finishes, Gate-4, Gate-3, Gate-2, mean gates, fewer crashes, then
smaller alpha. Materialize only the selected recurrent actor.

A selection authorizes only a separately preregistered confirmation on a
different nonzero episode offset. No selection rejects the smaller-fraction
line without unchanged retry.

## Source identity and remote gate

Bind the exact pushed commit; goal; VG033/VG042 endpoints and admissions;
VG044 report/admission; VG046 rejection; base interpolation evaluator; offset
wrapper; runner; preregistration; dedicated tests; compiled extension; native
sources/config; offset; alphas; and zero-authority fields.

The first remote preflight on commit `5cd35dd4...` passed both native suites,
the SM89 build, and `36/36` tests, then stopped before output/state creation
because direct wrapper execution lacked the repository root on `sys.path`.
Abort SHA-256 is `a918fcb4...`. The repair only adds that bootstrap and a
direct execution test; the rollout contract remains unused.

Source SHA-256 values:

- VG046 rejection:
  `3049a97ad216a90855a2b3ebeba199782390b09df87307d2f2d09e8bebf06055`
- base interpolation evaluator:
  `e1429cd95cf147c56fbc294993394c1a88c399d7e1b40c0b212b646d361503cc`
- offset wrapper:
  `0f81b5bb6bf41ee7a2566e589bf8c0332d38a7f817f018b99079cc54272528fd`
- runner:
  `2449e2c81f60dba3dc51dd78321b0776b0599f5b9adaf533e36fa965b46bc471`
- dedicated test:
  `425528c5955e5f66aa346e613661868f8ac006c4c5c85a8bfd73050f1ebada3b`
- preflight abort:
  `a918fcb49fa1db197ed9c83bf097537aa1d177268ca07fc07cee6f2e18dd19f0`

Require the retained Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 15 GiB free disk, Clang/OpenMP, ccache, both native regression
suites, a fresh SM89 float32 vision build, and the focused evaluator tests.

## Safety

This is offline native evaluation only. Shadow, VQ2 Training, and Submission
authority remain zero. VQ2 Submission remains forbidden without explicit
user authorization.
