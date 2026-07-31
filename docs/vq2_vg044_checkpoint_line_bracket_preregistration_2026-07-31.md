# VQ2 VG044 checkpoint update-fraction bracket — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable offline bracket tagged
`vq2_vg044_vg033_to_vg042_checkpoint_line_bracket_001`. VG043 proves that
the full VG042 update regresses VG033 from Gate-1/2/3 reach `32/31/8` to
`32/28/2` and crashes `1 -> 2`, even though its later-phase label metrics
improve. Treat this as an update-magnitude hypothesis, not authorization for
more gradient fitting.

Linearly interpolate every floating actor tensor in float64 and cast it back
to the exact endpoint dtype:

`state(alpha) = VG033 + alpha * (VG042 - VG033)`.

Nonfloating endpoint tensors must be identical. Screen fixed alphas
`0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50`; alpha zero is the paired
VG033 authority and may not be selected as a new candidate.

## Fixed rollout contract

- Evaluate 64 terminal count-5 episodes per alpha on the identical fresh seed
  `429158`, four native threads, 2,560 maximum steps, the true `0.75 m`
  aperture, and unchanged randomized course/plant/start distribution.
- Run candidates sequentially in increasing-alpha order. Hash every
  interpolated tensor state and immutable count report.
- Every action is the deterministic mean of one full-output recurrent
  PufferLib actor over the unchanged legal 4,119-value ABI and held public
  phase. Hidden state starts clean per episode.
- Teacher blend/action, action sampling, clipping, analytic fallback,
  checkpoint switching inside an episode, privileged actor input, optimizer
  updates, FlightSim packets, and sealed-test access are zero.

## Selection and materialization

An alpha qualifies only if, versus alpha zero on the same fixture:

- both count reports pass every hard action/history/phase/ordering/envelope/
  transport predicate;
- Gate-1 and Gate-2 reach do not regress;
- crash count does not regress; and
- full finishes or Gate-3 reach strictly improve.

Among qualified alphas, select lexicographically by full finishes, Gate-4
reach, Gate-3 reach, Gate-2 reach, mean ordered gates, fewer crashes, then
smaller alpha. Materialize exactly that interpolated recurrent actor as
`policy_selected.pt`; do not emit a checkpoint when no alpha qualifies.

This bracket is discovery evidence only. A selected checkpoint may authorize
one separately preregistered fresh-seed paired confirmation against VG033.
Failure rejects the line direction without unchanged retry.

## Source identity and remote gate

Bind the exact pushed commit; goal; VG033 checkpoint/report/admission; VG042
checkpoint/report/admission; VG043 rejection SHA-256
`f396c3774b4bbf059ef79285481ed4a93da0be0c643d68a004312650fba2e87c`;
compiled extension; native sources/config; evaluator; comparator; this
preregistration; runner; test; alphas; seed; horizon; and zero-authority
fields.

Source SHA-256 values:

- bracket evaluator:
  `e1429cd95cf147c56fbc294993394c1a88c399d7e1b40c0b212b646d361503cc`
- runner:
  `71866d225a2af10d59d21cc754cef0788c704982b33671dc854e3bbe067910c7`
- dedicated test:
  `5cc5634725e6f34107fa1b09dcd0cbca5171248c6d81ea0a368f88e57fb1308c`
- shared recurrent evaluator:
  `dc91107e23d1039168b60bf6790db9620aedc0f4e6f66629cf142d1c4766cd31`
- shared paired summarizer:
  `53d30a1e6a93f9432c075a075b54180a00edf22df6473e26d63a6699d706113a`

Require the retained Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 15 GiB free disk, Clang/OpenMP, ccache, both native regression
suites, a fresh SM89 float32 vision build, and the focused evaluator tests.

## Safety

This is offline native evaluation only. The selected artifact, if any, is
still one recurrent PufferLib policy. Shadow, VQ2 Training, and Submission
authority are zero. VQ2 Submission remains forbidden without explicit user
authorization.
