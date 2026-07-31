# VQ2 VG045 fresh paired count-5 confirmation — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable offline confirmation tagged
`vq2_vg045_vg044_paired_count5_confirmation_001`. VG044 selected checkpoint
fraction `0.10` (SHA-256 `d7ba25a6...`) on seed `429158`: versus VG033,
Gate-1/2/3 reach changed `64/62/11 -> 64/63/13`, crashes `5 -> 4`, misses
`6 -> 3`, and mean ordered gates `2.140625 -> 2.1875`. This discovery fixture
authorizes only the independent paired confirmation below.

## Fixed confirmation contract

- Compare frozen VG033 against VG044 alpha `0.10` concurrently on exactly 64
  terminal count-5 episodes per actor, fresh seed `429159`, four native
  threads per component, and 2,560 maximum steps.
- Both actors see the identical randomized course/plant/start distribution,
  true `0.75 m` aperture, unchanged legal 4,119-value ABI, held public phase,
  clean recurrent state per episode, and deterministic mean actions.
- Teacher blend/action, sampled action, action clipping, analytic fallback,
  checkpoint switching, privileged actor input, optimizer updates, FlightSim
  packets, and sealed-test access are zero.

## Admission and failure

VG044 qualifies for a separately preregistered count-11 diagnostic only if:

- both component reports pass every hard action/history/phase/ordering/
  envelope/transport predicate;
- Gate-1 and Gate-2 reach do not regress versus VG033;
- crash count does not regress; and
- full finishes or Gate-3 reach strictly improve.

Any failure rejects VG044 without unchanged retry and retains VG033. Passing
this count-5 confirmation grants no full screen, replay, shadow, or live
authority by itself.

## Source identity and remote gate

Bind the exact pushed commit; goal; parent/candidate manifests; VG033 and
VG044 checkpoints/reports/admissions; this preregistration; runner;
component; comparator; shared summarizer; dedicated test; compiled extension;
native sources/config; seed; horizon; and zero-authority fields.

Source SHA-256 values:

- parent manifest:
  `b7ac79dacb27159208f58529130edb78e46c35d5b97f8bd88c68dad58b9fb1ee`
- candidate manifest:
  `d96ad167c14ccf341cae61dad42e47f25f269a32420f35b501a6ff79d46cff29`
- VG044 admission:
  `85f1baf2fb5c053ac9c0f908dc0170a43aae126eaf61a1e1ccf3110dc35c0fb1`
- component:
  `925ee589351318ffa8d93449c9fc5a5f2f184f16503fc59c71b38158a7bd3de8`
- comparator:
  `1265747450a4558c978ff4de009350cab8ec3f7d20c7c1b22fffcf2213c000a4`
- shared summarizer:
  `53d30a1e6a93f9432c075a075b54180a00edf22df6473e26d63a6699d706113a`
- runner:
  `862e241844c7898cd355e1ccd5966cbf327033b1ebbf15d020b0d43cac2337e4`
- dedicated test:
  `a595f29581eff093fbb6a8b9655abcbb2a786f8101a43b47b63746382b0f0ebd`

Require the retained Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 15 GiB free disk, Clang/OpenMP, ccache, both native regression
suites, a fresh SM89 float32 vision build, and the focused evaluator tests.

## Safety

This is offline native evaluation only. Shadow, VQ2 Training, and Submission
authority remain zero. VQ2 Submission remains forbidden without explicit
user authorization.
