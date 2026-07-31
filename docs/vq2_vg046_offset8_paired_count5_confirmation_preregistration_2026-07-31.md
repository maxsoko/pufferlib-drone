# VQ2 VG046 offset-8 paired count-5 confirmation — 2026-07-31

## Evidence and one-run decision

Run exactly one resumable offline confirmation tagged
`vq2_vg046_vg044_offset8_paired_count5_confirmation_001`. VG045’s formal
comparison reproduced VG044’s parent and candidate behavioral projections
bit-exactly despite changing declared seed `429158 -> 429159`. The static
vector initializes native RNG from environment index, so the training seed is
not a new plant/course fixture. VG045 is rejected as non-independent evidence.

VG046 uses the existing default-preserving native
`evaluation_episode_offset=8`. Each vector environment deterministically
executes eight skipped resets before its measured episode, advancing course,
plant, start, and visual RNG state without changing the deployed actor,
observation ABI, or ordinary offset-zero behavior.

## Fixed confirmation contract

- Compare frozen VG033 against VG044 alpha `0.10` concurrently on exactly 64
  terminal count-5 episodes per actor, four native threads per component,
  2,560 maximum steps, declared source label seed `429160`, and measured
  episode offset `8`.
- Both actors receive identical offset-8 environments, true `0.75 m`
  apertures, unchanged legal 4,119-value observations and held public phase,
  clean recurrent state per measured episode, and deterministic mean actions.
- Both full behavior projections must differ from the frozen VG044/VG045
  offset-zero hashes `d0db42b0...` and `fc25942e...`.
- Teacher blend/action, sampled action, action clipping, analytic fallback,
  checkpoint switching, privileged actor input, optimizer updates, FlightSim
  packets, and sealed-test access are zero.

## Admission and failure

VG044 qualifies for a separately preregistered count-11 diagnostic only if:

- both reports prove applied offset `8`, are behaviorally distinct from
  offset zero, and pass every hard action/history/phase/ordering/envelope/
  transport predicate;
- Gate-1 and Gate-2 reach do not regress versus VG033;
- crash count does not regress; and
- full finishes or Gate-3 reach strictly improve.

Failure rejects VG044 without unchanged retry and retains VG033.

## Source identity and remote gate

Bind the exact pushed commit; goal; VG045 rejection; parent/candidate
manifests; VG033 and VG044 checkpoints/reports/admissions; preregistration;
runner; offset-aware component; comparator; shared summarizer; dedicated test;
compiled extension; native sources/config; fixture; and zero-authority fields.

Source SHA-256 values:

- VG045 rejection:
  `214c81c1a4987a45fb59df025a31d26f5a06cd0b09748f93670c56e4df32167d`
- parent/candidate manifests:
  `55ff77aac4aadd2d6540e7b582e1212ddf96b940f553a576ca78fa508e5908f3` /
  `08162ff4dd61c8ba58698eeea5e446a46f4c82290cb3505463a496b358a23902`
- offset-aware component:
  `85f7e2a5c813d816dc2aa974e2df07429a1b525181d2d45b7740772511ec35e1`
- comparator:
  `b5bd643dc8f284b100b1f2b2d0e7798ac405886b64cd6c57958501db88c0ee86`
- runner:
  `d9d092b7681d2f1a8dde1538947881c769e9eeee7d4f26440f9a96ab059491d5`
- dedicated test:
  `ca0649fb9a1acf15fa24beff27d75d19653181f02ff713938763ebedae33e9df`
- shared summarizer:
  `53d30a1e6a93f9432c075a075b54180a00edf22df6473e26d63a6699d706113a`

Require the retained Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 15 GiB free disk, Clang/OpenMP, ccache, both native regression
suites, a fresh SM89 float32 vision build, and the focused evaluator tests.

## Safety

This is offline native evaluation only. Shadow, VQ2 Training, and Submission
authority remain zero. VQ2 Submission remains forbidden without explicit
user authorization.
