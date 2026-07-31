# VQ2 VG042 phase-balanced recurrent refit — 2026-07-31

## Evidence and one-run decision

Run exactly one epoch-resumable offline refit tagged
`vq2_vg042_variable_gate_phase_balanced_refit_001`. Start from retained
VG033, not rejected VG040. Use fresh training seed `429156`; train four
epochs at AdamW learning rate `2e-6`, weight decay `1e-5`, eight agents per
source, 256-step BPTT, and four exposures for chunks containing a public
phase increment.

VG041 proves that whole-corpus VG039 fitting regresses the rollout frontier:
VG040 versus VG033 reaches Gate 3 in `6/32` versus `8/32` and crashes
`3` versus `1`. VG039 nevertheless contains legal later-phase labels:
training phase-0--5 row counts are
`109520/502280/886441/144061/21024/240`, and held-out validation counts are
`15520/67298/135681/14304/5020/0`.

## Fixed phase-balanced objective

Preserve every original recurrent sequence, observation, action history,
episode boundary, and hidden-state flow. Do not resample isolated rows.
Apply only an inside-source action-loss multiplier selected from the public
held phase:

`phase 0/1/2/3/4/5+ = 0.25/0.5/1/6/32/32`.

The resulting effective training shares across phases 0--5 are approximately
`1.01%/9.27%/32.71%/31.90%/24.83%/0.28%`; held-out shares across phases
0--4 are `0.92%/8.02%/32.33%/20.45%/38.28%`. The multiplier is computed
from legal observation slot 4,118 only and never enters the actor input or
runtime.

Every optimizer update independently normalizes nine sources and applies
outer weights:

- clean/VG009/recovered-VG012/VG016/VG019/VG024/VG027/VG032/VG039 =
  `0.25/0.03/0.03/0.06/0.08/0.08/0.12/0.15/0.20`.

This preserves VG033's total `0.65` anchor mass and splits its former `0.35`
latest-source mass between VG032 and phase-balanced VG039.

## Numerical admission

Select one epoch only by the fixed outer-source objective, using the
phase-balanced loss for VG039 validation and ordinary fixed validation for
all anchors. Admit only if:

- the selected objective and VG039 phase-balanced validation strictly improve
  their VG033 baselines;
- held-out VG039 phase-3 MSE does not regress and phase-4 MSE strictly
  improves;
- ordinary unweighted VG039 MSE remains at most `0.12`;
- all nine preservation caps pass:
  `0.02/0.02/0.06/0.05/0.04/0.04/0.10/0.12/0.12`;
- exact outer-source weights, `4.0x` transition exposure, finite metrics,
  source identity, and completed-state/report/checkpoint identity all pass.

Numerical admission may authorize only a separately preregistered paired
teacher-free rollout diagnostic against VG033. Failure rejects VG042 without
unchanged retry.

## Source lock and remote gate

The first source-locked preflight on commit `b4251d80...` passed both native
suites, the SM89 build, and `71/71` tests, then stopped during fixed baseline
evaluation before output creation because `_agent_batches` was referenced
through a module that does not export it. It wrote no state, checkpoint,
report, or optimizer update. The repaired source imports the helper from its
defining module, adds a direct helper-path test, and binds abort evidence
SHA-256
`a66277e69f10fca99f9d0fb702560a4c6bea168b1ce7abfc17de2e0973ec9b15`.

Before training, bind the exact pushed commit; goal; VG033 checkpoint/report/
admission; VG039 report/metadata/admission and arrays; VG041 rejection
evidence SHA-256
`a9eb3079c7e59cf52594ce89c07182356724d99a0ab545c7e96eb9f954a55dd9`;
runtime; native configuration; trainer/core/runner/tests; seed; hyperparameters;
weights; and zero-authority fields.

Source SHA-256 values:

- refit core:
  `6b0ff918012f4827113fcef70865d163357b5b51e9b7d9b9da813d97ff5a9ab7`
- nine-source configuration:
  `c959096bad1b0acc1cde810c80fc4ea5e8e7d674ebd3c3fbad0d614827fa5c44`
- VG042 trainer:
  `4959f5c2638a0911a4a7c92e6375d78baffa6ae8eeace6cec97549d11a9ba46d`
- runner:
  `d9ec5e93ff25fac0635f608f804fe2b29f134f85275e6b8da0f63bea78ec0b4b`
- existing core test:
  `065f147cd45f5b2e93d8ad28e05279cc2ac6ace08f56d2abfa6c74f221f44a0e`
- VG042 test:
  `1b81716ecd85f58ba11586e94704d35e9097abeab100c5ba7b425aaec69a55c5`

Require the retained Vast environment, CUDA, at least 32 visible CPUs, about
64 GiB RAM, 15 GiB free disk, Clang/OpenMP, ccache, both native regression
suites, a fresh SM89 float32 vision build, and the complete focused trainer
test shard.

## Safety

The actor remains one full-output recurrent PufferLib policy over the legal
4,119-value ABI. Teacher plant actions, teacher blend, privileged actor input,
checkpoint switching, analytic override, clipping, fallback, FlightSim
packets, shadow, VQ2 Training, sealed-test access, and Submission authority
are all zero. VQ2 Submission remains forbidden without explicit user
authorization.
