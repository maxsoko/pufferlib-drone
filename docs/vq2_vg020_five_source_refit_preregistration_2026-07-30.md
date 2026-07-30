# VQ2 VG020 five-source recurrent refit preregistration — 2026-07-30

## Candidate and one-run decision

Run one epoch-resumable offline fit tagged
`vq2_vg020_variable_gate_five_source_refit_001`, seed `429094`, starting from
admitted VG017 epoch 11. Parent checkpoint/report/admission SHA-256 values are
`6769476f309beb4bb502c64a83c4f992b47da2b1a8d8b6913398b4b19b3ab4a8`,
`476c730649c0f127feb4975fe2ad44f66be64b0ea8dde4d48c5ada7bb0d4c86b`,
and `df664cbac28ee676cbd5e056cb1f06745bf3e622710b3faa085eca90425741b6`.

Use only the admitted VG003 clean, VG009, recovered VG012, VG016, and VG019
datasets. VG019 report/metadata/admission SHA-256 values are
`2079ce9de0ff96bab8e218ba249307988799a7c7062d7175185d86608c213e82`,
`773273fc221e472d57d71a0407bdd3c579713aa91aab309a033c286c22541f22`,
and `9b110d50fb4f36c750725eb0f2f77afc8506d6eaeab739577b10e02aa715b286`.
VG007/VG008 and failed collector states remain quarantined.

## Fixed optimization contract

- Train 12 complete epochs with AdamW learning rate `1e-5`, weight decay
  `1e-5`, gradient-norm cap `1.0`, and action smoothness weight `1e-4`.
- Every optimizer step pairs one independent recurrent chunk from all five
  sources. Normalize each source loss by its own valid rows, then use exact
  clean/VG009/VG012/VG016/VG019 weights `0.35/0.10/0.10/0.15/0.30`.
  Raw record counts never set objective weights.
- Use four agents per source, 256-step time-major BPTT, and four optimizer
  exposures for every paired window containing any public-phase transition.
  This increases transition pressure relative to VG017 while halving the
  learning rate to protect the solved Gate-1 behavior.
- Reserve the final 32 VG003 agents and final 64 agents from each DAgger source
  for disjoint validation. The clean stream defines epoch length; all DAgger
  streams cycle independently.
- Select the epoch with minimum fixed-weight five-source validation score,
  including unchanged VG017 as epoch zero. Do not select on a teacher-free
  rollout.

## Numerical admission

VG020 is numerically admitted only when all of these hold after all 12 epochs:

- a child epoch strictly improves the fixed-weight five-source score over
  VG017;
- selected VG019 weighted MSE strictly improves over VG017;
- selected weighted MSE is at most `0.02` on VG003 and VG009 and at most
  `0.10` on recovered VG012, VG016, and VG019;
- every epoch has exact source-weight accounting and minimum `4.0x`
  transition-window exposure; and
- checkpoint/state/report source, parent, dataset, runtime, and RNG identities
  form a complete resumable hash chain.

Numerical admission authorizes only a fresh, separately preregistered
teacher-free screen. It is not policy or live-flight admission.

## Policy and legality boundary

The model remains the single 4,119-input recurrent full-output Puffer actor:
legal soft-mask camera history, IMU/actuator/action/timing history, and one
causal held 4 Hz official progress scalar feed the unchanged CNN/GRU/Gaussian
actor. Native state, total gate count, oracle values, selected geometry,
planner output, analytic action blending, clipping, switching, and fallback
remain absent from actor input and output.

Oracle actions are already-frozen training labels. VG020 emits no plant action,
teacher blend, simulator packet, shadow packet, student rollout, N712 access,
Training action, or Submission action.

## Source identity and resume

Before the first optimizer update, state locks the Git commit, trainer, runner,
this preregistration, goal prompt, all five dataset reports/metadata/admissions,
parent checkpoint/report/admission, model/training helpers, runtime, config,
splits, phase audits, initial validation, optimizer/scaler/RNG state, and safety
fields. Each completed epoch atomically replaces state. Resume is allowed only
from exact identity; completed output is read-only and must reproduce its
checkpoint/report/state chain.

Remote bootstrap requires the exact pushed commit, clean tracked tree,
source-locked virtual environment, supported CUDA GPU, 32 visible CPUs, about
64 GiB RAM, and 5 GiB free after all source datasets are staged. It rebuilds a
fresh float32 native binding and requires both native suites plus focused and
adjacent tests before epoch-zero state.

## Frozen source surface

- `scripts/train_vq2_variable_gate_five_source_refit.py` —
  `3421998f808158cdbeb3e2709ed02f745047da61e39ea5ea4bddc9ec19fe7dc5`
- `scripts/run_vq2_vg020_vast.sh` —
  `b44c55722a2e417bddc4c0a195d5b6e40c755385d07e9f7adf439cfbcec429b2`
- `tests/test_train_vq2_variable_gate_five_source_refit.py` —
  `eaab4e5ecc1a53f016382cf5f345637a7851101112644fef8341d0a98e2f859c`
- `scripts/train_vq2_variable_gate_four_source_refit.py` —
  `aeb629cd30a524f941e677214c31983c25874ae4d5afa7b21d555304b4f3fc93`
- `scripts/train_vq2_variable_gate_three_source_refit.py` —
  `4795040a359b9972acae93c98cd5126ec3504c137836aabdb1722bbc3cb3bf7a`
- `scripts/train_vq2_variable_gate_dagger_refit.py` —
  `afc7ea57da09365c3f6f98ef3f2faaf520dc16c685548645be96f743a08d00fb`
- `scripts/train_vq2_variable_gate_recurrent_bc.py` —
  `a3e4fb3e6b553d886044fed04a2e132b9738a3fd2eb6ca34ad3a27fab634184f`
- `scripts/train_vq2_recurrent_bc.py` —
  `a024a6fc387fbf734194d9505a9ed23e2b649b01f1ed9fe596ebc9a317b97583`

The source identity also binds the unchanged actor/observation modules, goal
prompt SHA `052ba7cb...`, exact earlier dataset evidence, and every parent/input
artifact before epoch zero.

## Next authority

If numerically admitted, preregister one new teacher-free 256-course screen on
fresh seeds. If rejected, preserve VG020 and diagnose its per-source validation
tradeoff before a causally distinct repair. FlightSim, Windows shadow, bounded
Training, and Submission remain forbidden.
