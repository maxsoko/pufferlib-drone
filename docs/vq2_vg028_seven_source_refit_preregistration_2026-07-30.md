# VQ2 VG028 seven-source recurrent refit preregistration — 2026-07-30

## One-run decision

Run exactly one epoch-resumable CUDA refit tagged
`vq2_vg028_variable_gate_seven_source_refit_001`, seed `429120`. Start from
the numerically admitted VG025 epoch-5 actor, checkpoint/report/admission
SHA-256 values `85671c31...`/`b9d22d28...`/`633a34d5...`. Add only the
admitted VG027 long-horizon visited-state labels, report/metadata/admission
SHA-256 values `08c78d66...`/`cf1e9ba2...`/`de2437f3...`.

VG027 must first admit at least one million legal records, 95% Gate-1 reach,
20% Gate-2 reach, and nonzero phase-2 labels. A terminal VG027 rejection
forbids this refit. VG028 is a new training trajectory, not an unchanged VG026
screen retry.

## Frozen fit contract

- Six epochs, seed `429120`, 256-step recurrent chunks, deterministic CUDA,
  AdamW learning rate `5e-6`, weight decay `1e-5`, gradient cap `1.0`, and
  temporal smoothness weight `1e-4`.
- Use eight agents per source chunk. This doubles VG025's four-agent source
  batch to reduce the number of tiny sequential GPU calls while every clean
  training sequence still appears exactly once per epoch and every other
  source remains cyclic. It changes the optimizer trajectory deliberately and
  is source-locked; it does not group source calls or claim bitwise equivalence
  to VG025.
- Every optimizer update contains clean VG003, VG009, recovered VG012, VG016,
  VG019, VG024, and VG027 chunks. Normalize each source loss independently,
  then apply exact weights `0.25/0.04/0.04/0.08/0.12/0.17/0.30`. Raw corpus
  size never sets source weight.
- Repeat every multi-source chunk containing any official phase transition
  exactly four times. Preserve the unchanged 4,119-value legal actor ABI,
  recurrent action-history semantics, action weights `1/1/4/1`, and VG025
  model architecture.
- Save a complete atomic state after each epoch: model, Adam moments, scaler,
  CPU/CUDA/Python/NumPy RNG, split indices, history, best weights, source
  hashes, runtime, and completed epoch. Resume only exact identity before a
  terminal result.

## Numerical admission

Choose the lowest fixed-weight seven-source validation epoch, not the final
epoch automatically. Admit only if it improves both the source-balanced
baseline and VG027 validation, has finite metrics, preserves clean/VG009/
VG012/VG016/VG019/VG024 weighted MSE at or below
`0.02/0.02/0.06/0.05/0.04/0.04`, keeps VG027 at or below `0.10`, proves exact
source-weight sums for every epoch, and proves at least `4.0x` transition
exposure.

Numerical admission authorizes only a separately preregistered staged
teacher-free screen. Screen a small fresh count-5 diagnostic first; only a
meaningful safe downstream improvement may spend the full 256-course screen.
Neither validation MSE nor a diagnostic rung authorizes live activity.

## Source identity and remote gate

Bind the pushed Git commit, runtime, compiled extension, all seven datasets and
admission evidence, VG025 parent checkpoint/report/admission, trainer, runner,
tests, this preregistration, goal prompt, observation/recurrent/loss/loaders,
and fixed config before the first optimizer step.

The authoritative sprint goal prompt is
`docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md`, SHA-256
`03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1`.

Require the preserved Vast workspace, CUDA, at least 32 visible CPUs, about 64
GiB RAM, at least 15 GiB free, Clang/OpenMP, ccache, a fresh SM89 float32 native
build, both native regression suites, and the complete focused training shard.

Frozen new source surface:

- `scripts/train_vq2_variable_gate_seven_source_refit.py` — SHA-256
  `ceeddef89c14698831659ae137a623ccf76beeaa479e55a754a06d5d6657717d`
- `scripts/run_vq2_vg028_vast.sh` — SHA-256
  `3584a80cc651797086d4b9cc36297ada73ad1c51886ae4ca10670bf088d45532`
- `tests/test_train_vq2_variable_gate_seven_source_refit.py` — SHA-256
  `77ae61f044506cdc9f13afa0ebd915406f64bc3ddb7e6f47982f0e9009f0dc52`

## Safety

The oracle actions are fixed dataset labels. Teacher blend, FlightSim packets,
sealed N712 access, shadow, VQ2 Training, and Submission are zero. A rejected
VG028 result cannot be retried unchanged.
