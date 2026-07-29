# VQ2 C007 measured visual suffix DAgger fit preregistration — 2026-07-28

## Objective

Fit one recurrent visual suffix child from frozen SF066 using admitted C006 and
three retained clean/failure anchors. N294 is immutable and absent from the
optimizer.

## Frozen implementation and data

- trainer: `scripts/train_vq2_measured_visual_suffix_dagger.py`
  - SHA-256 `e5d32de410386b4eb432b031637c3b4d3fa38596e2712ed48c21a808653acaf3`
- tests: `tests/test_train_vq2_measured_visual_suffix_dagger.py`
  - SHA-256 `7fad36dca29840622b934d0e69297fc30ef5cdb65c398e5651d7aab7d93f81dc`
- C006 result SHA-256
  `d008cb27d496c78e0285f4160a786d29bd65bf611d2e7a4c7756c4688c0d0638`
- C006 report/metadata SHA-256:
  - `e3c26dc8d8b6812a56994212ee1b392260cbc24bcb112a11140547bdd7ab4f79`
  - `77c0fe74a93ebc1afd501109e5cb3080a254a5758db7db7b5b4361fc37d8768f`
- SF066 parent checkpoint/report SHA-256:
  - `f4ee6782de66736110c79a892efaa70635a2ad6c14f0bfaeeaba9812da0ae7a8`
  - `2ad76bc51417dd2160ef44ec4d4029870d3d9b189efb45d795aacd133e9e226e`
- retained SF049/SF062/SF065 report/metadata hashes remain those locked by the
  trainer and SF066 lineage.

Focused composite/collection/training tests pass `24/24`; selected Python
sources compile.

## Fixed fit

Run tag `vq2_c007_measured_visual_suffix_dagger_fit_001` exactly once on CUDA:

- seed `43007`;
- parent SF066;
- five intact 64-agent logical recurrent groups:
  `SF049 clean`, `SF062 prior crash`, `SF065 under-turn`, `C006 true range`,
  `C006 live alias`;
- no sequence splicing;
- last eight local agents from every group reserved for validation (`40`
  logical validation agents total);
- `16` epochs, learning rate `2e-5`, AdamW, existing weight decay/gradient
  clip/chunk/batch/smoothness settings;
- phase-zero loss weight `1`, Gate-2 loss weight `2`;
- train encoder, GRU, action head, phase embedding, and joint phase residual;
- freeze `log_std`; preserve the 4,119-input/four-output ABI.

Select the best epoch by the existing aggregate numerical rank. Then audit the
same held-out eight local agents separately for all five sources.

## Numerical admission

Require the ordinary recurrent numerical gate and:

- each SF049/SF062/SF065 phase-zero and Gate-2 weighted MSE `<=0.02`;
- each corresponding per-action MSE `<=0.05`;
- each C006 true/alias Gate-2 weighted MSE `<=0.02`;
- each corresponding per-action MSE `<=0.05`; and
- every metric finite.

Failure rejects C007 without a rollout screen. Passing authorizes one
source-locked C005 true/alias deterministic-mean screen of the child; it does
not authorize FlightSim.

Training executes zero FlightSim/teacher/Submission action. Every saved child
plant action must later remain a complete Puffer output.
