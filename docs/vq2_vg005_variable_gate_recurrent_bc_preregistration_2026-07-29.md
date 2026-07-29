# VQ2 VG005 variable-gate recurrent BC preregistration — 2026-07-29

## Rejection parent and authorized correction

VG004 is rejected and must never resume. Its source-locked epoch-zero state
was written, then the first epoch stopped when the phase monotonicity audit saw
the required zero-filled rows after a shorter episode's valid prefix and
mistook that invalid padding for an in-episode decrease. State/log SHA-256
values are `a078b44127741c9ba9cf40a2665f2e0738d35ff25d5c2dab4a667a6853a4e7b8`
and `9adaabf1383e37ca8693b17c21b1d6980ad0f4dfe32608fed95fe39719884f91`.
The sealed state proves completed epoch and persisted optimizer-update counts
both zero, though some first-epoch updates may have existed only in the failed
process memory.

Run exactly one corrected fit tagged
`vq2_vg005_variable_gate_recurrent_bc_001`. The sole semantic repair is that
phase-decrease rejection applies only where the current row is valid; invalid
zero padding remains excluded from both increment and decrease counts. A new
regression supplies phase `[0, 1/16, 0]` with valid `[1, 1, 0]`, requires one
increment, and proves no false decrease. Real valid-row phase decreases still
fail closed. No VG004 state, optimizer, model, RNG, or output is reused.

## Immutable data, actor, and split

- VG003 report SHA-256:
  `b8069d05cec63038659893a4020fa61ca1276ae035bc1a2ea5754f75e2298e85`.
  Metadata SHA-256:
  `7a5d6359641c3e01f838e6fcdb57d32ddc18cc71ddbf1f8b3d0025309cabfe64`.
- Dataset: 2,396,907 legal/public causal rows from 256/256 successful
  randomized oracle courses, exact-uniform counts 5--12, zero collision,
  exact executed-action history alignment, and zero persisted privileged or
  total-count values.
- Seed `429031`. Actor is one `VQ2PhaseRecurrentActor`: frozen-shape 64x64 mask
  CNN/legal-sensor encoder, one phase embedding, one 256-wide GRU, and one
  joint four-output tanh Gaussian head. Input is exactly 4,119 values; the
  only progress input is causal 4 Hz held `active_gate_index / 16`.
- Agents 0--223 train and 224--255 validate, giving respectively 28 and four
  episodes at each count 5 through 12. Lowest finite validation weighted MSE
  selects the checkpoint.

## Optimization and transition exposure

The full VG004 optimization contract is retained unchanged: 12 epochs, agent
batch 8, AdamW learning rate `3e-4` and weight decay `1e-5`, gradient cap 1.0,
smoothness `1e-4`, and action weights `(1, 1, 4, 1)`. PyTorch/cuDNN are
deterministic with CUDA float16 autocast and gradient scaling.

Each complete time-major episode starts from zero recurrent state and carries
that state through consecutive 256-step BPTT windows, detaching only at window
boundaries. Any batch window containing a valid causal held-phase increment is
optimized three times from its same incoming state; every affected agent-
window must therefore report exposure at least 3.0. Non-transition batch
windows receive one update. Validation sequentially replays complete held-out
prefixes and reports overall and transition-row action error.

## Resume, evidence, and safety

Output is
`logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg005_variable_gate_recurrent_bc_001`.
Before the first update, a new epoch-zero state locks the Git commit, every
source/dataset hash, runtime, config, model/optimizer/scaler, and all RNG
states. Only a state with VG005 tag/status `training` and exact identity may
resume, from its last fully completed epoch. After state exists, the Vast
wrapper must not install, test, or change dependencies. Final checkpoint,
report, and completed state are hashed and synced locally.

Teacher blend, actor privilege, FlightSim packets, sealed N712 accesses, and
Submission authorization remain zero. Training completion does not authorize
deployment. The next gate is one separately preregistered teacher-free
deterministic screen totaling 256 new randomized courses: 64 at each count
5/8/11/12, at least 90% overall completion, zero crash.

## Frozen corrected sources

- `scripts/train_vq2_variable_gate_recurrent_bc.py` — `940b9cea940b220a3cc112862b6243fc92e40d87588618464d484d48506b0618`
- `scripts/run_vq2_vg005_vast.sh` — `9988b69f668faa38be27fb677882c5e277b83efc046c2039703014b5f6b79633`
- `tests/test_train_vq2_variable_gate_recurrent_bc.py` — `8046b3ee5a8661cf5a8087c26a4a04bf071c4fbb4e84b84dd9739c28d538c67d`
- `pufferlib/vq2_informed.py` — `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`
- `pufferlib/vq2_recurrent.py` — `5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199`
- `pufferlib/vq2_recurrent_phase.py` — `a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa`
- `scripts/train_vq2_recurrent_bc.py` — `a024a6fc387fbf734194d9505a9ed23e2b649b01f1ed9fe596ebc9a317b97583`

The trainer additionally hashes this preregistration and the exact VG003
report/metadata at epoch zero. The new commit and Vast runtime are captured
before any VG005 update.
