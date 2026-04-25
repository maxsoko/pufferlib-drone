# Use-More-of-PufferLib Checklist (Mapped to Exact Edits)

## 1) Throughput/GPU baseline first (highest ROI)
- Edit `pufferlib/config/drone_race.ini`
- Edit `pufferlib/config/drone_race_curriculum.ini`
- Edit `pufferlib/config/drone_race_curriculum_tier1.ini`
- Edit `pufferlib/config/drone_race_curriculum_tier2.ini`
- Edit `pufferlib/config/drone_race_curriculum_tier3.ini`
- Set:
  - `[vec] backend = Multiprocessing`
  - `[vec] num_envs = 8` (or 4 initially)
  - `[vec] num_workers = 8` (or 4)
  - `[train] num_envs = 8`
  - `[train] env_batch_size = 8`
  - `[train] batch_size = 512` (start)
  - `[train] minibatch_size = 512`
  - `[train] device = cuda`

## 2) Enable recurrent policy track (partial observability readiness)
- Edit `pufferlib/environments/drone_race/torch.py`
- Add `Recurrent = pufferlib.models.LSTMWrapper`
- Edit race config files above and add `[base] rnn_name = Recurrent`
- Keep one MLP variant with `rnn_name = None` for A/B comparison.

## 3) Use native experiment tracking + reproducibility
- No code edit required initially.
- Run with `--wandb` (or Neptune) and consistent group naming.

## 4) Controller modularization (align PRD Phase 3)
- Add `pufferlib/environments/drone_race/controller.py`
- Edit `pufferlib/environments/drone_race/environment.py` to use controller path in `step()`
- Add `tests/test_drone_race_controller.py` for saturation/fallback/determinism.

## 5) Perception realism knobs (sim2real prep)
- Edit `pufferlib/environments/drone_race/adapters.py`
- Add optional latency/bias/dropout burst behavior in camera stub adapter.
- Extend race configs with staged realism parameters.
- Extend `tests/test_drone_race_adapters.py` accordingly.

## 6) Eval diagnostics and visualization
- Edit `scripts/eval_drone_race.py` with component diagnostics columns.
- Add `scripts/plot_drone_race_eval.py` for trajectory/time distribution views.

## 7) Hyperparameter search once stable
- Add race sweep blocks in config (or dedicated sweep ini).
- Run `python -m pufferlib.pufferl sweep ...` after stable Tier3 baseline.

## 8) Ray backend (optional later)
- Validate `Multiprocessing` baseline first.
- Then compare against `Ray` backend for scale-out use.

## Scope boundary
- Primary surface: `drone_race` configs/env/scripts/tests.
- `drone_hover` remains unchanged unless cross-env shared-core fixes are explicitly required.
