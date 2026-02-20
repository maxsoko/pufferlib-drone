# Repository Guidelines

## Project Structure & Module Organization
- Top-level repos: `PufferLib/` (primary RL library + experiments) and `puffertank/` (aux repo).
- Core library code: `PufferLib/pufferlib/`.
- Environment code: `PufferLib/pufferlib/environments/` (e.g., `drone_hover/`).
- Configs: `PufferLib/pufferlib/config/` and `PufferLib/config/` (INI files).
- Scripts: `PufferLib/scripts/` (training, eval, utilities).
- Experiments & outputs: `PufferLib/experiments/` (models, CSVs).
- Tests: `PufferLib/tests/`.
- Project notes: `PufferLib/PRD.md` (experiment log + decisions).

## Build, Test, and Development Commands
- Install editable (typical): `pip install -e PufferLib`.
- Build native extension (if needed): `cd PufferLib && python setup.py build_ext --inplace`.
- Train (example): `python -m pufferlib.pufferl train drone_hover_curriculum --csv-log-dir experiments`.
- Hover->race warmstart (encoder transfer): `python -m pufferlib.pufferl train drone_race_curriculum_tier1 --warmstart-model-path experiments/<hover_model>.pt --warmstart-prefixes encoder --csv-log-dir experiments`.
- Evaluate (example): `PYTHONPATH=. python scripts/eval_drone_hover.py --episodes 20 --seed 42 --model-path experiments/<model>.pt`.
- DAgger/BC helpers: `PYTHONPATH=. python scripts/train_drone_hover_dagger.py ...` or `scripts/train_drone_hover_bc.py ...`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8–style naming (`snake_case` for functions/vars, `CamelCase` for classes).
- Keep modules small and focused; prefer explicit config flags in INI files.
- Name experiment artifacts consistently, e.g., `drone_hover_curriculum_<timestamp>.pt` and matching `.csv`.

## Testing Guidelines
- Tests live in `PufferLib/tests/`; add tests next to relevant modules when changing env logic.
- Run tests (typical): `cd PufferLib && python -m pytest tests`.
- Keep tests deterministic (seeded) when possible.

## Commit & Pull Request Guidelines
- Git history uses short, imperative messages (mostly lowercase) with occasional merges (e.g., “add model”, “update sweep param”).
- Follow that style: concise summary, no strict prefix required.
- PRs should include: brief description, commands run, and any new experiment CSV/model paths.

## Notes for Contributors
- Log major experiment runs and results in `PufferLib/PRD.md`.
- For eval parity, use `scripts/eval_drone_hover.py` with `--env-kwargs` to match training settings.
- Warmstart behavior: prefer `--warmstart-model-path` for cross-task transfer (hover->race); reserve `--load-model-path` for same-task resume/eval.
