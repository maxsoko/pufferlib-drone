# GPU Training Checklist (Drone Race)

1. Verify GPU and PyTorch CUDA visibility.
   - Command: `python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"`
2. Run a short CUDA smoke test before long jobs.
   - Use `drone_race_curriculum_tier1` for `10k-20k` timesteps.
3. Move race training to throughput defaults.
   - Set `backend=Multiprocessing`, `num_envs`/`num_workers` > 1, `device=cuda`.
4. Run curriculum ladder with warm-start checkpoints.
   - Tier1 -> Tier2 -> Tier3.
5. Standardize eval after each stage.
   - Run `quick20` and `full100` suites with `scripts/eval_drone_race.py`.
6. Enable experiment tracking.
   - Use `--wandb` (or Neptune) for run comparability.
7. Add one recurrent-policy comparison track.
   - Compare MLP vs recurrent on Tier2/Tier3.
8. Increase perception robustness over time.
   - Progressively raise camera noise/dropout settings.
9. Implement controller module milestones.
   - Separate setpoint tracking and fallback behavior from env dynamics glue.
10. Run sweep only after stable baseline.
   - Tune planner gains/reward weights and key train params.
11. Try compile and Ray after baseline stability.
   - `train.compile` and `vec.backend=Ray` are follow-on optimizations.
12. Keep `PRD.md` as single source of truth.
   - Append versioned updates and promotion decisions.
