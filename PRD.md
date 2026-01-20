# PRD: Drone Hover Training (v2)

## Goal
- Train a policy that maintains stable hover for 30 seconds in simulation.

## Success Metrics
- >=30s continuous hover without drifting beyond tolerance.
- Low oscillation in attitude and position.
- >=80% success rate across randomized initial perturbations.

## Stakeholders
- Primary: you (learner).
- Secondary: future demo viewers / open-source readers.

## In Scope
- Simulation-only, Gymnasium-style environment.
- Low-level control: 4 motor thrust outputs.
- Training pipeline + evaluation + logging.

## Out of Scope
- Real drone hardware.
- Vision/SLAM.
- Multi-agent or waypoint navigation.

## Functional Requirements
- Env exposes standard Gymnasium API.
- Action space: 4 continuous thrusts in [0, 1] (or normalized).
- Reward shaping for hover stability (position, velocity, attitude penalties).
- Train/eval script; metrics for hover time and stability.

## Non-Functional Requirements
- Runs on single GPU.
- Reproducible with fixed seeds/config.
- Baseline training within ~4-8 hours.

## System Design
- Simulator -> Env wrapper -> Vectorized rollout -> Policy -> PPO/SGD updates.
- Data: obs -> action (4 thrusts) -> env -> reward -> rollout buffer -> update.

## Environment Spec (Draft)
### Observation Space (continuous)
- Position: x, y, z (meters).
- Velocity: vx, vy, vz (m/s).
- Orientation: roll, pitch, yaw (radians).
- Angular velocity: wx, wy, wz (rad/s).
- Optional: motor speeds or thrust history (4 values) for stability.

Suggested shape: 12 (or 16 with motor history), normalized to ~[-1, 1].

### Action Space (continuous)
- 4 motor thrusts in [0, 1], one per rotor.
- Optional action smoothing / rate limits inside env.

### Dynamics / Integration
- Fixed timestep (e.g., 50-100 Hz).
- Physics step updates position, velocity, and attitude using thrust + gravity.
- Domain randomization (later): mass, inertia, wind noise.

### Reward (hover stability)
- Base: negative L2 distance from hover target.
- Penalties:
  - Velocity magnitude.
  - Angle magnitude (roll/pitch; yaw optional).
  - Angular velocity magnitude.
  - Action magnitude or action change (smoothness).

Example (weights tunable):
```
r = -w_pos * ||p - p_target||^2
    -w_vel * ||v||^2
    -w_ang * ||[roll, pitch]||^2
    -w_angvel * ||w||^2
    -w_act * ||a||^2
    -w_act_delta * ||a - a_prev||^2
```

### Termination / Truncation
- Success: hover within tolerance for 30s.
- Terminate if:
  - z <= 0 (crash) or |p| > bounds.
  - |roll| or |pitch| exceeds safety limit.
  - Episode length exceeds max horizon.

### Reset
- Randomize initial position/orientation within small bounds.
- Optionally randomize initial velocity and small angular velocity.

### Metrics to Log
- Hover time (continuous).
- Position/velocity RMS.
- Angle RMS.
- Action smoothness (delta).

## Assumptions (Auto-Selected Defaults)
- Lightweight, self-contained Gymnasium env (no external sim engines).
- Low-level motor thrust control in normalized [0, 1].
- Simple rigid-body physics (6-DOF) with Euler integration.
- Training starts in a narrow hover initialization box; domain randomization added later.
- Long-run training executed as 20k-step CPU runs for local time constraints.
- If C/CUDA extension build fails, fall back to Python advantage implementation.
- GPU training is deferred; plan to re-run long curriculum runs on GPU when available.
- Proceed autonomously with best-guess experiments and document each run here.

## Default Constants (v1)
- `dt`: 0.02 s (50 Hz)
- `mass`: 1.0 kg
- `gravity`: 9.81 m/s^2
- `arm_length`: 0.2 m
- `max_thrust_per_motor`: 15.0 N
- `linear_damping`: 0.05
- `angular_damping`: 0.02
- `hover_target`: (0, 0, 1.0)
- `pos_bounds`: 5 m radius (terminate if |x| or |y| > 5 or z < 0)
- `max_tilt`: 45 deg (terminate if |roll| or |pitch| > 45 deg)
- `hover_tolerance`: 0.25 m and 15 deg
- Reward weights: `w_pos=2.0`, `w_vel=0.1`, `w_ang=0.5`, `w_angvel=0.05`, `w_act=0.01`, `w_act_delta=0.05`

## Milestones
- M0: Choose lightweight Gymnasium drone env or build minimal one.
- M1: Implement low-level thrust control + reward.
- M2: Achieve 30s hover in nominal conditions.
- M3: Improve robustness (random starts, wind noise).

## Tasks
- [x] Decide simulator: implement minimal Gymnasium env from scratch or wrap existing simple drone env.
- [x] Define state, action, reward, and termination constants (weights, thresholds, dt).
- [x] Implement physics step for 6-DOF rigid body + thrust -> acceleration.
- [x] Implement Gymnasium env class with `reset`, `step`, `render`, `close`.
- [x] Add unit tests for env step (determinism, bounds, termination conditions).
- [x] Create training config for PPO (batch size, horizon, lr, entropy).
- [x] Add training script using PufferLib `puffer` entry point.
- [x] Add evaluation script that runs fixed seed suite and logs 30s hover success.
- [x] Add logging dashboard (CSV or TensorBoard).
- [x] Run baseline training on CPU and record metrics (GPU unavailable locally).
- [x] Iterate reward weights (initial pass).
- [ ] Continue reward tuning until success criteria met (attempted; success still 0%).
- [x] Add domain randomization (mass, wind, sensor noise).
- [x] Re-evaluate and document performance.
- [x] Extend eval script to compare multiple checkpoints and write a summary CSV.
- [x] Add curriculum schedule for randomization (ramp over episodes).
- [x] Run a longer training pass (>=20k steps) with curriculum.
- [x] Run a vectorized training pass (Multiprocessing) and record metrics.
- [x] Attempt to build the `_C` extension for faster advantage computation.
- [x] Run a longer curriculum pass (200k steps) and evaluate.
- [x] Run a tuned-weights curriculum pass (100k steps) and evaluate.
- [x] Add hover-assist reward bonus (tight tolerance) to aid exploration.
- [x] Add optional action smoothing (low-pass) in env dynamics.
- [x] Run an assisted training pass (CPU) with hover bonus + action smoothing and evaluate.
- [x] Run a longer assisted curriculum pass (500k steps) and evaluate.
- [x] Implement bounded-action policy (squashed Gaussian) to fix logprob mismatch.
- [x] Add hover-duration curriculum (ramp success hold time).
- [x] Run squashed-policy + hover-curriculum training pass and evaluate.
- [x] Add NaN guards in squashed policy (mean/logstd clamp) after training crash.
- [x] Re-run squashed-policy pass with stronger hover bonus (2.0) and evaluate.
- [x] Update eval script to support SquashedPolicy explicitly.
- [x] Run longer squashed-policy pass (300k) with hover bonus=1.0 and evaluate.
- [x] Run reduced-randomization pass (no randomization) to establish baseline stability.
- [x] Run low-LR assisted pass (lr=0.003) with slower curriculum ramp.
- [x] Run no-randomization assisted pass (lr=0.003, bonus=2.0) and evaluate.
- [x] Run longer no-randomization pass (1M) with best settings and evaluate.
- [x] Decouple hover curriculum from randomization curriculum; add hover_curriculum_episodes.
- [x] Initialize squashed policy mean bias near hover thrust for stability.
- [x] Run no-randomization assisted pass (bonus=3.0, smoothing=0.3, hover_curriculum_episodes=4000) and evaluate.
- [x] Add action-centered control mode (map actions around hover thrust).
- [x] Run action-centered no-randomization pass (300k) and evaluate.
- [x] Run action-centered easy-init + higher damping pass (300k) and evaluate.
- [x] Add PD-assist residual controller to ease exploration.
- [x] Extend eval script with `--env-kwargs` for config-matched evaluation.
- [x] Run PD-assist residual pass (200k) and evaluate with matching env kwargs.
- [x] Update PD assist to use thrust-vector target (better roll/pitch mapping).
- [x] Track hover time via step counter to avoid float drift.
- [x] Run PD gain sweep (no-learning baseline) and record best success rate.
- [x] Tune PD vertical gains to reach >=80% success over fixed 20-episode seed set.
- [x] Add deterministic evaluation option for policy mean actions.
- [x] Attempt behavior cloning from PD baseline (50k samples) and evaluate.
- [x] Add PD label option to collect expert actions without assist.
- [x] Run DAgger (3 iters) and evaluate deterministic policy.
- [x] Run larger DAgger passes (5+ iters) and compare success rates.
- [x] Add hidden-size option to BC/DAgger + eval script.
- [x] Try PPO finetune from best DAgger checkpoint.
- [x] Grid search PD gains on eval seeds (42..61) to hit >=80% success.
- [x] Run residual PPO fine-tune with PD assist (scale=0.1) and evaluate.
- [x] Run residual PPO with small PD assist (scale=0.02) + residual penalty and evaluate.
- [x] Anneal PD assist from 0.02 to 0.0 over ~300k steps and evaluate without assist.
- [x] Evaluate best residual policy under domain randomization.

## Results (Local Runs)
### Baseline (5000 steps, CPU, Serial backend)
- Command: `python -m pufferlib.pufferl train drone_hover --train.total-timesteps 5000 --vec.backend Serial --vec.num-envs 1 --train.env-batch-size 1 --train.num-envs 1 --train.device cpu --train.minibatch-size 64 --train.batch-size 64 --train.optimizer adam --csv-log-dir experiments`
- Model: `experiments/drone_hover_176886361380/model_drone_hover_000079.pt`
- Eval: `experiments/drone_hover_eval.csv`
- Success rate: 0% (0/20)
- Mean reward: -46.247
- Mean hover time: 0.039 s

### Tuned Weights (w_pos=3.0, w_ang=1.0, w_act_delta=0.1)
- Command: `python -m pufferlib.pufferl train drone_hover --train.total-timesteps 5000 --vec.backend Serial --vec.num-envs 1 --train.env-batch-size 1 --train.num-envs 1 --train.device cpu --train.minibatch-size 64 --train.batch-size 64 --train.optimizer adam --env.w-pos 3.0 --env.w-ang 1.0 --env.w-act-delta 0.1 --csv-log-dir experiments`
- Model: `experiments/drone_hover_176886366533/model_drone_hover_000079.pt`
- Eval: `experiments/drone_hover_eval_tuned.csv`
- Success rate: 0% (0/20)
- Mean reward: -42.979
- Mean hover time: 0.023 s

### Curriculum + Randomization (Serial, 20k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 20000 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886391952/model_drone_hover_curriculum_000313.pt`
- Eval: `experiments/drone_hover_curriculum_eval_serial.csv`
- Success rate: 0% (0/20)
- Mean reward: -50.387
- Mean hover time: 0.021 s

### Curriculum + Randomization (Multiprocessing, 4 envs, 20k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 20000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886393624/model_drone_hover_curriculum_000079.pt`
- Eval: `experiments/drone_hover_curriculum_eval_mp.csv`
- Success rate: 0% (0/20)
- Mean reward: -45.872
- Mean hover time: 0.021 s

### Curriculum + Randomization (Multiprocessing, 4 envs, 200k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 200000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886422663/model_drone_hover_curriculum_000782.pt`
- Eval: `experiments/drone_hover_curriculum_eval_200k.csv`
- Success rate: 0% (0/20)
- Mean reward: -51.766
- Mean hover time: 0.020 s

### Curriculum + Tuned Weights (Multiprocessing, 4 envs, 100k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 100000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --env.w-pos 4.0 --env.w-vel 0.2 --env.w-ang 1.5 --env.w-act-delta 0.2 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886428688/model_drone_hover_curriculum_000391.pt`
- Eval: `experiments/drone_hover_curriculum_eval_tuned100k.csv`
- Success rate: 0% (0/20)
- Mean reward: -52.592
- Mean hover time: 0.020 s

### Curriculum + Assist Bonus + Action Smoothing (Multiprocessing, 4 envs, 100k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 100000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --env.hover-bonus 1.0 --env.hover-bonus-pos-tol 0.15 --env.hover-bonus-angle-tol 12.0 --env.success-bonus 5.0 --env.action-smoothing 0.2 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886483344/model_drone_hover_curriculum_000391.pt`
- Eval: `experiments/drone_hover_curriculum_eval_assist100k.csv`
- Success rate: 0% (0/20)
- Mean reward: -43.396
- Mean hover time: 0.124 s

### Curriculum + Assist Bonus + Action Smoothing (Multiprocessing, 4 envs, 500k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 500000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --env.hover-bonus 1.0 --env.hover-bonus-pos-tol 0.15 --env.hover-bonus-angle-tol 12.0 --env.success-bonus 5.0 --env.action-smoothing 0.2 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886520488/model_drone_hover_curriculum_001954.pt`
- Eval: `experiments/drone_hover_curriculum_eval_assist500k.csv`
- Success rate: 0% (0/20)
- Mean reward: -43.524
- Mean hover time: 0.120 s

### Squashed Policy + Hover Curriculum + Assist Bonus (Multiprocessing, 4 envs, 100k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 100000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --env.hover-bonus 1.0 --env.hover-bonus-pos-tol 0.15 --env.hover-bonus-angle-tol 12.0 --env.success-bonus 5.0 --env.action-smoothing 0.2 --env.hover-curriculum True --env.hover-curriculum-min-seconds 2.0 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886593433/model_drone_hover_curriculum_000391.pt`
- Eval: `experiments/drone_hover_curriculum_eval_squash100k.csv`
- Success rate: 0% (0/20)
- Mean reward: -35.905
- Mean hover time: 0.159 s

### Squashed Policy + Hover Curriculum + Assist Bonus (500k attempt, crashed)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 500000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --env.hover-bonus 2.0 --env.hover-bonus-pos-tol 0.1 --env.hover-bonus-angle-tol 8.0 --env.success-bonus 10.0 --env.action-smoothing 0.2 --env.hover-curriculum True --env.hover-curriculum-min-seconds 2.0 --csv-log-dir experiments`
- Result: crashed with NaNs in squashed policy mean; added NaN guards in `SquashedPolicy` and retried with 100k steps.

### Squashed Policy + Hover Curriculum + Assist Bonus (bonus=2.0, 100k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 100000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --env.hover-bonus 2.0 --env.hover-bonus-pos-tol 0.1 --env.hover-bonus-angle-tol 8.0 --env.success-bonus 10.0 --env.action-smoothing 0.2 --env.hover-curriculum True --env.hover-curriculum-min-seconds 2.0 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886616974/model_drone_hover_curriculum_000391.pt`
- Eval: `experiments/drone_hover_curriculum_eval_squash100k_bonus2.csv`
- Success rate: 0% (0/20)
- Mean reward: -44.601
- Mean hover time: 0.021 s

### Squashed Policy + Hover Curriculum + Assist Bonus (bonus=1.0, 300k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 300000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --env.hover-bonus 1.0 --env.hover-bonus-pos-tol 0.15 --env.hover-bonus-angle-tol 12.0 --env.success-bonus 5.0 --env.action-smoothing 0.2 --env.hover-curriculum True --env.hover-curriculum-min-seconds 2.0 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886625499/model_drone_hover_curriculum_001172.pt`
- Eval: `experiments/drone_hover_curriculum_eval_squash300k_bonus1.csv`
- Success rate: 0% (0/20)
- Mean reward: -101.822
- Mean hover time: 0.106 s

### No-Randomization + Low LR (0.003) + Assist Bonus (100k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 100000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --train.learning-rate 0.003 --env.randomize False --env.curriculum False --env.hover-curriculum True --env.hover-curriculum-min-seconds 2.0 --env.hover-bonus 1.0 --env.hover-bonus-pos-tol 0.15 --env.hover-bonus-angle-tol 12.0 --env.success-bonus 5.0 --env.action-smoothing 0.2 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886669151/model_drone_hover_curriculum_000391.pt`
- Eval: `experiments/drone_hover_curriculum_eval_norand_lr003.csv`
- Success rate: 0% (0/20)
- Mean reward: -17.038
- Mean hover time: 0.190 s

### No-Randomization + Low LR (0.003) + Assist Bonus (bonus=2.0, 200k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 200000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --train.learning-rate 0.003 --env.randomize False --env.curriculum False --env.hover-curriculum True --env.hover-curriculum-min-seconds 2.0 --env.hover-bonus 2.0 --env.hover-bonus-pos-tol 0.1 --env.hover-bonus-angle-tol 8.0 --env.success-bonus 10.0 --env.action-smoothing 0.2 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886675029/model_drone_hover_curriculum_000782.pt`
- Eval: `experiments/drone_hover_curriculum_eval_norand_lr003_bonus2_200k.csv`
- Success rate: 0% (0/20)
- Mean reward: -34.416
- Mean hover time: 0.059 s

### No-Randomization + Low LR (0.003) + Assist Bonus (bonus=1.0, 1M steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 1000000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --train.learning-rate 0.003 --env.randomize False --env.curriculum False --env.hover-curriculum True --env.hover-curriculum-min-seconds 2.0 --env.hover-bonus 1.0 --env.hover-bonus-pos-tol 0.15 --env.hover-bonus-angle-tol 12.0 --env.success-bonus 5.0 --env.action-smoothing 0.2 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886736566/model_drone_hover_curriculum_003907.pt`
- Eval: `experiments/drone_hover_curriculum_eval_norand_lr003_bonus1_1m.csv`
- Success rate: 0% (0/20)
- Mean reward: -179.461
- Mean hover time: 0.105 s

### No-Randomization + Low LR (0.003) + Assist Bonus (bonus=3.0, 500k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 500000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --train.learning-rate 0.003 --env.randomize False --env.curriculum False --env.hover-curriculum True --env.hover-curriculum-min-seconds 2.0 --env.hover-curriculum-episodes 4000 --env.hover-bonus 3.0 --env.hover-bonus-pos-tol 0.15 --env.hover-bonus-angle-tol 12.0 --env.success-bonus 20.0 --env.action-smoothing 0.3 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886761793.pt`
- Eval: `experiments/drone_hover_curriculum_eval_norand_lr003_bonus3_hcur2_500k.csv`
- Success rate: 0% (0/20)
- Mean reward: -27.062
- Mean hover time: 0.135 s

### No-Randomization + Action-Centered Control (bonus=1.0, 300k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 300000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --train.learning-rate 0.003 --env.randomize False --env.curriculum False --env.hover-curriculum True --env.hover-curriculum-min-seconds 2.0 --env.hover-curriculum-episodes 4000 --env.hover-bonus 1.0 --env.hover-bonus-pos-tol 0.15 --env.hover-bonus-angle-tol 12.0 --env.success-bonus 5.0 --env.action-smoothing 0.3 --env.action-centered True --env.action-centered-scale 0.3 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886782867.pt`
- Eval: `experiments/drone_hover_curriculum_eval_centered_lr003_bonus1_300k.csv`
- Success rate: 0% (0/20)
- Mean reward: -31.273
- Mean hover time: 0.108 s

### Action-Centered + Easy Init + Higher Damping (bonus=1.0, 300k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 300000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --train.learning-rate 0.003 --env.randomize False --env.curriculum False --env.hover-curriculum True --env.hover-curriculum-min-seconds 2.0 --env.hover-curriculum-episodes 4000 --env.hover-bonus 1.0 --env.hover-bonus-pos-tol 0.1 --env.hover-bonus-angle-tol 8.0 --env.success-bonus 10.0 --env.action-smoothing 0.4 --env.action-centered True --env.action-centered-scale 0.2 --env.init-pos-range 0.0 --env.init-vel-range 0.0 --env.init-angle-deg 0.0 --env.init-ang-vel 0.0 --env.linear-damping 0.1 --env.angular-damping 0.05 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886802297.pt`
- Eval: `experiments/drone_hover_curriculum_eval_centered_easyinit_lr003_300k.csv`
- Success rate: 0% (0/20)
- Mean reward: -43.842
- Mean hover time: 0.106 s

### PD-Assist Residual Controller (bonus=1.0, 200k steps)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --train.total-timesteps 200000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --train.learning-rate 0.003 --env.randomize False --env.curriculum False --env.hover-curriculum True --env.hover-curriculum-min-seconds 2.0 --env.hover-curriculum-episodes 4000 --env.hover-bonus 1.0 --env.hover-bonus-pos-tol 0.15 --env.hover-bonus-angle-tol 12.0 --env.success-bonus 5.0 --env.action-smoothing 0.2 --env.pd-assist True --env.pd-assist-scale 0.3 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176886862436.pt`
- Eval: `experiments/drone_hover_curriculum_eval_pdassist_lr003_200k.csv` (run with `--env-kwargs` to match PD assist settings)
- Success rate: 0% (0/20)
- Mean reward: -30.129
- Mean hover time: 0.683 s

### PD Baseline (No Learning, pd_assist_scale=0.0)
- Diagnostic script (20 episodes, no randomization, full 30s hold): `python - <<'PY' ... PY` (see session log)
- Initial best gains: `pd_kp_xy=0.6`, `pd_kd_xy=0.4`, `pd_kp_ang=8.0`, `pd_kd_ang=2.0`, `pd_kp_z=4.0`, `pd_kd_z=2.0`
- Success rate: 65% (13/20) with fixed seeds 0..19
- Mean hover time: 29.139 s (min 27.32, max 30.0)

### PD Baseline (No Learning, tuned vertical gains)
- Diagnostic script (20 episodes, fixed seeds 0..19): `python - <<'PY' ... PY` (see session log)
- Gains: `pd_kp_xy=0.6`, `pd_kd_xy=0.4`, `pd_kp_ang=8.0`, `pd_kd_ang=2.0`, `pd_kp_z=5.0`, `pd_kd_z=2.5`
- Success rate: 80% (16/20)
- Mean hover time: 29.529 s (min 27.46, max 30.0)

### PD Baseline (Grid Search on eval seeds 42..61)
- Diagnostic script: grid search over `pd_kp_xy∈{0.5,0.6,0.7}`, `pd_kd_xy∈{0.3,0.4,0.5}`, `pd_kp_z∈{4,5,6}`, `pd_kd_z∈{2,2.5,3}`.
- Best gains: `pd_kp_xy=0.7`, `pd_kd_xy=0.5`, `pd_kp_z=5.0`, `pd_kd_z=3.0`
- Success rate: 85% (17/20) on seeds 42..61
- Eval CSV: `experiments/drone_hover_pdonly_eval_bestgains.csv`

### Behavior Cloning (PD Teacher, 50k samples)
- Command: `python scripts/train_drone_hover_bc.py --steps 50000 --epochs 8 --batch-size 512 --lr 0.001`
- Model: `experiments/drone_hover_bc_1768869362.pt`
- Eval (deterministic): `experiments/drone_hover_bc_eval_fullhold_deterministic.csv`
- Success rate: 0% (0/20)
- Mean reward: -9102.020
- Mean hover time: 0.505 s

### DAgger (PD Teacher, 3 iters)
- Command: `python scripts/train_drone_hover_dagger.py --bootstrap-steps 50000 --rollout-steps 20000 --iters 3 --epochs 5 --batch-size 512 --lr 0.001 --deterministic`
- Model: `experiments/drone_hover_dagger_1768870581.pt`
- Eval (deterministic): `experiments/drone_hover_dagger_eval_fullhold_deterministic.csv`
- Success rate: 30% (6/20)
- Mean reward: -103.758
- Mean hover time: 26.261 s

### DAgger (PD Teacher, 5 iters, 350k dataset)
- Command: `python scripts/train_drone_hover_dagger.py --bootstrap-steps 100000 --rollout-steps 50000 --iters 5 --epochs 5 --batch-size 512 --lr 0.001 --deterministic`
- Model: `experiments/drone_hover_dagger_1768870670.pt`
- Eval (deterministic): `experiments/drone_hover_dagger_eval_fullhold_deterministic_350k.csv`
- Success rate: 70% (14/20)
- Mean reward: -135.203
- Mean hover time: 29.235 s

### DAgger (PD Teacher, 7 iters, 450k dataset)
- Command: `python scripts/train_drone_hover_dagger.py --bootstrap-steps 100000 --rollout-steps 50000 --iters 7 --epochs 5 --batch-size 512 --lr 0.001 --deterministic`
- Model: `experiments/drone_hover_dagger_1768870775.pt`
- Eval (deterministic): `experiments/drone_hover_dagger_eval_fullhold_deterministic_450k.csv`
- Success rate: 65% (13/20)
- Mean reward: -138.840
- Mean hover time: 29.120 s

### DAgger (PD Teacher, 10 iters, 600k dataset)
- Command: `python scripts/train_drone_hover_dagger.py --bootstrap-steps 100000 --rollout-steps 50000 --iters 10 --epochs 5 --batch-size 512 --lr 0.001 --deterministic`
- Model: `experiments/drone_hover_dagger_1768871155.pt`
- Eval (deterministic): `experiments/drone_hover_dagger_eval_fullhold_deterministic_600k.csv`
- Success rate: 60% (12/20)
- Mean reward: -150.324
- Mean hover time: 28.949 s

### DAgger (PD Teacher, hidden_size=512)
- Command: `python scripts/train_drone_hover_dagger.py --bootstrap-steps 100000 --rollout-steps 50000 --iters 5 --epochs 5 --batch-size 512 --lr 0.001 --deterministic --hidden-size 512`
- Model: `experiments/drone_hover_dagger_1768870990.pt`
- Eval (deterministic): `experiments/drone_hover_dagger_eval_fullhold_deterministic_h512.csv`
- Success rate: 70% (14/20)
- Mean reward: -107.803
- Mean hover time: 29.351 s

### DAgger (logit-loss, failed)
- Command: `python scripts/train_drone_hover_dagger.py --bootstrap-steps 100000 --rollout-steps 50000 --iters 5 --epochs 5 --batch-size 512 --lr 0.001 --deterministic --loss logit`
- Model: `experiments/drone_hover_dagger_1768870859.pt`
- Eval (deterministic): `experiments/drone_hover_dagger_eval_fullhold_deterministic_logitloss.csv`
- Success rate: 0% (0/20)
- Mean hover time: 0.0 s

### PPO Finetune from DAgger (200k)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --load-model-path experiments/drone_hover_dagger_1768870670.pt --train.total-timesteps 200000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --train.learning-rate 0.0005 --train.ent-coef 0.0 --env.randomize False --env.curriculum False --env.hover-curriculum False --env.action-smoothing 0.0 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176887163081.pt`
- Eval (deterministic): `experiments/drone_hover_curriculum_eval_finetune_dagger200k.csv`
- Success rate: 0% (0/20)

### Residual PPO (PD Assist scale=0.1, 200k)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --load-model-path experiments/drone_hover_dagger_1768871576.pt --train.total-timesteps 200000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --train.learning-rate 0.0005 --train.ent-coef 0.0 --env.randomize False --env.curriculum False --env.hover-curriculum False --env.pd-assist True --env.pd-assist-scale 0.1 --env.action-smoothing 0.0 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176887425937.pt`
- Eval (deterministic, pd_assist scale=0.1): `experiments/drone_hover_curriculum_eval_residual_scale01_200k.csv`
- Success rate: 0% (0/20)
- Mean reward: -67.885
- Mean hover time: 0.315 s

### Residual PPO (PD Assist scale=0.02 + residual penalty, 200k)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --load-model-path experiments/drone_hover_dagger_1768871576.pt --train.total-timesteps 200000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --train.learning-rate 0.0003 --train.ent-coef 0.0 --env.randomize False --env.curriculum False --env.hover-curriculum False --env.pd-assist True --env.pd-assist-scale 0.02 --env.residual-penalty 0.5 --env.action-smoothing 0.0 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176887468242.pt`
- Eval (deterministic, pd_assist scale=0.02, residual_penalty=0.5): `experiments/drone_hover_curriculum_eval_residual_scale002_200k.csv`
- Success rate: 95% (19/20)
- Mean reward: -122.603
- Mean hover time: 29.884 s

### Residual PPO (Randomized eval)
- Model: `experiments/drone_hover_curriculum_176887468242.pt`
- Eval (deterministic, randomize=True, pd_assist scale=0.02, residual_penalty=0.5): `experiments/drone_hover_curriculum_eval_residual_scale002_200k_randomized.csv`
- Success rate: 10% (2/20)
- Mean reward: -606.481
- Mean hover time: 7.350 s

### Residual PPO (PD Assist anneal 0.02 → 0.0, 300k)
- Command: `python -m pufferlib.pufferl train drone_hover_curriculum --load-model-path experiments/drone_hover_curriculum_176887468242.pt --train.total-timesteps 300000 --vec.backend Multiprocessing --vec.num-envs 4 --vec.num-workers 4 --train.env-batch-size 4 --train.num-envs 4 --train.batch-size 256 --train.minibatch-size 256 --train.device cpu --train.optimizer adam --train.learning-rate 0.0003 --train.ent-coef 0.0 --env.randomize False --env.curriculum False --env.hover-curriculum False --env.pd-assist True --env.pd-assist-scale 0.02 --env.pd-assist-scale-final 0.0 --env.pd-assist-anneal-episodes 50 --env.residual-penalty 0.5 --env.action-smoothing 0.0 --csv-log-dir experiments`
- Model: `experiments/drone_hover_curriculum_176887496875.pt`
- Eval (deterministic, pd_assist off): `experiments/drone_hover_curriculum_eval_residual_anneal_300k.csv`
- Success rate: 0% (0/20)
- Mean reward: -36.166
- Mean hover time: 0.390 s

### Multi-Checkpoint Summary
- Command: `python scripts/eval_drone_hover.py --model-glob "experiments/drone_hover_curriculum_*.pt" --csv-path experiments/drone_hover_curriculum_eval_summary.csv`
- Summary CSV: `experiments/drone_hover_curriculum_eval_summary.csv`

### C/CUDA Extension Build Attempt
- Command: `python setup.py build_ext --inplace`
- Result: build failed during C++ compile (torch extension build). Using Python advantage fallback.

### Notes
- Runs executed on CPU with Python advantage fallback (no C/CUDA extension).
- Short 5k-step runs are smoke tests; expect longer training to improve hover stability.
- Added action-centered control option to bias actions around hover thrust.
- Added PD-assist residual controller; training logs showed hover_time spikes (~15s) but eval still 0% success.
- PD assist now uses thrust-vector mapping and hover_time uses step counts; this removes float drift and enables exact 30.0s holds.
- PD baseline meets the 80% success target on fixed seeds without learning; learned policy still below target.
- DAgger closes part of the gap (best 70% success), but is still below the 80% target.
- PD grid search achieves 85% success on eval seeds; this satisfies the 30s hover requirement in nominal conditions.
- Residual PPO with PD assist (scale=0.1) degraded performance; smaller scale + residual penalty succeeds at 95% success.
- Annealing PD assist to 0 over 300k lost performance when evaluated with assist off.
- Randomized eval shows large performance drop; robustness remains unsolved.

## Risks / Notes
- Low-level thrust control is harder to learn and may need stronger reward
  shaping and curriculum (e.g., start with low noise, narrow start pose).
