# Full-Policy Native Iteration Report — 2026-07-13

> **Target-geometry correction (2026-07-14):** this report used `1.5 m` as a
> native radius. Official evidence later established that `1.5 m` is the inner
> gate width, so the correct native radial tolerance is `0.75 m`. The experiments
> below remain valid curriculum/history evidence, but none of their radius-1.5
> results are target-geometry promotion evidence. Continue from
> `docs/full_policy_execution_prompt.md`, which contains the corrected command.

## Outcome

No checkpoint was promoted to the official simulator. The official-attempt
budget was preserved.

The iteration produced a simple native curriculum loop, corrected a native
recurrent-backend defect, and materially improved late-course behavior:

- training-only course-spline roll/thrust targets;
- per-agent mixed full/segment starts and gate-staged cross-track reward;
- continuous MinGRU state across rollout chunks, per-agent terminal masking,
  and sequence-start state restoration during PPO training;
- training-only per-gate aperture radii plus automated accept/reject annealing;
- configurable reward scaling before the stable backend reward clamp;
- deterministic late-course progress improved from a 14.06 m final-gate miss
  to repeatable full-course completion from the true start under the current
  relaxed final-gate curriculum.

The retained true-recurrent checkpoints are:

- best late segment: `checkpoints/drone_race_full_policy_stage_d_gate4/1783999349787/0000000001998848.bin`;
- closest full-start radius-1.5 gate-3 approach: `checkpoints/drone_race_full_policy_stage_d_gate4/1783999503025/0000000001998848.bin`;
- first full-start downstream-credit checkpoint at radius 2.0: `checkpoints/drone_race_full_policy_stage_d_gate4/1783999938032/0000000001998848.bin`.
- latest accepted parent at this report cut (gate 2 radius 2.0, final gate
  radius 12.625):
  `checkpoints/drone_race_full_policy_stage_d_gate4/1784003050653/0000000001998848.bin`.

The current parent completes 96.24% of development episodes with zero crashes
at its curriculum geometry. It is not promotable: gate 2 is still relaxed to
2.0 m, the final gate is still relaxed to 12.625 m, and full-start evaluation at
the target 1.5 m geometry still passes only two gates.

## Gate-position and observation contract

The native environment contains the fixed course positions in `env->gates[]`.
The current v3385 calibration in
`config/drone_race_full_policy_stage_d_gate4.ini` is:

| Native index | x | y | z |
|---:|---:|---:|---:|
| 0 | 22.35 | 0.19 | 0.30 |
| 1 | 43.77 | 2.04 | -3.45 |
| 2 | 67.53733 | -1.24074 | -10.23080 |
| 3 | 95.63180 | 5.10711 | -18.24322 |

Those absolute positions remain simulator/training state. They are used for
physics, resets, crossing checks, and training rewards, but are not placed in
the 32-value deployed-policy observation. The policy receives active-gate
camera-relative pose/motion, HIGHRES_IMU attitude/rates, race phase, previous
action, and elapsed fraction. It receives neither future-gate coordinates nor
absolute vehicle position/velocity.

## Environment changes

1. `teacher_course_spline`
   - Builds a training-only spline through the configured course.
   - Produces coupled roll/thrust action targets using the calibrated v3385
     attitude plant.
   - Does not mutate or extend the policy observation.

2. `mixed_start_curriculum` and `segment_start_probability`
   - Sample the official origin or a measured segment transition independently
     for each native agent.
   - Keep terminal/late-gate credit in the same PPO batch as full-start
     recurrent trajectories.

3. `cross_track_from_gate_index`
   - Allows stronger cross-track shaping to begin at a selected active gate.
   - Prevents a late-course reward change from modifying solved early segments.

All three options default off or behavior-preserving in the stage config.

4. Native recurrent-state correction
   - `reset_state=False` for the full-policy stage preserves state across
     32-step rollout calls.
   - The rollout kernel zeros only agents whose episodes terminated.
   - PPO records the state at each rollout start and restores the selected
     agent state for sequence training.
   - Sequences containing an internal episode boundary are excluded from
     prioritized replay; a sequence beginning immediately after a terminal is
     valid because its captured initial state is already zero.

5. Per-gate radius curriculum
   - `gate0_radius` through `gate3_radius` override only the selected training
     aperture; zero inherits the global radius.
   - Absolute gate centers remain internal and observations are unchanged.
   - `scripts/train_full_policy_radius_curriculum.py` screens the next radius,
     fine-tunes only when required, accepts only threshold-passing checkpoints,
     and writes resumable JSON state while retaining the previous parent on
     failure.

6. Reward preprocessing controls
   - `train.reward_scale` scales shaped rewards before `train.reward_clip`.
   - Both default to historical behavior (`1.0` scale and `1.0` clip).
   - The successful aperture continuation uses scale `0.005` and clip `1.0`,
     keeping the terminal reward near one while preserving relative teacher and
     centerline signal. Raising the clip to 20 instead caused value-loss growth
     and was rejected.

## Reproducible command templates

All PPO runs used one training process, 1024 agents, 60 Hz, 2M transitions per
chunk, and the native CUDA backend. The exact gate-2 command was:

```bash
CUDA_HOME=/usr/local/cuda-12.6 \
PATH=/root/pufferlib-drone/.venv/bin:/usr/local/cuda-12.6/bin:$PATH \
NVCC_ARCH=sm_86 \
.venv/bin/python -m pufferlib.pufferl train \
  drone_race_full_policy_stage_d_gate4 \
  --load-model-path SOURCE.bin \
  --checkpoint-interval 500000 \
  --train.total-timesteps 2000000 \
  --train.learning-rate 0.00005 \
  --env.start-gate-index 2 \
  --env.start-x 43.844997 --env.start-y 2.778257 --env.start-z -3.296632 \
  --env.start-vx 10 --env.start-vy 0.996883 --env.start-vz -2.037398 \
  --env.start-qw 0.982270 --env.start-qx 0.173252 \
  --env.start-qy 0.070530 --env.start-qz -0.012440 \
  --env.start-wx 0 --env.start-wy 0 --env.start-wz 0 \
  --env.start-elapsed-time 6.8 \
  --env.gate-radius 1.5 \
  --env.w-action-teacher 500 \
  --env.teacher-course-spline 1 \
  --env.teacher-from-gate-index 2 \
  --env.teacher-pitch-from-gate-index 4 \
  --env.teacher-roll-from-gate-index 2 \
  --env.teacher-thrust-from-gate-index 2
```

Mixed runs added exactly:

```bash
--env.mixed-start-curriculum 1 --env.segment-start-probability PROBABILITY
```

The staged cross-track experiments added exactly:

```bash
--env.w-cross-track WEIGHT --env.cross-track-from-gate-index 3
```

Deterministic development evaluations used:

```bash
.venv/bin/python scripts/eval_drone_race_checkpoint.py CHECKPOINT.bin \
  --env-name drone_race_full_policy_stage_d_gate4 \
  --eval-episodes 512 --horizon 32 --max-rollouts 64 \
  --json-path REPORT.json --csv-path REPORT.csv \
  --env.gate-radius 1.5
```

Segment evaluations appended the same measured gate-2 start fields shown in
the training command. Evaluation patches Gaussian `log_std` to `-20`, so the
reported action is the deployed deterministic mean.

The simple autonomous continuation is:

```bash
.venv/bin/python scripts/train_full_policy_radius_curriculum.py \
  --source-checkpoint SOURCE.bin \
  --initial-radius CURRENT_PASSING_RADIUS \
  --target-radius 1.5 --radius-step 0.125 \
  --min-radius-step 0.0078125 \
  --reward-scale 0.005 \
  --set env.gate-radius=1.5 \
  --set env.gate2-radius=2.0 \
  --set env.w-action-teacher=500 \
  --set env.teacher-course-spline=1 \
  --set env.teacher-from-gate-index=2 \
  --set env.teacher-pitch-from-gate-index=4 \
  --set env.teacher-roll-from-gate-index=2 \
  --set env.teacher-thrust-from-gate-index=2
```

This command never launches the official simulator and never replaces the last
passing checkpoint with a failed child. `--resume-state STATE.json` restarts an
interrupted run from the JSON's last accepted checkpoint and radius; unevaluated
children are deliberately ignored. When both bounded children fail, the runner
can now halve the failed local step and retry from the retained parent down to
`--min-radius-step`, recording every refinement in the atomic state file. The
first recovery resumed the accepted 9.25 m parent with a 0.03125 m step: the
unchanged checkpoint passed 9.21875 m at 92.97% and 9.1875 m at 90.39%, both
with zero crashes, after the direct 9.125 m jump had blocked.

`scripts/finish_full_policy_native_curriculum.py` waits for gate 3 to reach the
target, automatically runs the same retained-parent anneal for gate 2 from 2.0 m
to 1.5 m, and then performs the required 4096-episode deterministic promotion
evaluation at target geometry. It enforces one trainer at a time and does not
invoke the official simulator.

## Deterministic evidence

### Historical reset-state diagnostics

The following table was generated before the recurrent-backend audit. Those
runs reset MinGRU state every 32 steps (0.53 s). They remain useful for comparing
reward/curriculum changes under the old setup, but are not promotion evidence.

All rows are development evaluations at radius 1.5. `Gates` is ordered gates
passed; no listed run crashed.

| Candidate | Start | Change | Gates | Closest remaining gate | Result |
|---|---|---|---:|---:|---|
| `1783994834770/...1998848.bin` | gate 2 | wide baseline | 3 | 14.057 m | baseline |
| `1783995756581/...1998848.bin` | gate 2 | spline, teacher 50, 2M | 3 | 13.208 m | improved |
| `1783996972022/...1998848.bin` | gate 2 | teacher 500, +2M | 2.999 | 7.880 m | improved |
| `1783997091558/...1998848.bin` | gate 2 | unchanged, +2M | 3 | 6.720 m | improved |
| `1783997210147/...1998848.bin` | gate 2 | unchanged, +2M | 3 | 7.155 m | rejected |
| `1783997358596/...1998848.bin` | gate 3 | final-leg reset, teacher 500 | 3 | 5.054 m | worse than 4.957 m matched baseline |
| `1783997517679/...1998848.bin` | gate 3 | teacher 5000 | 3 | 5.048 m | rejected |
| `1783997641844/...1998848.bin` | gate 2 | teacher 5000 | 3 | 7.539 m | rejected |
| `1783998000056/...1998848.bin` | gate 2 | gate-3 cross-track 50 | 3 | 7.152 m | rejected |
| `1783998134748/...1998848.bin` | gate 2 | gate-3 cross-track 10 | 3 | 7.147 m | rejected |
| `1783998322192/...1998848.bin` | full | mixed 50%, teacher 500, 2M | 2 | 31.773 m | not promotable |
| `1783998322192/...1998848.bin` | gate 2 | same checkpoint | 3 | **6.439 m** | retained late-course best |
| `1783998474593/...1998848.bin` | gate 2 | same mixed run, +2M | 2.569 | 17.255 m | rejected |
| `1783998627475/...1998848.bin` | full | mixed 25%, +2M | 2 | 31.756 m | rejected |
| `1783998627475/...1998848.bin` | gate 2 | same checkpoint | 2.991 | 6.928 m | rejected |

The per-experiment JSON/CSV artifacts are under
`logs/drone_race_full_policy_stage_d_gate4/` with matching descriptive names.
Training throughput was approximately 33K–41K steps/s; each 2M chunk took
roughly 47–61 seconds.

### True-recurrent evidence

All rows below use state continuity across rollout chunks and terminal-only
per-agent resets.

| Candidate | Start/radius | Gates | Remaining miss/evidence | Result |
|---|---|---:|---:|---|
| `v21b_phase_adapter_v2_logstd_minus2.bin` | full / 1.5 | 2 | gate-3 miss | corrected baseline |
| `1783997091558/...1998848.bin` | gate 2 / 1.5 | 3 | 6.190 m final miss | corrected pre-fix checkpoint eval |
| `1783999190544/...1998848.bin` | gate 2 / 1.5 | 3 | 5.834 m | recurrent mixed +2M |
| `1783999349787/...1998848.bin` | gate 2 / 1.5 | 3 | **5.624 m** | retained segment best |
| `1783999503025/...1998848.bin` | full / 1.5 | 2 | about 1.73 m radial at gate 3 | retained full-start approach |
| `1783999938032/...1998848.bin` | full / 2.0 | 2.992 | reached final plane; 14.202 m miss | first true full-start downstream credit |
| `1784000641570/...1998848.bin` | full / g2 2.0, g3 15.0 | 4.000 | 100% success, 12.583 s | first full-course native curriculum success |
| `1784000901320/...1998848.bin` | full / g2 2.0, g3 13.5 | 3.993 | 99.59% success, 12.524 s | accepted |
| `1784001299760/...1998848.bin` | full / g2 2.0, g3 13.25 | 3.931 | 95.93% success, 12.049 s | accepted |
| `1784001546371/...1998848.bin` | full / g2 2.0, g3 13.0 screen | 3.882 | 93.03% success, 11.696 s | retained pre-normalization parent |
| `1784002500388/...1998848.bin` | full / g2 2.0, g3 12.875 | 3.728 | 86.19% success | unscaled fine-tune rejected automatically |
| `1784002736634/...1998848.bin` | full / g2 2.0, g3 12.875 | 3.970 | **97.30% success**, 12.232 s | reward scale 0.005; accepted |
| `1784002886011/...1998848.bin` | full / g2 2.0, g3 12.75 | 3.973 | **97.29% success**, 12.233 s | reward scale 0.005; accepted |
| `1784003050653/...1998848.bin` | full / g2 2.0, g3 12.625 | 3.962 | **96.24% success**, 12.111 s | reward scale 0.005; latest report-cut parent |

The full-start global radius-2.0 continuation, global radius-1.75 anneal,
unscaled gate-3 radius 12.875 fine-tune, clip-20 run (83.19%), and other
regressions were rejected. No target-radius or 4096-episode promotion evaluation
was justified.

Rollout partition invariance was verified on the same 1024 deterministic
episodes. Horizon 32 and horizon 64 both produced exactly:

```text
gates_passed=2.0
episode_length=601.810546875
closest_gate_range=32.132362365722656
final=(67.6351318359375, 0.28604355454444885, -8.601861953735352)
```

## Conclusions

- The simpler high-throughput direction is correct: native RL produces useful
  course changes in about one minute per controlled experiment.
- The main failure is not absent gate geometry. The largest infrastructure
  defect was unintended recurrent-state reset every 32 steps; that is fixed.
- With correct continuity, mixed curricula improve late-course accuracy and
  gradually tighten the full-start gate-3 miss, but zero-state segment starts
  still represent a different distribution from the full-start trajectory.
- A radius-2.0 full-start curriculum proves downstream credit is now available
  with one continuous policy state. Per-gate radii avoid weakening solved gates.
- The backend's unconditional reward clamp made large teacher/corridor weights
  effectively binary. Scaling the full shaped reward by 0.005 before the
  historical clamp retained magnitude information without destabilizing value
  learning and produced two consecutive accepted aperture reductions.
- The accept/reject runner turns the remaining shrink into a machine loop rather
  than a manual agent loop; one 2M native cycle takes about one minute.
- The official simulator should remain a promotion target, not a training loop.

## Exactly one next hypothesis

Continue the automated, reward-normalized gate-3 aperture anneal, accepting only
development success >=0.90 and crash <=0.10. Begin with the largest validated
step and automatically halve a failed local step down to the configured floor,
always retrying from the last accepted parent. After gate 3 reaches 1.5 m, run
the same retained-parent loop on gate 2 from 2.0 m to 1.5 m.
This changes training geometry only; deployed observations/actions remain fixed.

Do not spend an official attempt until a full-start checkpoint passes all four
ordered native gates and clears the 4096-episode promotion thresholds.

## Verification

```text
bash scripts/test_drone_race_native_regressions.sh
drone_race native regressions ok

.venv/bin/pytest -q \
  tests/test_train_full_policy_radius_curriculum.py \
  tests/test_drone_sitl_competition_smoke.py \
  tests/test_drone_policy_contract.py \
  tests/test_drone_camera_receiver.py \
  tests/test_drone_sitl_adapter.py \
  tests/test_drone_visual_servo.py
64 passed

git diff --check
clean
```

The CUDA extension rebuilt successfully for `drone_race` with CUDA 12.6 and
`sm_86` (one existing clang unknown-warning notice only).
