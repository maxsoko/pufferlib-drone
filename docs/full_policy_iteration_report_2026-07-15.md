# Full-Policy Iteration Report — 2026-07-15

## Outcome

The official-aperture recurrent policy is now precision-audited and retained at
transition floor `4.1875`. The active checkpoint is:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1875_best_mixed_calibration/rollrel0p99975.bin
```

Its exact-1024 deterministic report records success `0.900391`, gates
`3.865234`, crash `0.0`, Gate-2 crossing `0.964844`, terminal radial
`0.594761 m`, completion time `25.849455 s`, and native/checkpoint-layout
precision `4/4`.

Floor `4.1953125` and floor `4.21875` remain working boundaries, not retained
parents. No official-simulator run was launched.

## Precision and checkpoint-layout audit

PufferLib checkpoints always store FP32 master values, but their raw tensor
order follows the native arena's 16-byte alignment. BF16 tensors therefore
start on multiples of eight stored floats, while FP32 tensors start on multiples
of four. Loading a BF16-layout checkpoint directly into the FP32 backend does
not fail; it silently shifts every MinGRU tensor after `log_std`.

Implemented and verified:

- `scripts/convert_policy_checkpoint_layout.py` repacks BF16/FP32 checkpoint
  layouts without changing tensor values.
- `scripts/policy_callable_checkpoint.py` separates layout precision from
  arithmetic precision. Callable inference defaults to FP32 arithmetic.
- `scripts/eval_drone_race_checkpoint.py` requires an explicit compatible
  layout when requested and records both precision fields.
- `scripts/run_windows_full_policy.ps1` exports FP32 callable arithmetic and a
  selectable checkpoint-layout precision.
- `src/models.cu` and `src/pufferlib.cu` contain the CUDA compiler compatibility
  fixes needed by the FP32 build.

Authoritative build command:

```bash
PATH=/root/pufferlib-drone/.venv/bin:$PATH \
CUDA_HOME=/usr/local/cuda NVCC_ARCH=sm_86 \
bash build.sh drone_race --float
```

The loaded extension was checked with:

```bash
.venv/bin/python -c 'import pufferlib._C; print(pufferlib._C.precision_bytes)'
```

Expected and observed value: `4`.

The original BF16-layout parent is preserved at:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_round3_radial/drone_race_full_policy_official_fit/
1784120879975/0000000000163840.bin
```

Its correctly repacked FP32 behavior disproved the earlier apparent zero-gate
result. The direct cross-layout load was invalid evidence and must not be reused.

## Native curriculum change

The native environment now supports per-episode randomization of
`sitl_gate_transition_min_forward_speed` through these fields:

```text
sitl_gate_transition_min_forward_speed_randomize
sitl_gate_transition_min_forward_speed_min
sitl_gate_transition_min_forward_speed_max
```

The sampled value affects hidden dynamics only. It does not change the 32-value
observation contract. Fixed-floor evaluation is unchanged and remains the stage
promotion authority. The native regression
`test_transition_speed_curriculum_samples_per_episode` covers reset-time
sampling and the fixed-floor fallback.

## Promotion lineage

All rows below are deterministic exact-1024 screens at radius `0.75`, full
geometry, dropout `0`, no course/gate randomization, and FP32 native/layout
precision unless noted.

| Floor | Result | Retained mechanism |
|---:|---|---|
| `0` | success `0.998047`, gates `3.998047`, crash `0.0` | converted FP32 nominal parent |
| `3` | success `1.0`, crash `0.0` | fixed-floor PPO bridge |
| `3.5625` | success `>=0.90`, crash `0.0` | bounded PPO |
| `3.625` | success `>=0.90`, crash `0.0` | bounded PPO |
| `3.6875` | success `>=0.90`, crash `0.0` | bounded PPO |
| `4` | success `1.0`, gates `4.0`, crash `0.0` | pitch `0.8875`, relative roll `1.02` |
| `4.125` | success `0.942383`, gates `3.888672`, crash `0.0` | thrust `0.98`, relative roll `1.015` |
| `4.1875` | success `0.900391`, gates `3.865234`, crash `0.0` | mixed PPO, relative roll `0.99975` |

The retained floor-`4.1875` report is:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1875_best_mixed_calibration/
rollrel0p99975_exact1024.json
```

## Exact evaluator command contract

Every fixed-floor report in this iteration used this command shape, substituting
only checkpoint, output paths, label, and floor:

```bash
.venv/bin/python scripts/eval_drone_race_checkpoint.py CHECKPOINT.bin \
  --env-name drone_race_full_policy_official_fit \
  --eval-episodes 1024 --require-exact-episodes \
  --horizon 32 --max-rollouts 192 \
  --checkpoint-layout-precision-bytes 4 \
  --json-path REPORT.json --csv-path REPORT.csv --label LABEL \
  --env.num-gates 4 \
  --env.course-geometry-scale 1 \
  --env.course-geometry-scale-randomize 0 \
  --env.gate-radius 0.75 --env.gate-radius-randomize 0 \
  --env.sitl-gate-transition-min-forward-speed FLOOR \
  --env.sitl-gate-transition-min-forward-speed-randomize 0 \
  --env.sitl-gate-obs-dropout-from-index 1 \
  --env.sitl-gate-obs-dropout-range-m 0 \
  --env.gate-position-domain-randomize 0
```

No exact-4096 promotion was run because the trace-matched floor and dropout
stages are not solved.

## Floor 4.21875 boundary

The retained floor-`4.1875` parent scores only success `0.285156` at floor
`4.21875`. Thrust calibration restored Gate-2 throughput, and coupled
pitch/roll calibration plus one 32K mixed-floor PPO update raised the best child
to:

| Metric | Value |
|---|---:|
| success | `0.874023` |
| gates | `3.874023` |
| crash | `0.0` |
| Gate-2 crossing | `1.0` |
| terminal radial | `0.675329 m` |
| completion time | `24.996849 s` |

Checkpoint and report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p21875_postmixed_calibration/
rollp25_thrustrel1p00025.bin

logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p21875_postmixed_calibration/
rollp25_thrustrel1p00025_exact1024.json
```

This child is not retained because success is below `0.90`. A second PPO epoch
regressed to `0.865234`; pitch and finer roll/thrust sweeps did not promote it.

## Floor 4.1953125 fine boundary

The floor response is extremely sharp. The retained checkpoint scores success
`0.829102` at floor `4.19140625` and `0.751953` at floor `4.1953125`, despite
scoring `0.900391` at `4.1875`.

Thrust, pitch, and roll calibration followed by a 524K mixed-floor continuation
improved the best floor-`4.1953125` child to:

| Metric | Value |
|---|---:|
| success | `0.836914` |
| gates | `3.802734` |
| crash | `0.0` |
| Gate-2 crossing | `0.965820` |
| terminal radial | `0.577001 m` |
| completion time | `24.031668 s` |

Checkpoint and report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1953125_bestmixed_calibration/
rollp25_pitchm25_thrustrel0p9995.bin

logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1953125_bestmixed_calibration/
rollp25_pitchm25_thrustrel0p9995_exact1024.json
```

This child is not retained. A second 524K round increased Gate-2 crossing to
`0.985352` but reduced success to `0.800781`, proving that the sub-episode
objective was trading final accuracy for prefix throughput.

## PPO command and rejected reward branches

The successful mixed-PPO command family used:

```bash
.venv/bin/python -m pufferlib.pufferl train \
  drone_race_full_policy_official_fit \
  --load-model-path PARENT_LOGSTDM4.bin \
  --checkpoint-dir CHECKPOINT_DIR --log-dir LOG_DIR \
  --checkpoint-interval 2 --eval-episodes 0 \
  --train.total-timesteps 524288 \
  --train.learning-rate 0.0000005 \
  --train.anneal-lr 1 --train.min-lr-ratio 0.1 \
  --train.reward-scale 0.005 --train.reward-clip 1.0 \
  --train.ent-coef 0.0 \
  --env.invalid-penalty 100 \
  --env.w-cross-track 50 \
  --env.cross-track-vertical-weight 4 \
  --env.cross-track-from-gate-index 2 \
  --env.w-gate-crossing-error 25 \
  --env.gate-crossing-error-from-gate-index 2 \
  --env.sitl-gate-transition-min-forward-speed FLOOR_MAX \
  --env.sitl-gate-transition-min-forward-speed-randomize 1 \
  --env.sitl-gate-transition-min-forward-speed-min FLOOR_MIN \
  --env.sitl-gate-transition-min-forward-speed-max FLOOR_MAX
```

Rejected and not to be repeated unchanged:

- continuous speed-deficit reward;
- `w_gate_exit_velocity=50`;
- fixed-floor PPO from a calibrated action head;
- crossing-error weight `50` at the fine boundary;
- a second low-rate 32K epoch at floor `4.21875`;
- further BC, DAgger, phase-column, or phase-local decoder work;
- repeated pitch/roll/thrust coordinate sweeps around the documented children;
- treating an aggregate radial improvement as progress when failures move into
  an earlier gate.

## Training-noise and episode-horizon audit

The best floor-`4.1953125` mean policy was evaluated stochastically after only
changing checkpoint `log_std`:

| Training log-std | stochastic success | gates | Gate-2 crossing | crash |
|---:|---:|---:|---:|---:|
| `-6` | `0.993164` | `3.988281` | `0.995117` | `0.0` |
| `-8` | `0.600586` | `3.498047` | `0.897461` | `0.0` |

The `-6` report is:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1953125_stochastic_noise_audit/
logstdm6_stochastic_exact1024.json
```

This stochastic result is diagnostic, not deployable promotion evidence.
Deployment and native promotion still require deterministic mean actions.

The audit also found that `524288 / 1024 = 512` steps per agent, while a full
course takes about 1,700 steps. All 524K branches therefore ended before any
episode reached Gate 3. Late-reward changes produced byte-identical checkpoints
because those rewards were never sampled. This explains why repeating short
late-reward branches was circular.

The first episode-complete continuation used this exact training delta:

```bash
# Same command family as above, with:
--load-model-path \
  logs/drone_race_full_policy_official_fit/official_speed_curriculum/\
ppo_r075_v4_fp32_floor4p1953125_stochastic_noise_audit/logstdm6.bin \
--train.total-timesteps 2097152 \
--env.sitl-gate-transition-min-forward-speed-min 4.1875 \
--env.sitl-gate-transition-min-forward-speed-max 4.1953125
```

It completed in about `45.2 s`. Training logs at the first completed episode
show gates `3.0`, Gate-2 crossing `1.0`, crash `0.0`, and final radial settling
near `0.924109 m`. It saved checkpoints through step `2097152` at:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1875_to4p1953125_logstdm6_episodecomplete_2m/
drone_race_full_policy_official_fit/1784171013502/
```

The four post-terminal deterministic exact-1024 screens rejected this branch:

| Step | success | gates | Gate-2 crossing | crash | terminal radial |
|---:|---:|---:|---:|---:|---:|
| `1671168` | `0.479492` | `3.479492` | `1.0` | `0.0` | `0.704531 m` |
| `1835008` | `0.522461` | `3.522461` | `1.0` | `0.0` | `0.693091 m` |
| `1998848` | `0.503906` | `3.503906` | `1.0` | `0.0` | `0.691170 m` |
| `2097152` | `0.518555` | `3.518555` | `1.0` | `0.0` | `0.686873 m` |

Reports are under:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1875_to4p1953125_logstdm6_episodecomplete_2m_exact1024/
```

The branch learned a much faster mean line (reported completion times about
`13.7–15.0 s`) but lost final-gate reliability. It is not retained.

### Fixed-floor isolation

The predeclared isolation repeated the episode-complete continuation at fixed
floor `4.1953125`; floor randomization was disabled and all other policy,
reward, optimizer, geometry, and observation settings were unchanged:

```bash
.venv/bin/python -m pufferlib.pufferl train \
  drone_race_full_policy_official_fit \
  --load-model-path \
    logs/drone_race_full_policy_official_fit/official_speed_curriculum/\
ppo_r075_v4_fp32_floor4p1953125_stochastic_noise_audit/logstdm6.bin \
  --checkpoint-dir \
    logs/drone_race_full_policy_official_fit/official_speed_curriculum/\
ppo_r075_v4_fp32_floor4p1953125_fixed_logstdm6_episodecomplete_2m \
  --log-dir \
    logs/drone_race_full_policy_official_fit/official_speed_curriculum/\
ppo_r075_v4_fp32_floor4p1953125_fixed_logstdm6_episodecomplete_2m_train_logs \
  --checkpoint-interval 5 --eval-episodes 0 \
  --train.total-timesteps 2097152 \
  --train.learning-rate 0.0000005 \
  --train.anneal-lr 1 --train.min-lr-ratio 0.1 \
  --train.reward-scale 0.005 --train.reward-clip 1.0 \
  --train.ent-coef 0.0 \
  --env.invalid-penalty 100 \
  --env.w-cross-track 50 --env.cross-track-vertical-weight 4 \
  --env.cross-track-from-gate-index 2 \
  --env.w-gate-crossing-error 25 \
  --env.gate-crossing-error-from-gate-index 2 \
  --env.sitl-gate-transition-min-forward-speed 4.1953125 \
  --env.sitl-gate-transition-min-forward-speed-randomize 0
```

It completed 64 training epochs in about `46.5 s`. Its terminal training log
recorded gates `3.0`, crash `0.0`, success `0.0`, and closest final-gate range
`0.923551 m`. The four post-terminal deterministic exact-1024 screens also
rejected the branch:

| Step | success | gates | Gate-2 crossing | crash | terminal radial | completion time |
|---:|---:|---:|---:|---:|---:|---:|
| `1671168` | `0.500977` | `3.500977` | `1.0` | `0.0` | `0.713189 m` | `14.351685 s` |
| `1835008` | `0.497070` | `3.497070` | `1.0` | `0.0` | `0.714640 m` | `14.239290 s` |
| `1998848` | `0.498047` | `3.498047` | `1.0` | `0.0` | `0.712307 m` | `14.269075 s` |
| `2097152` | `0.506836` | `3.506836` | `1.0` | `0.0` | `0.709995 m` | `14.519684 s` |

Reports are under:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1953125_fixed_logstdm6_episodecomplete_2m_exact1024/
```

Fixed versus randomized floor selection is therefore not the cause. Both
episode-complete runs applied horizon-`32` PPO updates throughout the flight.
The measured `0.993164` stochastic parent was changed more than 50 times before
the first successful terminal sample could influence training, explaining why
both branches lost the successful behavior before learning from it.

### Delayed final-approach update

A generic `train.learning_start_timesteps` control was added with a historical
default of zero. Native rollouts and recurrent state continue during the delay;
only optimizer calls are withheld. Four focused tests cover the default,
threshold boundary, agent-step semantics, and negative-value rejection. The
warmup also exposed a reporting assumption that every training row has loss
fields; the reducer now omits pre-update rows from its homogeneous loss series.

The isolated run added these deltas to the fixed-floor command above:

```bash
--checkpoint-dir \
  logs/drone_race_full_policy_official_fit/official_speed_curriculum/\
ppo_r075_v4_fp32_floor4p1953125_fixed_logstdm6_delayed1600_2m \
--log-dir \
  logs/drone_race_full_policy_official_fit/official_speed_curriculum/\
ppo_r075_v4_fp32_floor4p1953125_fixed_logstdm6_delayed1600_2m_train_logs \
--checkpoint-interval 1 \
--train.learning-start-timesteps 1638400
```

The total budget remained `2097152`; all parent, horizon, floor, reward,
optimizer, geometry, and observation fields remained unchanged. SHA-256 proves
that step `1605632` is byte-identical to the input parent
(`b74837e37d7182e7304171f4059e6c766b48a323b9fd1e7cf2a3ecb12162d532`).
Step `1638400` is the first changed checkpoint. The final evaluation-only step
`2129920` is byte-identical to training step `2097152` and was not rescreened.

All 15 unique updated children were deterministic exact-1024 screened:

| Step | success | gates | Gate-2 crossing | crash | terminal radial | completion time |
|---:|---:|---:|---:|---:|---:|---:|
| `1638400` | `0.792969` | `3.758789` | `0.965820` | `0.0` | `0.590453 m` | `22.768784 s` |
| `1671168` | `0.735352` | `3.700195` | `0.964844` | `0.0` | `0.615933 m` | `21.097765 s` |
| `1703936` | `0.720703` | `3.670898` | `0.950195` | `0.0` | `0.613738 m` | `20.681612 s` |
| `1736704` | `0.722656` | `3.662109` | `0.939453` | `0.0` | `0.616426 m` | `20.733038 s` |
| `1769472` | `0.771484` | `3.710938` | `0.939453` | `0.0` | `0.604735 m` | `22.135801 s` |
| `1802240` | `0.752930` | `3.682617` | `0.929688` | `0.0` | `0.608903 m` | `21.604296 s` |
| `1835008` | `0.729492` | `3.635742` | `0.906250` | `0.0` | `0.607265 m` | `20.938053 s` |
| `1867776` | `0.739258` | `3.619141` | `0.879883` | `0.0` | `0.599347 m` | `21.225189 s` |
| `1900544` | `0.662109` | `3.479492` | `0.817383` | `0.0` | `0.623689 m` | `19.005928 s` |
| `1933312` | `0.660156` | `3.447266` | `0.787109` | `0.0` | `0.621631 m` | `18.955353 s` |
| `1966080` | `0.602539` | `3.330078` | `0.727539` | `0.0` | `0.632855 m` | `17.300880 s` |
| `1998848` | `0.534180` | `3.205078` | `0.670898` | `0.0` | `0.641282 m` | `15.343067 s` |
| `2031616` | `0.482422` | `3.092773` | `0.610352` | `0.0` | `0.655417 m` | `13.858059 s` |
| `2064384` | `0.416016` | `2.973633` | `0.557617` | `0.0` | `0.671885 m` | `11.946089 s` |
| `2097152` | `0.306641` | `2.738281` | `0.431641` | `0.0` | `0.697646 m` | `8.807176 s` |

Reports are under:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1953125_fixed_logstdm6_delayed1600_2m_exact1024/
```

The first delayed epoch still applies four Muon minibatch steps at learning
rate `5e-7`. It immediately reduces deterministic success from the parent's
`0.836914` to `0.792969`; subsequent updates eventually lose Gate-2 throughput.
The timing hypothesis is rejected, and the evidence now points to update
magnitude as the narrower variable.

### Single micro-update

The magnitude isolation used exactly one final-approach minibatch update:

```bash
--train.total-timesteps 1638400 \
--train.learning-start-timesteps 1638400 \
--train.learning-rate 0.00000005 \
--train.replay-ratio 0.25
```

The first attempt under
`ppo_r075_v4_fp32_floor4p1953125_fixed_logstdm6_delayed1600_micro40x`
did not reach the update. An evaluation-stop `elif` was accidentally reachable
during a warmup training epoch after early episode failures appeared; the log
ended at step `1310720` and only an unchanged step-`32768` checkpoint was saved.
That control-flow bug was fixed, a fifth regression test was added, and the
unexecuted hypothesis was rerun under the `_v2` suffix.

The valid run completed exactly one optimizer epoch and saved:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1953125_fixed_logstdm6_delayed1600_micro40x_v2/
drone_race_full_policy_official_fit/1784172998406/0000000001638400.bin
```

Its deterministic exact-1024 result is:

| Metric | Value |
|---|---:|
| success | `0.831055` |
| gates | `3.797852` |
| crash | `0.0` |
| Gate-2 crossing | `0.966797` |
| terminal radial | `0.576288 m` |
| completion time | `23.865852 s` |

The adjacent report is `step1638400_exact1024.json`. Success is below the
unchanged parent's `0.836914`, so the 40-fold smaller update still moves in the
wrong direction. Further learning-rate or replay-ratio tuning on this replay
path is rejected as circular.

### Terminal-aware recurrent replay

The native replay path previously discarded recurrent rows containing a
terminal when clean rows existed, and could fall back to invalid post-reset
hidden state when every row contained a terminal. It now:

- zeros the selected initial state when `terminal[0]` marks a reset before the
  first action;
- assigns replay priority only to the prefix before the first later terminal;
- normalizes advantages only over valid prefix tokens; and
- zeros policy/value gradients and preserves ratio/value entries throughout the
  invalid post-reset suffix.

Terminal-free rows retain the old traversal and reduction order. Verification:

```text
12 focused recurrent-replay/learning-start tests passed
FP32 CUDA backend build passed; precision_bytes=4
drone_race native regressions passed
```

The bounded learning test added these deltas to the fixed-floor command:

```bash
--checkpoint-dir \
  logs/drone_race_full_policy_official_fit/official_speed_curriculum/\
ppo_r075_v4_fp32_floor4p1953125_terminalaware1728_micro \
--train.total-timesteps 1769472 \
--train.learning-start-timesteps 1769472 \
--train.learning-rate 0.00000005 \
--train.replay-ratio 0.25
```

This preserves the parent through 1,696 steps per agent and performs one update
on the rollout spanning steps 1,696–1,728, where course-end terminals occur.
Checkpoint and report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1953125_terminalaware1728_micro/
drone_race_full_policy_official_fit/1784173511457/0000000001769472.bin

logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1953125_terminalaware1728_micro/
step1769472_exact1024.json
```

| Metric | Value |
|---|---:|
| success | `0.833984` |
| gates | `3.799805` |
| crash | `0.0` |
| Gate-2 crossing | `0.965820` |
| terminal radial | `0.577667 m` |
| completion time | `23.947830 s` |

This remains below the deterministic parent's `0.836914`. Correct terminal
credit does not make the stochastic PPO update improve the deterministic
objective, so further positive updates are rejected.

### Opposite-gradient line search

The parent and terminal-aware child were full-checkpoint blended with
`output = parent + alpha * (child - parent)` and exact-screened at floor
`4.1953125`:

| Alpha | success | crash |
|---:|---:|---:|
| `-1` | `0.835938` | `0.0` |
| `-2` | `0.831055` | `0.0` |
| `-4` | `0.833008` | `0.0` |
| `-8` | `0.824219` | `0.0` |

Artifacts and reports are under:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1953125_terminalaware_opposite_line/
```

Neither sign improves the `0.836914` base. The terminal-aware PPO direction is
closed rather than subject to another scale or learning-rate sweep.

### Harder-child cross-screen

The strongest existing floor-`4.21875` child had never been evaluated at the
intermediate floor `4.1953125`. Its read-only cross-screen produced:

| Metric | Value |
|---|---:|
| success | `0.601562` |
| gates | `3.601562` |
| crash | `0.0` |
| Gate-2 crossing | `1.0` |
| terminal radial | `0.698102 m` |
| completion time | `17.252958 s` |

Report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p21875_postmixed_calibration/
rollp25_thrustrel1p00025_floor4p1953125_exact1024.json
```

This child scores `0.874023` at the higher `4.21875` floor, proving its response
is a narrow non-monotonic dynamics pocket rather than a generally stronger
policy.

### Harder-child local floor map

The predeclared read-only four-point map around `4.21875` is complete:

| Floor | success | gates | crash | Gate-2 crossing | terminal radial | completion time |
|---:|---:|---:|---:|---:|---:|---:|
| `4.21484375` | `0.783203` | `3.783203` | `0.0` | `1.0` | `0.678190 m` | `22.409542 s` |
| `4.22265625` | `0.830078` | `3.830078` | `0.0` | `1.0` | `0.679006 m` | `23.736542 s` |
| `4.2265625` | `0.864258` | `3.864258` | `0.0` | `1.0` | `0.660990 m` | `24.717661 s` |
| `4.23046875` | `0.828125` | `3.828125` | `0.0` | `1.0` | `0.678107 m` | `23.669590 s` |

Reports are beside the child checkpoint and use the suffixes
`floor4p21484375_exact1024.json`, `floor4p22265625_exact1024.json`,
`floor4p2265625_exact1024.json`, and `floor4p23046875_exact1024.json`.

None reaches the `0.90` retention threshold. The strongest result remains the
known `0.874023` exactly at floor `4.21875`; this local checkpoint family is
closed, including further floor points and calibration variants.

### Floor 6 parent baselines

The retained mean and strongest closed higher-floor mean were both screened at
the required fixed floor `6`:

| Checkpoint | success | gates | crash | Gate-2 crossing | missed gate | terminal radial | closest range | final progress |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| retained floor `4.1875` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `1.067169 m` | `55.890087 m` | `0.434597` |
| closed floor `4.21875` child | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `1.140901 m` | `55.963676 m` | `0.433852` |

Both pass Gate 0 at about `3.8 m/s`, receive the floor-`6` post-gate transition,
and deterministically miss Gate 1 without crashing. The stronger local `4.21875`
child provides no transfer benefit at floor `6`; this parent comparison is
closed. Reports are the adjacent `rollrel0p99975_floor6_exact1024.json` and
`rollp25_thrustrel1p00025_floor6_exact1024.json` files.

### Floor 6 exploration feasibility

Changing only the retained checkpoint's Gaussian action `log_std` from about
`-4` to `-6` and sampling actions at floor `6` produced:

| Metric | deterministic mean | stochastic `log_std=-6` |
|---|---:|---:|
| success | `0.0` | `0.0` |
| gates | `1.0` | `1.0` |
| crash | `0.0` | `0.0` |
| Gate-2 crossing | `0.0` | `0.0` |
| missed gate | `1.0` | `1.0` |
| terminal radial | `1.067169 m` | `1.079920 m` |
| signed right / vertical | `-0.575777 / +0.898483 m` | `-0.575259 / +0.913737 m` |
| final progress | `0.434597` | `0.434463` |

Report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p1875_best_mixed_calibration/
rollrel0p99975_logstdm6_floor6_stochastic_exact1024.json
```

The diagnostic supplies no recovery trajectory and worsens radial error.
Stochastic PPO at floor `6` is closed.

Older artifacts also show that the earlier floor-`4` calibrated
`pitch0p8875_rollrel1p02.bin` mean passed only one gate at floor `6` with radial
`1.038140 m`; global thrust factors `0.9` and `0.8` worsened radial to
`1.285521` and `1.511626 m`. Those reports are in
`ppo_r075_v4_fp32_floor4_pitch_calibration/floor6_exact1024.json` and
`ppo_r075_v4_fp32_floor6_thrust_calibration/`. This closes global action scaling
at floor `6` and indexes the older evidence so it is not rediscovered.

### Gate-1 telemetry and log-capacity correction

The first paired telemetry attempt was operationally invalid. Ten new log
fields exceeded the release binding's fixed 96-entry dictionary; because the
capacity assertion is absent under `NDEBUG`, both evaluations corrupted the
heap and aborted. No checkpoint result was inferred from those processes. Their
stderr artifacts are preserved as:

```text
rollrel0p99975_floor4p1875_gate1telemetry_capacity96_failed.log
rollrel0p99975_floor6_gate1telemetry_capacity96_failed.log
```

`Dict` now grows safely when full, the drone binding begins with 128 entries,
and native regressions cover dictionary growth plus the first-60-step action
window. The FP32 backend rebuilt successfully, reported precision `4`, and an
exact-8 smoke emitted all derived Gate-1 fields before the exact rerun.

The unchanged retained checkpoint then produced:

| Gate-1 metric | floor `4.1875` | floor `6` |
|---|---:|---:|
| phase steps | `367.019531` | `263.257812` |
| early-60 pitch | `-0.038505` | `-0.034572` |
| early-60 roll | `-0.399561` | `-0.400115` |
| early-60 thrust | `-0.264449` | `-0.250813` |
| early-60 yaw | `+0.000182` | `+0.000182` |
| full-phase pitch | `-0.025072` | `-0.017053` |
| full-phase roll | `-0.195158` | `-0.309127` |
| full-phase thrust | `-0.087465` | `-0.109892` |
| full-phase yaw | `+0.000174` | `+0.000173` |

The authoritative policy metrics remain bit-for-bit consistent with the earlier
reports: floor `4.1875` success `0.900391`, gates `3.865234`, crash `0.0`; floor
`6` success `0.0`, gates `1.0`, crash `0.0`, radial `1.067169 m`. The first
second of post-transition control is nearly unchanged even though floor `6`
ends with signed Gate-1 vertical error `+0.898483 m`. This selects insufficient
response to camera-derived gate down-rate (`observation[2]`) as the next bounded
parameter subspace, rather than another output-bias search.

### Gate down-rate encoder screen

The layout-aware utility `scripts/scale_policy_encoder_feature.py` verifies that
only one declared FP32 encoder column changes. Scaling camera-derived gate
down-rate column `2` produced:

| Factor | success | gates | crash | Gate-2 crossing | terminal radial | signed vertical | early thrust |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1.25` | `0.0` | `1.0` | `0.0` | `0.0` | `1.189951 m` | `+1.049787 m` | `-0.249639` |
| `1.5` | `0.0` | `1.0` | `0.0` | `0.0` | `1.373555 m` | `+1.250921 m` | `-0.246873` |
| `2.0` | `0.0` | `1.0` | `0.0` | `0.0` | `1.678409 m` | `+1.584536 m` | `-0.240749` |

The unscaled baseline radial is `1.067169 m` with early thrust `-0.250813`.
Every factor worsens the miss and none changes a discrete outcome. The
predeclared rule therefore closes this encoder-column family; an inverse point
will not be added after seeing the result.

### Deterministic final-MinGRU ES generation

`scripts/evolve_policy_tensor.py` generates antithetic FP32 probes and verifies
that only the declared final MinGRU tensor changes. The first shell launch
failed before generation because its manifest redirection targeted a directory
that did not yet exist. No checkpoint or report was produced; after explicit
directory creation, the unchanged predeclared generation ran.

All reports below are deterministic exact-1024 floor-`6` screens; gates are
`1.0`, success and crash are `0.0`, and Gate-2 crossing is `0.0` throughout:

| Probe | terminal radial |
|---|---:|
| direction 0 `+` / `-` | `1.067160 / 1.067211 m` |
| direction 1 `+` / `-` | `1.067245 / 1.067162 m` |
| direction 2 `+` / `-` | `1.067178 / 1.067238 m` |
| direction 3 `+` / `-` | `1.067235 / 1.067183 m` |
| fitness direction `+` / `-` | `1.067186 / 1.067236 m` |

The actual antithetic L2 radii are `1.041–1.044e-6`; the combined direction is
`1.016e-6`. The best change is only about `0.000009 m`, so no probe reaches the
predeclared `0.10 m` or discrete-outcome rule. Deterministic local parameter
search around this retained mean is closed.

### Transition-floor fidelity audit

The native floor currently assigns world-X velocity instantaneously after a
gate crossing. At that same official gate-identity change, both native and live
camera-motion filters correctly reset their new target's rate estimate to zero.
The speed discontinuity is therefore initially hidden from the policy and is
not the physically controllable annealing mechanism required by the execution
goal. This explains the sharp non-monotonic boundary and why local policy
perturbations do not bridge floor `6`. Legacy results remain valid only for the
instantaneous setting and will not be relabeled.

The one predeclared physical-ramp screen at `2.0 m/s²` did not improve the
policy:

| Metric | Value |
|---|---:|
| success | `0.0` |
| gates | `1.0` |
| crash | `0.0` |
| Gate-2 crossing | `0.0` |
| terminal radial | `1.560399 m` |
| signed right / vertical | `-0.850784 / +1.308030 m` |
| Gate-1 phase steps | `228.142578` |

The assist combines with the policy's own acceleration and shortens Gate 1
further. The `2.0 m/s²` hypothesis is closed; no other rate is authorized from
this parent.

The code comment and v3385 fit describe Gate-2 closing speed after Gate 1, but
the implementation applies the floor immediately after Gate 0 too. Thus every
floor-`6` result so far terminates at Gate 1 under a speed boost not justified by
the stated trace target. This scope mismatch must be tested before more policy
optimization.

### Corrected floor activation scope

`sitl_gate_transition_from_gate_index` now defaults to `1`, preserving every
legacy report. Setting it to `2` delays the floor until Gate 1 has been passed,
matching the stated Gate-2 closing-speed purpose. With the ramp disabled, the
unchanged retained checkpoint at floor `6` produces:

| Metric | Value |
|---|---:|
| success | `0.0` |
| gates | `2.0` |
| crash | `0.0` |
| Gate-2 terminal crossing | `1.0` |
| Gate-2 radial | `2.203115 m` |
| signed right / vertical | `+1.725223 / +1.370109 m` |
| final progress | `0.675365` |
| episode length | `1187.101562` steps |

Report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor6_from_gate2/retained_floor6_fromgate2_exact1024.json
```

This passes the discrete continuation rule by moving failure from Gate 1 to
Gate 2 without crashes. It is not promotion, but it is now the active,
trace-scoped floor-`6` failure signature.

### Corrected-scope terminal-aware micro-update

The single authorized update preserved the `log_std=-6` parent through 1184
steps per agent and updated only the `1184–1216` rollout:

```text
total_timesteps = learning_start_timesteps = 1245184
learning_rate = 5e-8
replay_ratio = 0.25
activation index = 2
ramp = 0
```

Training completed one optimizer epoch in `22.882400 s` at `43806` SPS, with
policy loss `-0.005416`, value loss `0.116870`, and KL `8.05e-8`. Step
`1245184` is the only changed checkpoint; the final evaluation-only step
`1277952` has the same SHA-256
`454accb08dc93f92f4ed9c51ff21199cb5080a9ddacaaf94ddae48a723ccd628`.

Its exact-1024 result is:

| Metric | Parent | Child |
|---|---:|---:|
| success | `0.0` | `0.0` |
| gates | `2.0` | `2.0` |
| crash | `0.0` | `0.0` |
| Gate-2 terminal crossing | `1.0` | `1.0` |
| Gate-2 radial | `2.203115 m` | `2.203145 m` |

Report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor6_fromgate2_terminalaware1216_micro/
step1245184_exact1024.json
```

The child is slightly worse. Corrected-scope PPO is closed; no second update,
opposite line, learning-rate change, or replay-ratio branch is authorized.

### Corrected-scope coarse boundary

The two predeclared non-training anchors are complete with the unchanged
retained checkpoint, activation index `2`, and ramp `0`:

| Metric | floor `4` | floor `5` |
|---|---:|---:|
| success | `1.0` | `0.0` |
| gates | `4.0` | `2.0` |
| crash | `0.0` | `0.0` |
| Gate-2 crossing / terminal crossing | `1.0 / 0.0` | `0.0 / 1.0` |
| Gate-2 radial | `0.531454 m` | `1.398045 m` |
| signed right / vertical | `+0.326100 / +0.419565 m` | `+1.028751 / +0.946609 m` |
| completion time | `29.394688 s` | `0.0 s` |
| final progress | `1.0` | `0.683466` |

Reports:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_corrected_scope_coarse/
retained_floor4_fromgate2_exact1024.json
retained_floor5_fromgate2_exact1024.json
```

The checkpoint passes the exact-1024 retention contract at corrected-scope
floor `4`. Floor `5` is the first failing coarse anchor and moves the active
failure to Gate 2 without crashes. Both required anchors were recorded before
selecting another hypothesis; no intermediate floor was added.

### Corrected-scope physical-ramp screen

The one authorized `2.0 m/s²` screen at floor `5` produced:

| Metric | instant baseline | physical ramp |
|---|---:|---:|
| success | `0.0` | `0.0` |
| gates | `2.0` | `2.0` |
| crash | `0.0` | `0.0` |
| Gate-2 terminal crossing | `1.0` | `1.0` |
| Gate-2 radial | `1.398045 m` | `1.657418 m` |
| signed right / vertical | `+1.028751 / +0.946609 m` | `+1.283240 / +1.048919 m` |
| final progress | `0.683466` | `0.680835` |

Report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor5_fromgate2_ramp2/
retained_floor5_fromgate2_ramp2_exact1024.json
```

The ramp worsens radial by `0.259373 m`, changes no discrete outcome, and fails
the predeclared continuation rule. Combined with the legacy-scope rejection,
this closes physical transition ramps; no second rate is authorized.

### Direction-preserving speed-floor screen

The zero-default implementation scales the complete current velocity just
enough for its Gate-normal component to reach the floor. Native regressions and
the FP32 build pass, and the loaded backend reports precision `4`.

The sole corrected-scope floor-`5` screen produced:

| Metric | world-X baseline | preserve direction |
|---|---:|---:|
| success | `0.0` | `0.0` |
| gates | `2.0` | `2.0` |
| crash | `0.0` | `0.0` |
| Gate-2 terminal crossing | `1.0` | `1.0` |
| Gate-2 radial | `1.398045 m` | `1.984565 m` |
| signed right / vertical | `+1.028751 / +0.946609 m` | `+1.672432 / +1.068225 m` |
| final progress | `0.683466` | `0.677542` |

Report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor5_fromgate2_preserve_velocity_direction/
retained_floor5_fromgate2_preserve_velocity_direction_exact1024.json
```

Radial worsens by `0.586519 m`, so the hypothesis fails its continuation rule.
No blended or segment-tangent direction variant is authorized from this result;
the active curriculum retains the legacy direction.

### Corrected-scope midpoint

The sole authorized midpoint screen at floor `4.5` produced:

| Metric | Value |
|---|---:|
| success | `0.0` |
| gates | `2.0` |
| crash | `0.0` |
| Gate-2 terminal crossing | `1.0` |
| Gate-2 radial | `0.980955 m` |
| signed right / vertical | `+0.664541 / +0.721489 m` |
| final progress | `0.687664` |

Report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_corrected_scope_midpoint/
retained_floor4p5_fromgate2_exact1024.json
```

Radial improves by `0.417091 m` relative to floor `5`, but the checkpoint does
not meet retention and still misses the `0.75 m` aperture by `0.230955 m`. The
predeclared stop applies: no finer deterministic floor point is authorized
before a new training mechanism.

### Corrected-scope stochastic feasibility

The existing retained mean with `log_std=-6` was sampled once at floor `4.5`:

| Metric | deterministic mean | stochastic `-6` |
|---|---:|---:|
| success | `0.0` | `0.0` |
| gates | `2.0` | `2.0` |
| crash | `0.0` | `0.0` |
| Gate-2 crossing | `0.0` | `0.0` |
| terminal radial | `0.980955 m` | `0.871013 m` |
| signed right / vertical | `+0.664541 / +0.721489 m` | `+0.579753 / +0.649861 m` |
| Gate-1 phase steps | `389.099609` | `400.351562` |

Report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p5_fromgate2_stochastic_feasibility/
retained_logstdm6_floor4p5_fromgate2_stochastic_exact1024.json
```

Radial improves by `0.109942 m`, satisfying the metric continuation threshold,
but no episode crosses Gate 2. The stochastic policy cannot promote and this
does not reopen PPO. It does show that the incoming transition state and timing
matter; no second log standard deviation is authorized.

### One-step transition-delay screen

The zero-default delay associates the new gate at the crossing and applies the
unchanged impulse at the start of the next control step. Native regressions and
the FP32 rebuild pass. The sole corrected-scope floor-`4.5` screen is
behaviorally identical to the no-delay baseline:

| Metric | delay `0` | delay `1` |
|---|---:|---:|
| success | `0.0` | `0.0` |
| gates | `2.0` | `2.0` |
| crash | `0.0` | `0.0` |
| Gate-2 radial | `0.980955 m` | `0.980955 m` |
| signed right / vertical | `+0.664541 / +0.721489 m` | `+0.664541 / +0.721489 m` |
| final progress | `0.687664` | `0.687664` |

Report:

```text
logs/drone_race_full_policy_official_fit/official_speed_curriculum/
ppo_r075_v4_fp32_floor4p5_fromgate2_delay1/
retained_floor4p5_fromgate2_delay1_exact1024.json
```

The next physics integration sees the same velocity in both schedules, so the
following gate-motion sample is already equivalent. The delay hypothesis is
closed; no other delay length is authorized.

## Current retained state and next hypothesis

Retain the unchanged `rollrel0p99975.bin` checkpoint at corrected-scope floor
`4`. Do not promote any floor-`4.1953125` or floor-`4.21875` child listed above.

The corrected-scope physical-ramp result is rejected. Corrected-scope PPO,
local parameter search, gate down-rate scaling, and global action calibration
are also closed.

The physical ramp and direction-preserving floor are rejected; use the legacy
instantaneous direction for the active curriculum.

Floor `4.5` is the active near-boundary failure, and no finer deterministic
floor screen is authorized.

Ramp, direction, and delay variants of the transition floor are closed.

Exactly one next hypothesis: reconcile the stale 32-field phase-observation
schema before selecting another optimizer. Native and live runtime agree that
columns `28/29` are Gate-2 down/down-rate and `30/31` are Gate-3
right/right-rate, while the metadata tuple and legacy dataset expansion still
describe the older Gate-3-heavy layout. Treat runtime parity as authoritative,
correct only the stale tooling and tests, and run no policy evaluation because
the runtime vector and checkpoint behavior remain unchanged.

The official simulator remains gated on floors `6`, `8`, and `10`, dropout, and
an exact-4096 native promotion. It has not been run during this iteration.

### Jacobian-CMA baseline lock (2026-07-16)

The stale phase-observation tooling is reconciled. Runtime, live metadata, and
dataset expansion now agree on columns `23–31`, including Gate-2 down/down-rate
at `28/29` and Gate-3 right/right-rate at `30/31`; the focused schema/smoke
suite passed `28` tests. This changes no runtime observation or checkpoint.

Before any mutated checkpoint, the retained parent SHA-256 was verified as
`39e51c90d71775bf1f2986236f500e27a430755b56063f806d3a1c28f15bdaf3`, and the
loaded backend reported native precision `4`. Both corrected-scope baselines
were then reproduced with the authoritative evaluator.

Common command prefix and fixed overrides:

```text
./.venv/bin/python scripts/eval_drone_race_checkpoint.py \
  logs/drone_race_full_policy_official_fit/official_speed_curriculum/ppo_r075_v4_fp32_floor4p1875_best_mixed_calibration/rollrel0p99975.bin \
  --env-name drone_race_full_policy_official_fit \
  --eval-episodes 1024 --require-exact-episodes \
  --checkpoint-layout-precision-bytes 4 --max-rollouts 192 --horizon 32 \
  --env.num-gates 4 --env.course-geometry-scale 1 \
  --env.course-geometry-scale-randomize 0 --env.gate-radius 0.75 \
  --env.gate-radius-randomize 0 \
  --env.sitl-gate-transition-min-forward-speed-randomize 0 \
  --env.sitl-gate-transition-from-gate-index 2 \
  --env.sitl-gate-transition-max-accel-m-s2 0 \
  --env.sitl-gate-transition-preserve-velocity-direction 0 \
  --env.sitl-gate-transition-delay-steps 0 \
  --env.sitl-gate-obs-dropout-from-index 1 \
  --env.sitl-gate-obs-dropout-range-m 0 \
  --env.gate-position-domain-randomize 0
```

The floor-`4` command added
`--env.sitl-gate-transition-min-forward-speed 4`, label
`jacobian_cma_parent_floor4_exact1024`, and JSON/CSV paths under
`logs/drone_race_full_policy_official_fit/jacobian_cma/baselines/`. It completed
in `32.235068 s`: success `1.0`, gates `4.0`, crash `0.0`, Gate-2 crossing
`1.0`, radial `0.531454 m`, and completion `29.394688 s`.

The floor-`4.5` command used the identical command with floor `4.5`, label
`jacobian_cma_parent_floor4p5_exact1024`, and its matching output paths. It
completed in `23.180954 s`: success `0.0`, gates `2.0`, crash `0.0`, Gate-2
crossing `0.0`, terminal crossing `1.0`, radial `0.980955 m`, signed
right/vertical `+0.664541/+0.721489 m`, and final progress `0.687664`.

Both results exactly reproduce their historical metrics. Native/layout
precision is `4/4`, no collision or invalid action occurred, and no official
simulator run was made.

The new branch is pre-registered in `config/jacobian_cma_experiment.json`:
Jacobian-guided constrained CMA-ES, dimension `12`, population `24`, parents
`12`, seed `3385`, and at most `8` generations. It uses persistent full
covariance, antithetic sampling, floor-`4` preservation, fixed development and
held-out episode sets, and at most one spectrum-justified expansion to dimension
`24`. No hyperparameter grid or seed retry is authorized.

Exactly one next hypothesis: a recurrent action-Jacobian subspace spanning the
encoder, all MinGRU projections, and policy decoder can expose coordinated
floor-`4.5` corrections that one-tensor ES could not reach, while anchor
whitening and an actual floor-`4` constraint preserve the solved prefix/course.

### Native traces and pre-outcome subspace calibration

The FP32 backend was rebuilt after adding a read-only `rollout_trace` binding:

```text
PATH=/root/pufferlib-drone/.venv/bin:$PATH \
CUDA_HOME=/usr/local/cuda NVCC_ARCH=sm_86 \
bash build.sh drone_race --float
bash scripts/test_drone_race_native_regressions.sh
```

The build reports precision `4`; native regressions pass. The binding copies
exact native rollout observations, deterministic mean actions, rewards,
terminals, and exact rollout-start MinGRU states. It changes no policy or
environment behavior. The focused trace/callable test set initially passed
`11` tests.

Two eight-agent, full-history traces were captured with the retained checkpoint,
horizon `32`, episode offset `0`, and the same corrected-scope configuration as
the baselines:

```text
./.venv/bin/python scripts/capture_native_policy_trace.py <parent> \
  logs/drone_race_full_policy_official_fit/jacobian_cma/traces/parent_floor4_trace.npz \
  --floor 4 --total-agents 8 --horizon 32 --episode-offset 0 \
  --max-rollouts 64 --layout-precision-bytes 4

./.venv/bin/python scripts/capture_native_policy_trace.py <parent> \
  logs/drone_race_full_policy_official_fit/jacobian_cma/traces/parent_floor4p5_trace.npz \
  --floor 4.5 --total-agents 8 --horizon 32 --episode-offset 0 \
  --max-rollouts 64 --layout-precision-bytes 4
```

The floor-`4` trace has SHA-256
`58f7d55053aa0b5e785ab61466a5b380e5f994e0a29fefc927d404e11f2a39a5` and
records success `1.0`, gates `4.0`, crash `0.0`, and Gate-2 radial
`0.525431 m` on its fixed eight-agent subset. The floor-`4.5` trace has SHA-256
`ffd3c901a8830a2e0a1bea823e5b7443f96cff94d6904ea85a076f933e5a217b` and
records success `0.0`, gates `2.0`, crash `0.0`, and Gate-2 terminal radial
`0.979559 m`. These subset metrics are trace provenance, not promotion reports.

Full-history callable replay was verified directly against both frozen native
traces, including every rollout-start hidden state and terminal reset:

```text
./.venv/bin/python scripts/verify_native_policy_trace.py <parent> \
  logs/drone_race_full_policy_official_fit/jacobian_cma/traces/parent_floor4_trace.npz \
  --json-path logs/drone_race_full_policy_official_fit/jacobian_cma/traces/parent_floor4_trace_replay_verification.json

./.venv/bin/python scripts/verify_native_policy_trace.py <parent> \
  logs/drone_race_full_policy_official_fit/jacobian_cma/traces/parent_floor4p5_trace.npz \
  --json-path logs/drone_race_full_policy_official_fit/jacobian_cma/traces/parent_floor4p5_trace_replay_verification.json
```

Both reports pass at action/state tolerance `5e-5`. Floor `4` has maximum
operational action error `1.85e-6`, maximum hidden-state error `2.38e-7`, and
eight replayed terminal resets across `56` blocks. Floor `4.5` has the same
maxima and eight resets across `40` blocks. Native raw means are clipped only
for this comparison because both the environment and callable apply the
official `[-1, 1]` action contract; raw means remain stored unchanged.

The pre-registered basis command was:

```text
./.venv/bin/python scripts/jacobian_cma_subspace.py build <parent> \
  logs/drone_race_full_policy_official_fit/jacobian_cma/traces/parent_floor4p5_trace.npz \
  logs/drone_race_full_policy_official_fit/jacobian_cma/traces/parent_floor4_trace.npz \
  logs/drone_race_full_policy_official_fit/jacobian_cma/subspace/basis_dim12_seed3385.npz \
  --dimension 12 --failure-probes 24 --anchor-probes 8 \
  --damping-fraction 0.001 --seed 3385 --trace-agents 2 \
  --target-failure-action-rms 0.005 --maximum-anchor-action-rms 0.001 \
  --device auto
```

One attempted build produced no artifact because long-sequence forward-mode JVP
calibration became non-finite despite finite forward actions. The equivalent
normalized Hutchinson estimate from the already pre-registered reverse-mode
products replaced only that unstable calculation. A subsequent correctness
audit found that the generalized SVD vectors still needed mapping back through
the anchor metric; after that mapping, their sampled anchor/failure sensitivity
ratio remained about one. Before any candidate outcome evaluation, the final
construction therefore projected the generalized directions out of the eight
sampled anchor-Jacobian rows and orthonormalized them in raw FP32 checkpoint
space. These were implementation corrections, not new probes, seeds, or an
optimizer retry.

The final basis SHA-256 is
`05c4c2bd742180ca048307179c73ce7012327f0c4220c20a6a918c8208255793`.
It spans all `152064` allowed encoder, policy-decoder, and three MinGRU values;
`log_std` and the value row are excluded. Its maximum Gram error is `5.96e-7`,
anchor-null residual is `8.98e-8`, and all `24` singular values plus trace/basis
hashes are embedded in the artifact.

The frozen sensitivity spectrum, in descending order, is:

```text
[867.1213, 667.0544, 594.3281, 423.7142, 411.2239, 365.9832,
 342.1426, 271.3637, 259.9495, 247.8893, 244.3224, 236.0421,
 222.8182, 218.0653, 213.3591, 205.8104, 198.4686, 196.0135,
 187.9444, 179.0325, 177.7309, 171.4257, 162.6331, 154.3658]
```

Every candidate changes only the encoder, the four policy decoder rows, and
all three MinGRU projection matrices. `log_std`, the value row, the 32-field
input contract, and all architecture/physics fields remain byte-identical to
the parent outside that allowed set.

Generation-zero scale calibration used the unchanged seed-`3385` antithetic
population only on fixed parent observations; it observed no environment
outcomes:

```text
./.venv/bin/python scripts/calibrate_jacobian_cma_scale.py <parent> <basis> \
  <floor4.5-trace> <floor4-trace> \
  logs/drone_race_full_policy_official_fit/jacobian_cma/calibration \
  --dimension 12 --population-size 24 --parent-count 12 --seed 3385 \
  --target-failure-rms 0.005 --maximum-anchor-rms 0.001 \
  --maximum-anchor-max 0.01 --agents 2
```

The initial coefficient scale `0.0855355` gave median failure focus RMS
`0.00254668`, anchor p95 RMS `0.00466353`, and anchor p95 max `0.0212527`.
The hard anchor-RMS constraint was binding, fixing the sole calibrated scale at
`0.0183414`. Predicted post-scale values are failure RMS `0.000546084`, anchor
p95 RMS `0.001`, and anchor p95 max `0.00455721`. The complete 24-vector record
is `calibration/calibration.json`; none was outcome-evaluated. The optimizer
seed, population, covariance, generation budget, and basis dimension remain
unchanged. The CMA/subspace focused suite passes `11` tests.

Exactly one next action: run the frozen, resumable constrained CMA population
against fixed floor-`4.5` development episodes, with cheap trace feasibility,
actual floor-`4` evaluations for generation leaders, and disjoint held-out
validation before any exact-1024 candidate screen.

### Constrained CMA generation 0

The bounded run was launched with the frozen manifest and basis:

```text
./.venv/bin/python scripts/run_jacobian_cma.py \
  config/jacobian_cma_experiment.json \
  logs/drone_race_full_policy_official_fit/jacobian_cma/subspace/basis_dim12_seed3385.npz \
  logs/drone_race_full_policy_official_fit/jacobian_cma/traces/parent_floor4p5_trace.npz \
  logs/drone_race_full_policy_official_fit/jacobian_cma/traces/parent_floor4_trace.npz \
  logs/drone_race_full_policy_official_fit/jacobian_cma/run_dim12_seed3385
```

All evaluator invocations are deterministic exact-count native runs cached by
checkpoint SHA-256 plus the full evaluation configuration. Development uses
`128` episodes at per-vector offset `0`; held-out validation uses `128` episodes
at offset `1`. The parent target/anchor development and target validation
reports were cached before candidates.

Generation `0` evaluated the exact seed-`3385` antithetic population. `22/24`
candidates met the finite/action-drift trace constraint. Candidate `17` was the
lexicographic development winner: success `0.0`, gates `2.0`, crash `0.0`, no
Gate-2 crossing, terminal radial `0.894246 m`, final progress `0.688536`, and
anchor trace RMS `0.000925`. This improves development radial by about
`0.0867 m` without changing the discrete outcome.

Candidate `17` has SHA-256
`b9ba54bc878b83578c27cc20ce56ec1a50996dd11f74bbfd5b242d379cd30e3a`,
actual delta L2 `0.0517660`, `151507` changed allowed FP32 values, and
coefficients:

```text
[1.08929205, -0.25820011, -1.44279778, 0.98650527,
 -0.19233857, 1.54972637, 0.18345226, 0.72523165,
  0.55697137, -0.55933416, 0.16960211, 0.08712361]
```

Its anchor trace full/focus RMS is `0.000898/0.000925`, maximum `0.003241`.

The four development leaders were evaluated at actual floor `4`; all four
retain success `1.0` and crash `0.0`. Candidate `17` records Gate-2 crossing
radial `0.434273 m` there. Its disjoint floor-`4.5` validation is success `0.0`,
gates `2.0`, crash `0.0`, no Gate-2 crossing, terminal radial `0.895128 m`, and
final progress `0.688529`. The held-out radial improvement against the fixed
offset-`1` parent baseline is `0.087212 m`,
below the predeclared `0.10 m` continuation threshold, so no exact-1024 screen
was authorized.

CMA consumed the complete ranking once: sigma `1.0 → 0.946398`, mean L2
`1.30806`, covariance eigenvalue range `0.938209–1.138405`, and condition number
`1.21338`. State and every coefficient/report are saved under
`run_dim12_seed3385/generation_000/`; no candidate or configuration will be
repeated.

Exactly one next action: continue unchanged with covariance-adapted generation
`1`; do not promote generation `0` or run it exact-1024.

### Constrained CMA generation 1

Generation `1` resumed from the saved state and evaluated the next unchanged
seed stream. `18/24` candidates met the trace trust region; all four actual
floor-`4` leader checks again passed success `1.0` with crash `0.0`.

Candidate `20` is the winner. Its checkpoint SHA-256 is
`e93f2ac42e3ee73932b120c9ce34381b97af4e393d5273e521fbbe453cafa4db`, with
actual allowed-parameter delta L2 `0.0671734`. Its coefficient vector is:

```text
[1.85710728, 1.46210063, 0.46994027, 0.66719162,
 0.79797077, 0.56203759, 0.62254775, -1.38289988,
 -0.58999974, -1.26460588, 0.49357063, 1.31037223]
```

Anchor trace full/focus RMS is `0.000994/0.000992` and maximum is
`0.003966/0.003109`, within both hard constraints. Floor-`4.5` development is
success `0.0`, gates `2.0`, crash `0.0`, no Gate-2 crossing, terminal radial
`0.876148 m`, and final progress `0.688724`. Disjoint validation reproduces
radial `0.876035 m`; this clears the `0.10 m` improvement rule and authorized
the first candidate exact screens in this mechanism.

Authoritative exact-1024 floor `4.5` is still success `0.0`, gates `2.0`, crash
`0.0`, Gate-2 crossing `0.0`, terminal radial `0.875587 m`, and final progress
`0.688729`. This is a `0.105368 m` radial improvement over the exact parent but
not a discrete improvement or promotion. Authoritative exact-1024 floor `4`
remains success `1.0`, gates `4.0`, crash `0.0`, Gate-2 crossing `1.0`, and
crossing radial `0.402641 m`. Candidate `20` therefore proves a safe metric
direction but is not retained because target success is `0.0`.

CMA consumed generation `1` once: sigma `0.946398 → 0.873174`, mean L2
`1.50277`, covariance eigenvalue range `0.880578–1.152662`, and condition
`1.30898`. Every development, validation, exact, and anchor command is stored
in the content-addressed reports and candidate/summary JSON under
`generation_001/`.

Exactly one next action: continue the same CMA state into generation `2` to
seek a discrete Gate-2 crossing; do not retain candidate `20` or alter scale.

### Constrained CMA generation 2 and basis audit

Generation `2` produced `15/24` trace-feasible candidates; all four actual
floor-`4` leader checks retained success `1.0` and crash `0.0`. Candidate `15`
won development with success `0.0`, gates `2.0`, crash `0.0`, no Gate-2
crossing, and radial `0.878450 m`.

Its SHA-256 is
`8a91036df6ea74af3804571fa2a5ad74ee3b01f36ca286c367327e005f858a8c`,
actual allowed delta L2 is `0.0483495`, and coefficients are:

```text
[1.27621400, 0.89918876, -0.08599148, -0.18581489,
 -0.43504867, -0.04196188, 1.17572236, 1.49715459,
 0.31891516, 0.01861993, -0.69795960, -0.25646678]
```

Its anchor trace full/focus RMS is `0.000950/0.000854`, with maximum
`0.002976/0.002796`. Disjoint target validation is success `0.0`, gates `2.0`,
crash `0.0`, Gate-2 crossing `0.0`, and radial `0.882343 m`. The held-out
improvement against the fixed held-out parent is `0.099997 m`, just short of
the strict `0.10 m` exact
screen rule; no exact-1024 evaluation was run and the candidate is not retained.

After the predeclared third update, the sole basis audit measured `84.8239%`
captured sensitivity energy and `15.1761%` discarded energy. The pre-registered
expansion threshold is `25%` discarded, so an expansion to dimension `24` is
not evidence-backed and is closed. No new basis, seed, or scale is authorized.

CMA state after generation `2`: sigma `0.780211`, mean L2 `1.50200`, covariance
eigenvalue range `0.841106–1.232918`, and condition `1.46583`.

Exactly one next action: continue the unchanged dimension-`12` state into
generation `3`; do not exact-screen or retain candidate `15`.

### Constrained CMA generation 3

A pre-update audit caught and corrected an orchestration defect: if fewer than
`12` samples were feasible, the generic ranking would have filled the CMA
parent set with trace-rejected candidates. The active process was interrupted
while computing action drift (not during a native evaluation), a tested hard
guard was added, and the run resumed from generation `2` state. Content-addressed
reports prevented any completed outcome evaluation from repeating. The focused
guard/CMA suite passed `14` tests. Generation `3` ultimately supplied `15/24`
feasible candidates, so no constraint stop was needed and only feasible
candidates populated its `12` parents.

Candidate `23` won development at success `0.0`, gates `2.0`, crash `0.0`, no
Gate-2 crossing, and radial `0.878973 m`. Its SHA-256 is
`4d4402c3b656a0e73fbf18f8648c039bcdcbfbdbf5a6953e0a2fb1a5a2b7a7c6`,
actual allowed delta L2 is `0.0491866`, and coefficients are:

```text
[0.63025767, 0.95783669, -0.20987253, 0.47639599,
 -0.17654525, 0.04036798, 1.90124989, 1.13017237,
 0.62506831, 0.39547583, -0.12021049, -0.34587854]
```

Anchor trace full/focus RMS is `0.000966/0.000913`, maximum `0.003778`. All four
actual floor-`4` leader checks passed. Candidate `23` validated at radial
`0.878520 m`, clearing the `0.10 m` rule; authoritative exact-1024 floor `4.5`
is success `0.0`, gates `2.0`, crash `0.0`, no Gate-2 crossing, and radial
`0.878693 m`. Exact-1024 floor `4` remains success `1.0`, gates `4.0`, crash
`0.0`, and Gate-2 crossing radial `0.408559 m`. This target result is worse than
generation `1` candidate `20` (`0.875587 m`) and has no discrete improvement;
candidate `23` is rejected.

CMA state after the valid update: sigma `0.687738`, mean L2 `1.50793`, covariance
eigenvalue range `0.796387–1.176013`, and condition `1.47668`.

Exactly one next action: continue unchanged with generation `4`; retain the
original floor-`4` parent, not either metric-only child.

### Constrained CMA generation 4

Generation `4` supplied `19/24` trace-feasible candidates, enough to populate
all `12` CMA parents without violating the hard trust region. All four actual
floor-`4` leader checks passed with success `1.0` and crash `0.0`.

Candidate `1` won development at success `0.0`, gates `2.0`, crash `0.0`, no
Gate-2 crossing, and terminal radial `0.882075 m`. Its SHA-256 is
`8cac7a5a097493613f2d826dedffd01c97303684f038b2bfa10a1533e3a5cc00`,
actual allowed delta L2 is `0.0459150`, and coefficients are:

```text
[0.39008173, 0.22001237, -1.17060375, 0.27702731,
 1.42477787, -0.00113460, 1.24088931, 0.36184946,
 -0.80767894, -0.15343338, 0.32314065, 0.37171561]
```

Anchor trace full/focus RMS is `0.000956/0.000982`, maximum `0.003724`.
Disjoint held-out floor `4.5` remains success `0.0`, gates `2.0`, crash `0.0`,
with no Gate-2 crossing and radial `0.883601 m`. The `0.098739 m` radial
improvement over the fixed held-out parent is below the `0.10 m` exact-screen rule, so
candidate `1` was neither exact-1024 screened nor retained.

CMA state after generation `4`: sigma `0.634930`, mean L2 `1.68189`, covariance
eigenvalue range `0.781148–1.173200`, and condition `1.50189`. A tested terminal
record now distinguishes trust-region exhaustion from exhausting all eight
generations without promotion; the runner test suite passes `7` tests.

Exactly one next action: continue the same saved CMA state into generation `5`;
retain the original floor-`4` parent.

### Constrained CMA generation 5

Generation `5` supplied `16/24` trace-feasible candidates, and all four actual
floor-`4` leader checks retained success `1.0` with crash `0.0`. Candidate `17`
won development at success `0.0`, gates `2.0`, crash `0.0`, zero Gate-2
crossings, and radial `0.883243 m`. Its SHA-256 is
`dc666bc68546e50f5a13345fa0107cf7f096deb53900b1fbf2b20cbd16534077`,
actual delta L2 is `0.0619910`, `151512` allowed FP32 values changed, and its
coefficients are:

```text
[-0.08240714, 0.91694444, -1.93997073, -0.30098400,
 -0.34028238, -0.65672886, 1.11377561, -0.46923149,
 -0.73227739, -1.14777195, 0.82418799, 1.47684956]
```

Anchor trace full/focus RMS is `0.000887/0.000898`, maximum `0.003372`.
Disjoint held-out floor `4.5` is success `0.0`, gates `2.0`, crash `0.0`, no
crossing, and radial `0.882754 m`. The `0.099586 m` held-out improvement is
still below the exact-screen threshold by `0.000414 m`; no exact evaluation was
authorized and the candidate is rejected.

CMA state after generation `5`: sigma `0.620679`, mean L2 `1.70945`, covariance
eigenvalue range `0.739713–1.459053`, and condition `1.97246`.

Exactly one next action: continue the unchanged saved state into generation
`6`; retain the original parent.

### Constrained CMA generation 6

Generation `6` supplied `15/24` feasible candidates and four passing actual
floor-`4` leader checks. Candidate `4` won development with success `0.0`, gates
`2.0`, crash `0.0`, no crossing, and radial `0.879376 m`. Its SHA-256 is
`6058ff73f317511aff08e91cb58f0534a454ed71e0afe86aae8fc8aad42d9a8d`,
actual delta L2 is `0.0469781`, `151496` allowed FP32 values changed, and its
coefficients are:

```text
[0.69957548, 0.09140836, -0.99301076, -0.23531084,
 -0.25944263, 0.08075710, 1.55323935, 1.04732716,
 -0.93714374, -0.24511288, 0.33965629, 0.61985600]
```

Anchor trace full/focus RMS is `0.000915/0.000832`, maximum `0.003120`.
Held-out target radial is `0.878470 m`, a qualifying `0.103869 m` improvement,
so the predeclared exact screen ran. Exact-1024 floor `4.5` remains success
`0.0`, gates `2.0`, crash `0.0`, no Gate-2 crossing, and radial `0.878673 m`.
Exact-1024 floor `4` remains success `1.0`, gates `4.0`, crash `0.0`, with
Gate-2 crossing radial `0.405144 m`. The target result is worse than generation
`1` candidate `20` and has no discrete improvement; candidate `4` is rejected.

CMA state after generation `6`: sigma `0.572897`, mean L2 `1.78726`, covariance
eigenvalue range `0.717012–1.494462`, and condition `2.08429`.

Exactly one next action: consume the final pre-registered generation `7`; do
not retain candidate `4` or change the seed, basis, or scale.

### Constrained CMA generation 7 and bounded stop

The final generation supplied `16/24` feasible candidates. All four actual
floor-`4` leader checks passed. Candidate `18` won development at success
`0.0`, gates `2.0`, crash `0.0`, zero Gate-2 crossings, and radial
`0.867579 m`. Its SHA-256 is
`08b93de37a9d421d9ae3368a9991db18fc5aad59f3a2f881de3e39b4d7470c7c`,
actual delta L2 is `0.0548559`, `151496` allowed FP32 values changed, and its
coefficients are:

```text
[0.56476545, 0.35563609, -0.80944306, -0.50971884,
 0.64810437, -1.47372401, 1.64415967, 1.33565629,
 -0.60994768, -0.01313074, -0.01071584, 0.36494389]
```

Anchor trace full/focus RMS is `0.000954/0.000942`, maximum `0.003810`.
Held-out floor `4.5` reaches radial `0.867065 m`, a `0.115275 m` improvement,
but still success `0.0`, gates `2.0`, crash `0.0`, and no crossing.
Authoritative exact-1024 floor `4.5` is success `0.0`, gates `2.0`, crash `0.0`,
zero Gate-2 crossings, radial `0.867062 m`, and final progress `0.688820`.
Exact-1024 floor `4` remains success `1.0`, gates `4.0`, crash `0.0`, with
Gate-2 crossing radial `0.404844 m`. Candidate `18` is the branch's best radial
result but fails the discrete and success contract, so it is rejected and the
unchanged parent remains retained.

Final CMA state is generation `8`, sigma `0.550017`, mean L2 `1.91969`,
covariance eigenvalue range `0.684100–1.520317`, and condition `2.22236`.
Covariance did not collapse; the frozen eight-generation budget was exhausted
without promotion. `final.json` records stop reason
`maximum_generation_budget_exhausted_without_promotion` and indexes summaries
`0–7`.

### Final negative-result audit

The exact run command is the generation-`0` command above; every evaluator
subcommand, complete configuration, checkpoint hash, episode offset, and wall
time is embedded in its candidate/summary record and content-addressed cache.
Fixed development identity is `128` deterministic episodes at offset `0`;
held-out identity is `128` at offset `1`; exact screens use `1024` at offset
`0`. Seed `3385`, basis hash `05c4c2bd…5793`, coefficient scale `0.0183414`,
population `24`, parents `12`, and all trust limits remained unchanged.

Recorded per-generation wall evidence is:

| Generation | Filesystem run interval (s) | Unique native-evaluator wall (s) |
|---:|---:|---:|
| 0 | 810.409 | 771.482 |
| 1 | 757.246 | 718.564 |
| 2 | 605.817 | 566.704 |
| 3 | 572.370 | 628.520 |
| 4 | 709.139 | 670.394 |
| 5 | 632.436 | 590.430 |
| 6 | 650.081 | 612.085 |
| 7 | 676.240 | 637.892 |

The filesystem intervals total `5413.738 s`; unique recorded evaluator work
totals `5196.071 s`. Generation `3`'s evaluator total includes reports produced
before its tested interruption and reused from cache, while its filesystem
interval begins at the resumed coefficient file, explaining the local overlap.

The rejected-candidate command was:

```text
./.venv/bin/python scripts/audit_jacobian_cma_run.py \
  logs/drone_race_full_policy_official_fit/jacobian_cma/run_dim12_seed3385 \
  logs/drone_race_full_policy_official_fit/jacobian_cma/run_dim12_seed3385/rejected_candidates.json
```

It accounts for all `192` candidates with `192` unique FP32 coefficient vectors
and `192` unique merged checkpoint hashes: `136` development evaluations, `32`
actual anchor evaluations, and `8` exact evaluations, represented by `184`
unique cache references with no duplicate reference. The JSON table preserves
every full coefficient vector, drift report, available development/anchor/
held-out/exact metrics, and explicit rejection reason. No official-simulator
process was launched.

Final verification passes `53` relevant Python tests and the native drone-race
regression suite. A reconstruction audit independently rematerialized the
expected FP32 vector for every one of the `192` candidates and found exact
equality; native precision is `4`, all values are finite, and `log_std` plus the
value row are byte-identical to the parent in every checkpoint. The trace and
basis hash/Gram assertions, final generation index, unique-candidate index,
best exact metrics, inactive-process check, and `git diff --check` also pass.

The hypothesis is falsified within its pre-registered budget: persistent
cross-layer covariance adaptation found a reproducible safe radial direction
but no ordered Gate-2 crossing and no floor-`4.5` success. There is no retained
child and no permission to retry a seed, sigma, tensor subset, dimension, or
renamed variant.

Exactly one next action: stop the Jacobian-CMA branch and retain the original
floor-`4` checkpoint; do not launch the official simulator or silently fall
back to a closed optimizer.
