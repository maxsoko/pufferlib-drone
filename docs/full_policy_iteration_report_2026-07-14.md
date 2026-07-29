# Full-Policy True-Aperture Iteration Report — 2026-07-14

## Outcome

Native curriculum is still in progress; no checkpoint is currently promotable
and no second official attempt has been authorized.

This iteration corrected the target geometry, preserved the official failure
evidence, established a three-gate true-aperture parent, and simplified the
remaining native loop:

- all target gate radii are now `0.75 m`;
- gates 0-2 remain fixed at target geometry while only gate 3 is annealed;
- the retained-parent runner resumes its persisted refined schedule instead of
  reconstructing or repeating a failed jump;
- training-only Gaussian variance can be reduced without changing deterministic
  policy actions;
- redundant stochastic post-training epochs are disabled because every child
  already receives a separate deterministic development screen;
- resume skips re-screening the already accepted parent and rejects changed
  environment overrides, removing another deterministic-eval cycle safely;
- new child records include checkpoint steps, wall-clock training duration, and
  effective end-to-end SPS so cycle-time changes remain measurable;
- a passive watcher performs only the exact all-gate 4096-episode promotion
  after the Gate-3 state reaches `0.75 m`.

At that point, the authoritative live state was
`logs/drone_race_full_policy_stage_d_gate4/radius_curriculum_true_aperture_g3_state.json`.

## Later v3385 transfer-parity update

The earlier true-aperture state above remains valid historical evidence, but it
is no longer the active continuation. Official full-policy attempt 004 passed
gate 0, advanced to `active_gate_index=1`, lost the associated gate track near
`4.08 m`, and timed out without collision. Native parity now applies the same
close-range observation loss from gate index 1 at `4.25 m`.

Under the corrected contract, the former promoted source falls to
`success=0.830078` on the exact 1024-episode two-gate screen. A bounded
phase-column search produced a diagnostic parent at:

```text
checkpoints/drone_race_full_policy_stage_d_gate4/
  gate1_m1_gate2_phase_scale_screen/z0.bin
```

Its exact results are:

```text
two gates, dropout 4.25 m:   success=0.954102  crash=0.0
three gates, dropout 2.50 m: success=0.954102  crash=0.0
three gates, dropout 4.25 m: success=0.953125  crash=0.0
four gates, dropout 2.50 m:  success=0.0       crash=0.0
```

Gate positions are present as internal `env.gates[]` absolute coordinates, but
the policy receives only active-gate official-observable geometry and gate
phase. The correction did not add privileged coordinates.

Near-deterministic PPO with `log_std=-8` was rejected after producing KL values
above 1.0. Stable PPO with `log_std=-4` and `learning_rate=1e-6` held KL near
`0.001-0.006`, but full-policy and relaxed-aperture children still improved the
last-gate radial error by degrading the solved prefix. All children were
rejected. The final diagnostic state is:

```text
logs/drone_race_full_policy_stage_d_gate4/
  v3385_gate3_dropout_radius_curriculum_state.json
```

The unchanged parent passes a relaxed gate-3 radius of `12.75 m`; no bounded
child passes `12.5 m`. Do not resume that radius state and do not run another
official attempt. The next hypothesis is a clean full-policy RL curriculum
trained under the corrected dropout contract from its first two-gate stage,
using prefix length or temporary aperture only to provide terminal credit.

## Geometry correction and official evidence

The official `TS002_GATE_INNER_WIDTH_M=1.5` value is the complete inner width,
not a radial tolerance. Native `gate_radius` measures radial distance from the
gate center at the crossing plane, so the equivalent target is `0.75 m`.

The earlier relaxed candidate
`checkpoints/drone_race_full_policy_stage_d_gate4/bc_phase2_feature24_mix_0p25_scale_1p45/0000000000000000.bin`
completed exactly 4096 native radius-1.5 episodes with:

```text
success_rate=0.968505859375 (3967/4096)
crash=0.0
gates_passed=3.93701171875
completion_time=12.284111 s
```

That result is curriculum evidence only. Its first official v3385 flight is
recorded in
`logs/sitl/competition_smoke_policy_full_policy_native_promo_20260714_02_attempt_001.json`.
The run collided with the left post of gate 0 at `4.469 s`, with:

```text
official_active_gate_index=0
ordered_gate_passes=0
closest_range_m=2.848665
closest_body_vector=[2.848665, 0.756677, 0.387240]
effective_command_hz=50.123
heartbeat_hz=2.0
command_rate_violations=0
telemetry_drain_limit_hits=0
```

The `0.756677 m` right displacement independently agrees with a `0.75 m`
half-width boundary. The checkpoint must not be retried officially.

## True-aperture parent

Behavior cloning plus a minimal phase-conditioned feature correction produced
`checkpoints/drone_race_full_policy_stage_d_gate4/bc_true_aperture_phase2_feature24_line_2p75/0000000000000000.bin`.
It passes gates 0-2 with all three radii fixed at `0.75 m`, has zero crashes, and
misses only the final gate. It provided a clean Gate-3 radius boundary:

| Gate-3 radius | Development success | Crash | Decision |
|---:|---:|---:|---|
| 4.75 m | 0.970760 | 0.0 | valid initial parent |
| 4.50 m | 0.897661 | 0.0 | train before accepting |

The first default-variance PPO child accepted `4.50 m` at `0.935880`; two more
chunks eventually accepted `4.25 m` at `0.902519`. Both default-variance
children at `4.00 m` failed and regressed from `0.775710` to `0.704102`, so the
last passing parent remained intact and the runner refined the step to `0.125 m`.

## Efficient low-variance continuation

The source checkpoints carried Gaussian `log_std` near `-2` (standard deviation
about `0.135`). Stochastic training rollouts therefore knocked most agents off
the already-solved `0.75 m` prefix gates before they could collect Gate-3 signal.

`scripts/train_full_policy_radius_curriculum.py --exploration-log-std -4.0`
now rewrites only the four training-time Gaussian log-standard-deviation values
in a temporary checkpoint. Direct validation proved:

```text
log_std: approximately [-1.98, -2.00, -2.00, -1.99] -> [-4, -4, -4, -4]
deterministic_action_max_delta=0.0
```

On the same retained `4.25 m` parent, the first low-variance 2M-step child
accepted `4.125 m` at `success=0.939024`, `crash=0.0`. The same child then passed
the tighter `4.00 m` screen without another training chunk at
`success=0.917683`, `crash=0.0`. This reversed the two-child default-variance
failure while reducing the number of required training cycles.

The resumable command is:

```bash
.venv/bin/python scripts/train_full_policy_radius_curriculum.py \
  --resume-state logs/drone_race_full_policy_stage_d_gate4/radius_curriculum_true_aperture_g3_state.json \
  --target-radius 0.75 \
  --min-radius-step 0.0078125 \
  --train-timesteps 500000 \
  --max-finetunes-per-radius 4 \
  --exploration-log-std -4.0 \
  --trainer-eval-episodes 0 \
  --reward-scale 0.005 \
  --set env.gate-radius=0.75 \
  --set env.gate2-radius=0.75 \
  --set env.w-action-teacher=500 \
  --set env.teacher-course-spline=1 \
  --set env.teacher-from-gate-index=3 \
  --set env.teacher-pitch-from-gate-index=4 \
  --set env.teacher-roll-from-gate-index=3 \
  --set env.teacher-thrust-from-gate-index=3
```

### Short bounded correction result

At Gate-3 radius `1.40625 m`, the retained parent was already one deterministic
episode short of the development threshold:

```text
parent:       success=0.899610  crash=0.0
2M child:     success=0.866089  crash=0.0  train=45.74 s  rejected
0.5M child:   success=0.915122  crash=0.0  train=15.52 s  accepted
```

The 2M update overshot and degraded a nearly passing parent. The 491,520-step
checkpoint improved it while using about one third of the training time. The
runner therefore defaults to 0.5M bounded children and screens after each one;
four sequential children retain the former 2M maximum budget, but successful
early corrections stop immediately.

The passive promotion watcher is:

```bash
.venv/bin/python scripts/finish_full_policy_native_curriculum.py \
  --gate3-state logs/drone_race_full_policy_stage_d_gate4/radius_curriculum_true_aperture_g3_state.json
```

It does not launch a trainer or the official simulator. After the radius state
is complete at exactly `0.75 m`, it evaluates all four radii at `0.75 m` with
`--eval-episodes 4096 --max-rollouts 160` and accepts only `env/n == 4096`,
success `>=0.90`, crash `<=0.10`, and ordered completion.

The first shortened cycle confirmed the expected wall-time improvement. The
older low-variance `3.625 m` cycle ran from Puffer run creation at
`02:43:07.721` through its deterministic report at `02:44:32.444` (about
`84.7 s`). With internal trainer evaluation disabled, the `2.50 m` cycle ran
from `02:51:27.827` through its deterministic report at `02:52:31.892` (about
`64.1 s`), saving `20.6 s` or roughly 24%. The new child was accepted at
`success=0.944336`, `crash=0.0`; the latency reduction did not weaken the
promotion decision.

## Verification

```text
bash scripts/test_drone_race_native_regressions.sh
drone_race native regressions ok

cc -O3 -I. scripts/collect_full_policy_teacher.c \
  -o build/collect_full_policy_teacher -lm
collector_build_ok

.venv/bin/pytest -q \
  tests/test_train_full_policy_radius_curriculum.py \
  tests/test_finish_full_policy_native_curriculum.py
17 passed

.venv/bin/python -m py_compile \
  scripts/train_full_policy_radius_curriculum.py \
  scripts/finish_full_policy_native_curriculum.py

git diff --check
clean
```

## Exactly one next hypothesis

Continue the retained-parent Gate-3 anneal with training-only `log_std=-4.0`.
The lower variance should keep enough rollouts on the solved true-aperture
prefix to shrink Gate 3 more efficiently; accept only deterministic
development success `>=0.90` with crash `<=0.10`, then let the passive watcher
run the exact all-gate promotion at `0.75 m`.

## Later clean-start correction

The radius-anneal hypothesis above is now closed. Its repaired recurrent parent
could clear relaxed Gate 3 only near a `12.75 m` radius; bounded PPO children
improved the last-gate miss by degrading the already-solved prefix. It is not an
efficient route to the official `0.75 m` aperture.

The active continuation is a clean recurrent policy trained from the corrected
v3385 observation contract. `scripts/train_full_policy_clean_curriculum.py`
implements the resumable controller and
`config/full_policy_clean_curriculum.json` declares the stages. The schedule
uses one policy across a proximal control bootstrap, the official first gate,
2/3/4-gate prefixes, camera-dropout distances `2.5 -> 3.25 -> 4.25 m`, and
course-position jitter. Training may temporarily use a `1.0 m` aperture or a
slightly shorter dropout distance, but every retention decision evaluates the
full start at the stage's exact settings. A child cannot overwrite the retained
checkpoint without success `>=0.90` and crash `<=0.10` over exactly 1024
episodes; final promotion remains exactly 4096 episodes at radius `0.75 m`.

Verification added for the clean runner:

```text
.venv/bin/pytest -q \
  tests/test_train_full_policy_clean_curriculum.py \
  tests/test_train_full_policy_transfer_curriculum.py \
  tests/test_train_full_policy_radius_curriculum.py
27 passed
```

The next bounded experiment is the from-scratch proximal bootstrap with
`--max-stages 1`. If it passes its exact screen, resume the same state for the
official Gate-0 stage; do not return to phase-column or radius surgery.

That bounded experiment passed on the first child:

```text
checkpoint: checkpoints/drone_race_full_policy_stage_d_gate4/1784023348989/0000000003997696.bin
training:   3,997,696 steps, 96.913 s, 41,250 effective SPS
exact eval: n=1024, success=1.0, crash=0.0, gates=1.0
time:       0.621629 s
report:     logs/drone_race_full_policy_stage_d_gate4/clean_1784023346204_s00_a0000_proximal_control_bootstrap.json
state:      logs/drone_race_full_policy_stage_d_gate4/full_policy_clean_curriculum_state.json
```

The state is paused cleanly with `next_stage_index=1`. The next hypothesis is
that the same recurrent weights can learn the official-position Gate 0 using a
training-only `1.0 m` aperture and retain only if the exact `0.75 m` screen
passes.

The checkpoint already satisfied that hypothesis without further PPO. Its
official-position Gate-0 exact screen produced `n=1024`, success `1.0`, crash
`0.0`, and completion time `1.466673 s` at the true `0.75 m` aperture. The
runner therefore skipped the configured easier-aperture training and advanced
the state to `next_stage_index=2`. This validates the intended efficient path:
screen first and spend training time only when a genuinely new prefix fails.

The next hypothesis is that one bounded 4M-step update with training radius
`1.0 m` is enough to add Gate 1 under `2.50 m` close-range observation dropout,
while the exact screen retains only success at radius `0.75 m`.

## Full-course-first correction

The prefix/segment hypothesis is superseded. Isolated-gate resets did not carry
the recurrent state or entry velocity produced by the real course, and a Gate-0
terminal sprint exited at the native `20 m/s` cap with too little time to learn
the following descent. The simpler retained lineage now starts every episode at
the official origin with all four gates. Wide full-course apertures provide
terminal credit; radius, camera dropout, and course jitter are tightened only
after an exact full-start screen passes.

The declarative schedule is
`config/full_policy_clean_curriculum.json`. The runner now supports safe pending-
stage refresh, one-job invocation limits, signed crossing diagnostics at any
missed gate, and a `0.10 m` minimum radial improvement before chaining an
incomplete child. Native reward diagnostics also record Gate-0 exit velocity and
can penalize full 3D error against the velocity required by the next gate.

The fresh 18 m bootstrap command was:

```text
.venv/bin/python scripts/train_full_policy_clean_curriculum.py \
  --max-stages 1 --max-train-attempts 1 \
  --state-path logs/drone_race_full_policy_stage_d_gate4/full_policy_coursewide_curriculum_state.json
```

It produced:

```text
checkpoint: checkpoints/drone_race_full_policy_stage_d_gate4/1784059094357/0000000003997696.bin
training:   3,997,696 steps, 101.105 s, 39,540 effective SPS
exact eval: n=1024, success=1.0, crash=0.0, gates=4.0
time:       4.866663 s
radial:     15.961093 m at the final plane
```

The next stage screens exact radius 15 m and trains with a slightly relaxed
aperture. One 0.5M child trained at 17 m reduced the deterministic final miss to
`15.844692 m` in `17.185 s` (`28,602` effective SPS). Tightening the training
aperture to 16 m reduced it again to `15.673779 m` in `17.653 s` (`27,843`
effective SPS). Both exact screens retained three ordered gates, zero crashes,
and zero completion success, so neither child passed the stage. The second child
is only a working checkpoint; the accepted 18 m parent remains protected.

The authoritative continuation is:

```text
state:   logs/drone_race_full_policy_stage_d_gate4/full_policy_coursewide_curriculum_state.json
working: checkpoints/drone_race_full_policy_stage_d_gate4/1784059312590/0000000000491520.bin
status:  stage_exhausted at coursewide_radius_15p00
```

Use `--resume-state` to continue this lineage. `--state-path` creates a new
lineage; confusing the two caused one unnecessary fresh bootstrap and is now
called out explicitly in the execution prompt. The training hot loop continues
to use `.venv/bin/python`: switching to `uv` would improve setup and dependency
resolution, not native PPO, evaluation, or simulator-reset throughput.

## Exactly one next hypothesis after the full-course correction

One more bounded 0.5M update at training radius 16 m will preserve nonzero
terminal credit and reduce the exact radius-15 final-plane miss by at least
`0.10 m`. Accept it only if it passes the exact screen or meets that material-
progress threshold; otherwise stop this unchanged PPO condition and redesign
the final-approach reward before spending another training cycle.
