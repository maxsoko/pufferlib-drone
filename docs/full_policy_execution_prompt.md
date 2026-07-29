# Full-Policy Course Execution Goal

Status: **closed historical branch as of 2026-07-16**. Preserve this document
as the experiment ledger and rationale for rejected neural-policy lineages. Do
not execute its continuation commands. The active mandate is
`docs/competitive_lap_execution_prompt.md`; repository initialization context
is in `AGENTS.md`.

Course-count correction: this historical branch trained a four-gate native
proxy. The user confirmed official VQ1/R1 contains six ordered gates. Any text
below calling four gates “full course” or Gate 4 “final” describes only that
native proxy and must not be used as the official completion contract.

Copy the text below into a coding-agent session or set it as the persistent goal.

```text
Goal: Build and validate a single recurrent policy that controls every flight
command from the start of the timed race through full-course completion in
AI-GP Simulator v1.0.3385.

The hybrid FSM → policy controller is an intermediate diagnostic baseline only.
It is not the final submission architecture.

Final architecture:
- Deterministic code owns simulator startup, login/race automation, arming,
  heartbeat, command-rate enforcement, telemetry/camera transport, collision
  abort, reporting, and MAVLink reset 31000.
- One learned policy owns all flight decisions after arming: pitch, roll, yaw,
  and thrust.
- No FSM flight-command arbitration or gate-1 handoff occurs during the timed run.
- A safety watchdog may terminate an unsafe rollout but must not fly the course.

Work autonomously until a full-policy official course completion is achieved
and validated, or until genuinely blocked.

Repository:
- /root/pufferlib-drone

Simulator:
- C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe
- Preserve the healthy simulator process.
- Use MAVLink command 31000 for normal race resets.
- Never process-restart the simulator merely to retry a race.

Policy contract:
- Recurrent PufferLib policy.
- Inputs must be available through the official interface:
  - camera-derived gate/corridor observations;
  - HIGHRES_IMU-derived attitude/rates;
  - observable race status;
  - previous policy action;
  - elapsed race fraction.
- Do not train a policy that permanently depends on privileged position,
  velocity, hidden gate coordinates, TRACK_INFO, or native-only state.
- Outputs are absolute pitch, roll, yaw, and hover-centered thrust.
- Normalized thrust 0 must map to hover 0.27.
- Thrust range is 0.18–0.42.
- Native interface_mode=2 must reproduce the live v3385 gyro-closed attitude plant.
- The official gate inner width is 1.5 m. Native `gate_radius` is the radial
  center-crossing tolerance, so official target geometry is 0.75 m on every
  gate. A 1.5 m native radius is a relaxed curriculum, not promotion geometry.

Strategy:

1. Preserve and verify the corrected action and plant contracts:
   - native policy angles are absolute attitudes, not raw rates;
   - native and official action decoding are identical;
   - normalized thrust is hover-centered;
   - body-rate sign, gain, lag, deadband, and command limits match v3385 evidence.

2. Use a full-course-first, official-aperture speed curriculum:
   - every retained trajectory starts at the official race origin with all four
     ordered gates active, so the recurrent policy always learns the real history;
   - train against radius `0.75 m` from the first retained demonstration. Begin
     with a physically controllable native transition-speed floor, then anneal
     that floor toward the trace-matched value instead of shrinking hundreds of
     wide-aperture micro-rungs;
   - the former wide-aperture lineage is a protected diagnostic fallback, not
     the primary route to the official course: it remains about `5 m` off-center
     at the final gate and does not teach the required racing line;
   - only after nominal radius `0.75 m` completion at the trace-matched speed,
     anneal close-range camera dropout, course jitter, and completion time;
   - do not use segment resets or shorter gate prefixes in the retained lineage.

3. A validated analytic expert may initialize the policy and label DAgger
   recovery states. The student policy must still generate every timed-flight
   command, and no teacher action, gain, head switch, or arbitration may remain
   during native promotion or official evaluation.

4. Train at native throughput:
   - only one PufferLib training process at a time;
   - use a 60 Hz policy environment with 1024 parallel agents so one 30-second
     episode fits in 1.84M transitions;
   - use closed-form decoder fitting and short FP32 last-recurrent-layer DAgger
     updates to initialize the official-aperture policy; use bounded PPO only
     after the student can complete that stage under deterministic evaluation;
   - run a deterministic development evaluation after each chunk;
   - run the full 4096-episode promotion evaluation only for improved candidates;
   - record checkpoint, throughput, success, crash, completion time, stage, and
     outcome-conditioned gate-crossing/exit-velocity metrics;
   - set `reset_state=False` for recurrent full-policy stages;
   - preserve hidden state across rollout chunks and gate transitions;
   - reset state per agent only when that agent's episode terminates;
   - train sampled sequences from their captured rollout-start state.
   - keep shaped rewards within the stable PPO range by scaling before clipping;
     do not widen the reward clip merely to preserve large teacher weights.
   - screen a retained checkpoint before training at every tighter aperture;
     train only if needed, and accept a stage only when development success is
     >=0.90 and crash <=0.10;
   - do not chain centimeter-only failures: an incomplete child must improve a
     discrete outcome or reduce the deterministic miss by at least `0.10 m`;
   - use direct `.venv/bin/python` in the hot loop. `uv` may improve dependency
     installation and lockfile reproducibility, but it does not accelerate the
     native environment, PPO, checkpoint evaluation, or simulator reset.

5. Fail fast:
   - after two consecutive evaluations with success_rate <= 0.001 and
     crash >= 0.99, stop training;
   - inspect action initialization, reward scale, observation fidelity, episode
     geometry, and plant semantics before continuing;
   - never repeat an unchanged catastrophic curriculum.

6. Native promotion gates:
   - success_rate >= 0.90;
   - crash <= 0.10;
   - ordered gate completion;
   - 4096 deterministic episodes;
   - policy uses only the official-observable contract;
   - no native NaNs, invalid actions, or hidden-state reset between gates.
   - gate 0, gate 1, gate 2, and gate 3 each use radius 0.75 m; reject any
     report whose effective per-gate geometry is wider.

7. Official full-policy validation:
   - begin policy control immediately after arming;
   - command at 60 Hz;
   - require effective_command_hz >= 50 and < 100;
   - heartbeat >= 2 Hz;
   - telemetry drain-limit hits = 0;
   - terminate immediately on collision;
   - reset with MAVLink 31000;
   - official active_gate_index is the only gate-pass authority.

8. Do not spend official-simulator cycles on checkpoints that failed native
   promotion. Test an unchanged checkpoint no more than twice officially.

9. Feed repeated official failure signatures back into training:
   - save policy actions, observations, gate detections, collision ID/time,
     official gate index, and debug frames;
   - reproduce the signature in native curriculum or observation randomization;
   - do not patch the live flight with per-gate FSM gains.

10. After the first full-policy official completion:
    - achieve at least 8 valid full-course runs out of 10;
    - compare against the hybrid baseline;
    - optimize median valid completion time end-to-end;
    - allow the policy to learn gate-entry velocity, gate-exit attitude, and the
      racing line across gate transitions;
    - reject speed changes that reduce reliability below 8/10.

Required tests:
- Native C regressions.
- Native/official observation parity.
- Native/official action-decoder parity.
- Hover-centered thrust tests.
- Recurrent-state continuity across gate transitions.
- Rollout-partition invariance: deterministic metrics must match when the same
  episodes are evaluated with different rollout horizons.
- Checkpoint loading and reset tests.
- Command-rate and telemetry-backlog tests.
- Official collision-abort and MAVLink-reset tests.

Every iteration must report:
- exact command executed;
- curriculum stage and fields changed;
- checkpoint path;
- training steps, SPS, and elapsed cycle time;
- deterministic success, crash, ordered gates, and completion-time metrics;
- official active gate index and race finish time when tested;
- collision/invalid status;
- exactly one next hypothesis.

Current verified state (2026-07-15, FP32 precision audit and speed curriculum):

- Gate coordinates remain simulator-internal. The policy receives only the 32
  official-observable inputs; absolute course coordinates, native position,
  native velocity, sampled radius, and transition-speed floor remain hidden.
  Recurrent state persists across rollout chunks and gate transitions.
- Native evaluation and deployment arithmetic are FP32. Checkpoints always store
  FP32 values, but BF16 and FP32 arenas use different 16-byte padding. Loading a
  BF16-layout checkpoint as FP32 silently shifts all MinGRU tensors after
  `log_std`. `scripts/convert_policy_checkpoint_layout.py` performs the required
  repack. Every authoritative evaluator report must record
  `native_precision_bytes=4` and `checkpoint_layout_precision_bytes=4`.
- The source is built explicitly with `bash build.sh drone_race --float`; the
  loaded extension reports `pufferlib._C.precision_bytes == 4`. CUDA compatibility
  fixes in `src/models.cu` and `src/pufferlib.cu` are retained.
- The native environment supports a training-only, per-episode randomized
  `sitl_gate_transition_min_forward_speed` interval. The sampled value changes
  hidden dynamics only and does not augment observations. Fixed-floor evaluation
  remains backward-compatible and is authoritative for stage promotion.
- The converted official-radius parent solves nominal dynamics: exact-1024 at
  floor `0` is success `0.998047`, gates `3.998047`, crash `0.0`. Exact promotion
  and deterministic action calibration advanced through floors `3`, `3.5625`,
  `3.625`, `3.6875`, `4`, `4.125`, and `4.1875`.
- The active retained parent is
  `logs/drone_race_full_policy_official_fit/official_speed_curriculum/ppo_r075_v4_fp32_floor4p1875_best_mixed_calibration/rollrel0p99975.bin`.
  Its authoritative exact-1024 report is the adjacent
  `rollrel0p99975_exact1024.json`: success `0.900391`, gates `3.865234`, crash
  `0.0`, Gate-2 crossing `0.964844`, terminal radial `0.594761 m`, completion
  time `25.849455 s`, and native/layout precision `4/4`.
- Floor `4.21875` is not promoted. The best deterministic child reached success
  `0.874023`, gates `3.874023`, crash `0.0`, and Gate-2 crossing `1.0` at
  `ppo_r075_v4_fp32_floor4p21875_postmixed_calibration/rollp25_thrustrel1p00025.bin`.
  Floor `4.1953125` is also not promoted; its best exact child reached success
  `0.836914`, gates `3.802734`, crash `0.0`, Gate-2 crossing `0.965820`, and
  terminal radial `0.577001 m` at
  `ppo_r075_v4_fp32_floor4p1953125_bestmixed_calibration/rollp25_pitchm25_thrustrel0p9995.bin`.
- Rejected branches are recorded, not reusable parents: continuous speed-deficit
  reward, exit-velocity weight `50`, fixed-floor PPO, crossing-error weight `50`,
  a second low-rate PPO epoch, repeated decoder coordinate sweeps, and later
  checkpoints after the first mixed-PPO peak. BC, phase-column residuals, and
  further DAgger variants remain rejected by the prior exact evidence.
- A critical training audit found that log-std `-4` fails Gate 0 stochastically,
  while the same mean policy at log-std `-6` achieves exact-1024 stochastic
  success `0.993164`, gates `3.988281`, crash `0.0`, and Gate-2 crossing
  `0.995117` at fixed floor `4.1953125`. Log-std `-8` falls to success `0.600586`.
  This is a training diagnostic only; deterministic mean actions remain the
  deployment and promotion contract.
- A `524288`-transition run gives each of 1024 agents only 512 environment steps,
  so no episode reaches Gate 3. Consequently, late-reward variations produced
  byte-identical policy checkpoints. The first episode-complete log-std `-6` run
  used `2097152` transitions, reached Gate 3 in training, and saved checkpoints
  through step `2097152` under
  `ppo_r075_v4_fp32_floor4p1875_to4p1953125_logstdm6_episodecomplete_2m`.
  Its post-terminal deterministic screens are rejected: steps `1671168`,
  `1835008`, `1998848`, and `2097152` scored success `0.479492`, `0.522461`,
  `0.503906`, and `0.518555`, respectively, with crash `0.0` and Gate-2
  crossing `1.0`. The randomized-floor run learned a faster but unreliable
  final leg and did not promote floor `4.1953125`.
- The otherwise identical fixed-floor isolation is also closed and rejected.
  It trained for `2097152` transitions at floor `4.1953125` under
  `ppo_r075_v4_fp32_floor4p1953125_fixed_logstdm6_episodecomplete_2m`.
  Post-terminal exact-1024 screens at steps `1671168`, `1835008`, `1998848`,
  and `2097152` scored success `0.500977`, `0.497070`, `0.498047`, and
  `0.506836`, respectively. Every screen had crash `0.0` and Gate-2 crossing
  `1.0`; terminal radial remained `0.709995–0.714640 m`. Fixed versus randomized
  floor selection therefore does not explain the regression.
- The combined trace identifies a policy-update timing failure: horizon `32`
  applies PPO updates throughout the roughly 1,700-step flight. The measured
  `0.993164` stochastic parent is changed more than 50 times before its final
  approach. Merely extending another horizon-32 run repeats this failure mode.
- A default-zero `train.learning_start_timesteps` control now allows rollout and
  recurrent state to advance while withholding optimizer updates. Unit tests
  cover default behavior, the threshold boundary, agent-step semantics, and
  negative-value validation. Warmup rows also exposed and fixed a log-reducer
  assumption that every training row contains loss fields.
- The delayed-update isolation is closed and rejected. Its step `1605632`
  checkpoint is byte-identical to the log-std `-6` parent, proving that warmup
  did not alter weights; step `1638400` is the first updated child. Exact-1024
  deterministic success over all 15 updated children starts at `0.792969`,
  never exceeds that value, and ends at `0.306641`; crash is `0.0` throughout.
  Gate-2 crossing falls from `0.965820` to `0.431641`. The first four-minibatch
  Muon update is already too large for this sharp recurrent boundary.
- The 40-fold smaller micro-update is also closed and rejected. The first
  attempted run stopped at step `1310720` without updating because the new
  warmup path incorrectly entered the evaluation-stop branch; it produced only
  an unchanged step-`32768` checkpoint. The control flow was corrected and the
  regression suite increased to five tests. The valid rerun performed exactly
  one minibatch update at step `1638400` with learning rate `5e-8` and
  `replay_ratio=0.25`. Its exact-1024 deterministic report is success `0.831055`,
  gates `3.797852`, crash `0.0`, Gate-2 crossing `0.966797`, terminal radial
  `0.576288 m`, and completion time `23.865852 s`. This is below its `0.836914`
  parent, so update magnitude is not the remaining cause.
- Native recurrent replay is now terminal-aware. A selected terminal-at-start
  row receives zero initial state; the first later terminal ends the valid
  prefix; priority and advantage normalization ignore the suffix; PPO zeros
  suffix gradients and preserves its old ratios/values. Terminal-free chunks
  keep the previous arithmetic path. The FP32 CUDA build, 12 focused replay and
  learning-start tests, and all native drone regressions pass.
- The first valid-terminal learning test is closed and rejected. It preserved
  the log-std `-6` parent through `1696` steps per agent, then made one
  terminal-aware micro-update over the `1696–1728` step rollout. Its exact-1024
  deterministic report is success `0.833984`, gates `3.799805`, crash `0.0`,
  Gate-2 crossing `0.965820`, terminal radial `0.577667 m`, and completion time
  `23.947830 s`. This remains below the `0.836914` parent. Correct terminal
  credit therefore does not make the stochastic PPO direction improve the
  deterministic deployment objective.
- The opposite-gradient full-checkpoint line search is closed and rejected.
  Exact-1024 success at alphas `-1`, `-2`, `-4`, and `-8` is `0.835938`,
  `0.831055`, `0.833008`, and `0.824219`, respectively, with crash `0.0`.
  Neither sign of the terminal-aware PPO direction improves its `0.836914`
  base, so this direction must not be revisited.
- The strongest floor-`4.21875` child was cross-screened at floor `4.1953125`
  and is rejected there: success `0.601562`, gates `3.601562`, crash `0.0`,
  Gate-2 crossing `1.0`, terminal radial `0.698102 m`, and completion time
  `17.252958 s`. Its `0.874023` result at the higher floor is therefore a narrow
  non-monotonic operating pocket, not a generally stronger parent.
- The predeclared four-point map of that pocket is complete and the checkpoint
  family is closed. At floors `4.21484375`, `4.22265625`, `4.2265625`, and
  `4.23046875`, exact-1024 success is respectively `0.783203`, `0.830078`,
  `0.864258`, and `0.828125`; gates are `3.783203`, `3.830078`, `3.864258`, and
  `3.828125`; crash is `0.0` and Gate-2 crossing is `1.0` throughout. Terminal
  radial is `0.678190`, `0.679006`, `0.660990`, and `0.678107 m`; completion
  time is `22.409542`, `23.736542`, `24.717661`, and `23.669590 s`. None reaches
  the `0.90` retention threshold, and no more local floor points or calibration
  variants of this child may be tried.
- The required fixed floor-`6` baseline is complete for both surviving means.
  The retained floor-`4.1875` checkpoint and strongest closed floor-`4.21875`
  child each score success `0.0`, gates `1.0`, crash `0.0`, Gate-2 crossing
  `0.0`, missed-gate `1.0`, and completion time `0.0`. Their closest-gate range
  is `55.890087` and `55.963676 m`, final progress is `0.434597` and `0.433852`,
  and terminal radial is `1.067169` and `1.140901 m`, respectively. Both pass
  Gate 0 at only about `3.8 m/s`; the floor-`6` transition then changes the
  post-gate trajectory and both deterministically miss Gate 1. The high-floor
  child gives no transfer benefit, so the read-only floor-`6` parent comparison
  is closed.
- The retained mean with action `log_std=-6` also fails the bounded stochastic
  floor-`6` feasibility diagnostic: success `0.0`, gates `1.0`, crash `0.0`,
  Gate-2 crossing `0.0`, missed-gate `1.0`, terminal radial `1.079920 m`, signed
  right/vertical miss `-0.575259/+0.913737 m`, closest range `55.903316 m`, and
  final progress `0.434463`. This neither changes a discrete outcome nor
  improves the deterministic `1.067169 m` radial, so stochastic PPO at floor
  `6` is closed.
- Older floor-`6` evidence is now indexed here to prevent accidental repetition.
  The earlier floor-`4` calibrated mean `pitch0p8875_rollrel1p02.bin` also passed
  only one gate at floor `6` with terminal radial `1.038140 m`. Global thrust
  factors `0.9` and `0.8` worsened that miss to `1.285521` and `1.511626 m`.
  Combined with the documented action-head searches at lower floors, global
  pitch/roll/thrust calibration is not an authorized floor-`6` mechanism.
- Gate-1 telemetry is now complete. The first attempted paired evaluation was
  operationally invalid: adding ten log fields exceeded the release binding's
  fixed 96-entry dictionary and both processes aborted with a heap assertion.
  The failed logs are preserved with `capacity96_failed.log` suffixes. The
  generic vector dictionary now grows safely, the drone binding starts with 128
  entries, and native regressions cover both dictionary growth and the 60-step
  Gate-1 window. The FP32 rebuild and an exact-8 smoke passed before rerun.
- With the unchanged retained checkpoint, exact-1024 at floors `4.1875` and `6`
  records Gate-1 phase lengths `367.019531` and `263.257812` steps. The first
  60-step average pitch/roll/thrust/yaw actions are respectively
  `-0.038505/-0.399561/-0.264449/+0.000182` at `4.1875` and
  `-0.034572/-0.400115/-0.250813/+0.000182` at `6`. The early control response
  is therefore nearly unchanged despite the speed transition, while the signed
  terminal vertical error moves from the retained run's aggregate `-0.108984 m`
  to a Gate-1 miss of `+0.898483 m`. Full-phase averages are documented in the
  dated report. The evidence selects insufficient response to the observable
  gate down-rate, not a global output bias.
- The bounded gate down-rate encoder-column screen is closed. Scaling only
  `observation[2]` by `1.25`, `1.5`, and `2.0` leaves success `0.0`, gates
  `1.0`, crash `0.0`, and Gate-2 crossing `0.0` throughout while worsening
  terminal radial monotonically to `1.189951`, `1.373555`, and `1.678409 m`.
  Signed vertical miss worsens to `+1.049787`, `+1.250921`, and `+1.584536 m`;
  early thrust becomes less negative. No factor meets the discrete or `0.10 m`
  continuation rule. Do not add an inverse factor after observing this result;
  the down-rate column family is closed.
- The single final-MinGRU antithetic ES generation is complete and closes local
  deterministic parameter search around the retained mean. All eight seeded
  probes and both fitness-direction signs remain success `0.0`, gates `1.0`,
  crash `0.0`, and Gate-2 crossing `0.0`. Their terminal radial range is only
  `1.067160–1.067245 m`; the best probe improves the `1.067169 m` parent by
  about `0.000009 m`, nowhere near `0.10 m`. The launch first encountered a
  no-output directory-redirection error; no probe existed from that operational
  failure, and the unchanged predeclared generation was then run successfully.
- A source audit identifies the remaining fidelity defect. The existing
  transition floor instantaneously mutates world-X velocity after a gate, while
  the official-parity camera-motion filter correctly resets to zero when the
  official gate identity changes. The resulting discontinuity is hidden during
  the initial post-transition control frames and is not a physically
  controllable speed curriculum. Prior floor results remain valid under the
  legacy instantaneous setting and must not be relabeled.
- The opt-in `2.0 m/s²` physical-ramp screen is closed. Exact-1024 at floor `6`
  remains success `0.0`, gates `1.0`, crash `0.0`, and Gate-2 crossing `0.0`;
  terminal radial worsens to `1.560399 m` with signed right/vertical miss
  `-0.850784/+1.308030 m`, and Gate-1 phase length falls to `228.142578` steps.
  Assist plus the policy's own acceleration shortens the approach further. Do
  not try another ramp rate from this parent. The zero-default implementation
  remains isolated from all legacy evidence.
- The same audit exposes a more fundamental scope error: the transition code
  comment and v3385 fit motivate high closing speed for Gate 2 *after Gate 1*,
  but the current floor is applied after Gate 0 as well. Every floor-`6` failure
  above is consequently a Gate-1 miss caused by an unvalidated early boost.
- The activation-index correction passes its continuation rule. With legacy
  instant application retained but `sitl_gate_transition_from_gate_index=2`,
  exact-1024 at floor `6` advances from gates `1.0` to `2.0` with success `0.0`,
  crash `0.0`, and a Gate-2 terminal crossing on every episode. Gate-2 radial
  is `2.203115 m`, signed right/vertical miss is `+1.725223/+1.370109 m`, final
  progress is `0.675365`, and episode length is `1187.101562` steps. This is the
  first valid corrected-scope floor-`6` parent signature; it does not promote
  the checkpoint, but it supersedes the mis-scoped Gate-1 failure as the active
  training target.
- The sole corrected-scope terminal-aware PPO micro-update is closed and
  rejected. It trained for `1245184` transitions in `22.882400 s` at `43806`
  SPS, with one optimizer epoch (`KL=8.05e-8`). The step-`1245184` checkpoint is
  the only changed artifact; step `1277952` is byte-identical. Exact-1024 still
  records success `0.0`, gates `2.0`, crash `0.0`, Gate-2 terminal crossing
  `1.0`, and radial `2.203145 m`, slightly worse than the `2.203115 m` parent.
  Corrected-scope PPO must not be repeated with another update, sign, replay
  ratio, or learning rate.
- The corrected-scope coarse boundary is complete. At floor `4`, the unchanged
  retained checkpoint scores success `1.0`, gates `4.0`, crash `0.0`, Gate-2
  crossing `1.0`, Gate-2 crossing radial `0.531454 m`, and completion time
  `29.394688 s`. At floor `5`, it scores success `0.0`, gates `2.0`, crash
  `0.0`, Gate-2 terminal crossing `1.0`, radial `1.398045 m`, signed
  right/vertical miss `+1.028751/+0.946609 m`, and completion time `0.0`.
  Floor `4` is therefore the highest passing corrected-scope anchor and floor
  `5` is the first failing coarse anchor. The reports are in
  `ppo_r075_v4_fp32_corrected_scope_coarse/`; no intermediate floor was added.
- The one corrected-scope physical-ramp screen is closed and rejected. At
  floor `5`, activation index `2`, and `2.0 m/s²`, exact-1024 remains success
  `0.0`, gates `2.0`, crash `0.0`, and Gate-2 terminal crossing `1.0` while
  radial worsens from `1.398045` to `1.657418 m`. Signed right/vertical miss
  also worsens to `+1.283240/+1.048919 m`. It changes no discrete outcome and
  misses the required `0.10 m` improvement, so the physical-ramp mechanism is
  closed under both legacy and corrected activation scopes. Do not try another
  ramp rate.
- The one direction-preserving speed-floor screen is also closed and rejected.
  Its zero-default implementation scales the complete Gate-1 exit velocity
  when enabled, and native regressions plus the FP32 rebuild pass. At corrected
  floor `5`, exact-1024 remains success `0.0`, gates `2.0`, crash `0.0`, and
  Gate-2 terminal crossing `1.0`; radial worsens from `1.398045` to
  `1.984565 m` and signed right/vertical miss worsens to
  `+1.672432/+1.068225 m`. It changes no discrete outcome and fails the
  continuation rule. Do not try a blended or segment-tangent direction from
  this result.
- The single corrected-scope midpoint is complete and not retained. At floor
  `4.5`, exact-1024 records success `0.0`, gates `2.0`, crash `0.0`, Gate-2
  terminal crossing `1.0`, radial `0.980955 m`, signed right/vertical miss
  `+0.664541/+0.721489 m`, and final progress `0.687664`. This improves radial
  by `0.417091 m` versus floor `5` but remains `0.230955 m` outside the target
  aperture. Per the predeclared rule, do not add a finer floor point before a
  new training mechanism.
- The corrected-scope `log_std=-6` feasibility screen meets only the metric
  continuation rule. At floor `4.5`, exact-1024 stochastic sampling remains
  success `0.0`, gates `2.0`, crash `0.0`, and Gate-2 crossing `0.0`, but
  terminal radial improves from `0.980955` to `0.871013 m` and signed
  right/vertical miss improves to `+0.579753/+0.649861 m`. Gate-1 phase length
  rises from `389.099609` to `400.351562` steps. This does not promote or
  reopen PPO, but it shows that the near-boundary miss responds to the incoming
  transition state. No other log standard deviation is authorized.
- The one-step delayed impulse is closed. Its zero-default implementation,
  native regression, and FP32 rebuild pass, but corrected-scope floor `4.5`
  produces exactly the baseline actions and terminal state: success `0.0`,
  gates `2.0`, crash `0.0`, radial `0.980955 m`, and signed right/vertical miss
  `+0.664541/+0.721489 m`. Applying the impulse at the next control-step start
  precedes the same next physics integration, so the following motion sample
  was already equivalent. Do not try another delay length.
- The stale 32-field schema tooling is reconciled without changing runtime
  observations or checkpoint behavior. The metadata and dataset expander now
  agree with native/live columns `28/29 = Gate-2 down/down-rate` and `30/31 =
  Gate-3 right/right-rate`; the focused schema/smoke suite passed `28` tests.
- The Jacobian-CMA branch locked fresh authoritative baselines before mutation.
  Parent SHA-256 is `39e51c90d71775bf1f2986236f500e27a430755b56063f806d3a1c28f15bdaf3`.
  Corrected floor `4` exact-1024 again scores success `1.0`, gates `4.0`, crash
  `0.0`, Gate-2 radial `0.531454 m`, and completion `29.394688 s`. Corrected
  floor `4.5` again scores success `0.0`, gates `2.0`, crash `0.0`, Gate-2
  terminal radial `0.980955 m`, and final progress `0.687664`. Native/layout
  precision is `4/4`; no official run occurred.
- Exact native full-history trace capture and the pre-outcome Jacobian basis are
  complete. Trace hashes are `58f7d550…39a5` at floor `4` and
  `ffd3c901…217b` at floor `4.5`. The dimension-`12` cross-layer basis hash is
  `05c4c2bd…5793`, its FP32 Gram error is `5.96e-7`, and `log_std` plus the
  value row are excluded. Generation-zero action-only calibration observed no
  outcomes and fixed coefficient scale `0.0183414`; the floor-`4` anchor RMS
  constraint was binding. Native regressions and the focused trace/subspace/CMA
  tests pass.
- Constrained CMA generation `0` is complete. `22/24` candidates were trace
  feasible and all four actual floor-`4` leader checks retained success `1.0`
  with zero crashes. Candidate `17` improved floor-`4.5` development radial to
  `0.894246 m` and disjoint validation radial to `0.895128 m`, but made no
  Gate-2 crossing and improved the fixed held-out miss by only `0.087212 m`, below
  the `0.10 m` exact-screen rule. It is not retained or exact-1024 screened.
  CMA advanced once with sigma `0.946398` and covariance condition `1.21338`.
- Generation `1` is complete. Candidate `20` (SHA-256 `e93f2ac4…a4db`) passes
  exact-1024 floor `4` at success `1.0`/crash `0.0` and improves exact-1024
  floor-`4.5` terminal radial from `0.980955` to `0.875587 m`, satisfying the
  `0.10 m` metric rule. It still has success `0.0`, gates `2.0`, crash `0.0`,
  and zero Gate-2 crossings, so it is not retained. CMA advanced to generation
  `2` with sigma `0.873174` and covariance condition `1.30898`.
- Generation `2` is complete. Candidate `15` validates at radial `0.882343 m`
  with no crossing, `0.099997 m` better than the fixed held-out parent and therefore
  just short of the `0.10 m` exact-screen rule. All four floor-`4` leader checks
  pass. The scheduled audit finds `84.8239%` captured and only `15.1761%`
  discarded sensitivity energy, so the sole dimension-`24` expansion is not
  justified and is closed. CMA generation `3` uses the unchanged dimension-12
  basis, sigma `0.780211`, and covariance condition `1.46583`.
- Generation `3` is complete after adding a tested guard that forbids rejected
  samples from filling the CMA parent set. It had `15/24` feasible samples.
  Candidate `23` passes exact floor `4` but reaches only `0.878693 m` exact at
  floor `4.5`, with success `0.0` and zero crossings; this is worse than
  generation `1` candidate `20` and is rejected. CMA generation `4` uses sigma
  `0.687738` and covariance condition `1.47668`.
- Generation `4` is complete with `19/24` feasible candidates and four passing
  actual floor-`4` leader checks. Candidate `1` (SHA-256 `8cac7a5a…cc00`)
  validates at floor-`4.5` radial `0.883601 m`, success `0.0`, gates `2.0`, and
  zero crossings. Its `0.098739 m` improvement is below the `0.10 m`
  exact-screen rule, so it is rejected without an exact-1024 run. Generation
  `5` continues from saved state with sigma `0.634930` and covariance condition
  `1.50189`.
- Native trace replay is now explicitly verified for both floor traces. Maximum
  callable/native operational-action error is `1.85e-6`, maximum rollout-start
  state error is `2.38e-7`, and eight terminal resets are reproduced in each
  trace.
- Generation `5` is complete with `16/24` feasible candidates. Candidate `17`
  validates at radial `0.882754 m`, only `0.000414 m` short of the exact-screen
  threshold, with no crossing; it is rejected without exact evaluation.
- Generation `6` is complete with `15/24` feasible candidates. Candidate `4`
  exact-screens at floor `4.5` radial `0.878673 m`, success `0.0`, and zero
  crossings while exact floor `4` remains success `1.0`; it is rejected.
- The bounded branch is complete. Final generation `7` candidate `18`
  (SHA-256 `08b93de3…0c7c`) is the best metric-only result: exact floor `4.5`
  radial `0.867062 m`, success `0.0`, gates `2.0`, crash `0.0`, and zero Gate-2
  crossings. Exact floor `4` remains success `1.0`. All `192` candidate vectors
  and checkpoints are unique and indexed in
  `run_dim12_seed3385/rejected_candidates.json`; no child is retained.
- Final CMA state is generation `8`, sigma `0.550017`, covariance condition
  `2.22236`, and stop reason
  `maximum_generation_budget_exhausted_without_promotion`. The mechanism is a
  documented negative and may not be restarted under a new seed, sigma, tensor
  subset, or renamed variant.
- Final verification passes `53` relevant Python tests, native drone-race
  regressions, all `192` exact FP32 reconstruction/exclusion checks, the
  trace/basis/final-artifact audit, and `git diff --check`.
- No checkpoint has passed floor `6`, `8`, or `10`, camera dropout, or exact-4096
  native promotion. No official-simulator run has been launched, and no
  checkpoint is eligible for it.

Current direction:

- Retain the unchanged checkpoint at corrected-scope floor `4`; it passes the
  deterministic exact-1024 contract with success `1.0` and crash `0.0`. Floor
  `5` is the active corrected-scope failure boundary.
- The physical ramp and direction-preserving transition mechanisms are closed;
  retain the legacy instantaneous direction for the active curriculum.
- Floor `4.5` is the active near-boundary failure, but no finer deterministic
  floor screen is authorized.
- Ramp, direction, and delay variants of the transition floor are closed.
- The Jacobian-guided constrained CMA branch is closed after its full frozen
  eight-generation budget. It improved exact radial miss to `0.867062 m` but
  produced no ordered Gate-2 crossing and no floor-`4.5` success. Retain the
  original floor-`4` parent; do not treat the metric-only child as a promotion.
- Exactly one next action: stop this branch. Do not launch the official
  simulator, retry the mechanism, or silently substitute any previously closed
  optimizer.
- Do not repeat log-std `-4`, sub-episode late-reward changes, the randomized
  or fixed-floor episode-complete `-6` runs, the 15-update delayed branch,
  further learning-rate/replay-ratio tuning of the same invalid replay path,
  repeated PPO along the now-correct but deterministic-negative direction,
  either sign of the terminal-aware micro-update direction,
  action-head sweeps already represented in the dated report, BC/DAgger, or the
  rejected reward branches.
- Only after floors `6`, `8`, and `10`, dropout, and exact-4096 native promotion
  may the unchanged checkpoint return to the official simulator.

Intermediate milestone:
- A full policy passes official gate 2 while controlling from race start. A
  hybrid gate-2 pass does not satisfy this milestone.

Completion:
- A single policy controls all timed-flight commands from race start through an
  official full-course finish.
- At least 8 of 10 official runs are valid and collision-free.
- Reports prove command-rate, heartbeat, telemetry, camera, ordered-gate,
  completion-time, and invalid-run compliance.
- The full policy's median valid time becomes the optimization target for future
  time trials.
```


## Competitive-lap live status: N191 Candidate 051

- The active VQ1/R1 simulator remains the same responsive PID `24476`; no
  relaunch occurred. Candidate 051 used the tracked disarm-first MAVLink reset:
  normal disarm, `100 ms` wait, then one `COMMAND_LONG 31000` with
  confirmation and params 1--7 zero. Boot rolled `247975 -> 3412 ms`, fresh
  race start was `3317 ms`, index `0`, finish `-1`.
- Candidate 051 passed Gates 1 and 2, then collided on Gate 3 at `10.25 s`
  (collision `1001`, impact `10.746137619018555`). It stopped at index `2`,
  finish `-1`, with healthy `62.024 Hz` control and no transport faults.
  N191's Gate-2 floors never activated; the pass used the unchanged N189 path.
- The isolated Gate-3 hazard is terminal bank after projection is already safe:
  roll reached `-0.067/-0.467/-0.800` at `3.664/3.266/2.712 m`. Candidate
  051 is rejected without retry. Candidate 052 and FullLap remain unauthorized
  until an offline-separated source-pinned terminal-attitude rule passes tests,
  exact reset proof, and passive zero-command shadow.
- Authoritative smoke/attempt/summary/post-stop hashes are
  `620a71944b57d7278d6038294709f0596feb9f0b0c41002d519e5e79d6c6828e`,
  `328140427c05274c7df417684843e1a4e6056cdc5e89ea3e5c5fd58f51d16753`,
  `f8145cd38e6673c8059c1676961977787f9752895b9d88faa76519794d92d3e3`,
  and `19670708617bdaa0e0e47463d29daf94025139b309fdde7f2d2b27979c52ac77`.


## N192 Gate-3 Terminal Safe-Level — Offline Freeze

- N192 evaluates exact source-pinned N191 once, then reuses N184's already
  tested terminal-aperture projection as a final Gate-3 roll-only governor.
  While visible, closing at least `1 m/s`, and within `6 m`, it projects
  right/down to the `2 m` terminal plane. If absolute projected right is at
  most `0.5 m` and absolute projected down at most `1.0 m`, it latches
  normalized roll to `0.0` until official gate transition. Pitch, thrust, and
  yaw are unchanged; N191's Gate-2 rules and N189's preceding Gate-3 control
  remain exact.
- Offline replay passes with no blockers. It changes zero actions on protected
  Gate-3-pass Candidates 016, 018, 020, 021, and 022, and zero actions on
  recent fixed-point Candidates 045 and 048. On Candidate 051 alone among
  those current paths it changes exactly three roll commands, first at
  `9.907 s` and `3.664 m`, where projection is right `-0.255 m` and down
  `-0.072 m`; it replaces `-0.067/-0.467/-0.800` with level roll. It also
  activates on older failed terminal-bank traces 019, 024, 040, and 041. That
  supports mechanism separation but is not counterfactual pass proof.
- Actual Windows focused controller/verifier/composition/runner tests pass
  `28/28`, and the PowerShell runner parses. Policy/verifier/offline-report/
  runner SHA-256 values are
  `3750e1620f4a7d244b93c370f31ec135a89572e64a60888d7ea032c244949fae`,
  `34c77c5357f6c33ca34298e4cb93dea0f3dfbd83f5b47c141088b22b9b893cc5`,
  `17c2d14cbae9bbb935030ceb881b4b29ce406f6da3b963e52a98afabc9ce0b08`,
  and `7e6f5faf7bdf72c87b9e9b9843065f5dd09b8df83feb60996b207d82c1cef2e0`.
- No simulator control command occurred during N192 development. Candidate 052
  and FullLap remain unauthorized. Next only: keep reused PID `24476`, recover
  sole VQ1/R1 with `-NoRelaunch -SkipLoginClick` only if inactive, then use
  the current tracked normal-disarm plus learned-target MAVLink
  `COMMAND_LONG 31000` reset. Require boot rollback over `1000 ms`, fresh
  nonnegative race start, index `0`, finish `-1`, and visible Gate 1. Then
  run passive zero-command Shadow 053 with `-Gate3TerminalLevelN191`; it must
  send zero reset, arm, setpoint, and disarm commands and pass exact hashes,
  parity, cadence, telemetry, camera, and fault checks before a separately
  preregistered bounded flight.


## N192 Live Reset Proof and Passive Shadow 053 — Passed

- The simulator remained responsive on the original PID `24476`; no process
  was launched. Candidate 051 had stopped inactive at base/status `65/3`, so
  the sole VQ1/R1 UI was recovered with `-NoRelaunch -SkipLoginClick` while
  preserving the process. The command-free ready capture then showed
  base/status `193/4`, index `0`, finish `-1`, visible Gate 1, and old
  boot/race epoch `963417/954276 ms`.
- The current tracked reset learned target IDs, sent normal disarm, waited
  `100 ms`, and sent exactly one MAVLink `COMMAND_LONG 31000` with
  confirmation and params 1--7 zero. Boot rolled `963417 -> 3069 ms`; the
  fresh race-start epoch was `3286 ms`. A subsequent command-free proof
  showed boot `18362 ms`, base/status `193/4`, index `0`, finish `-1`,
  visible Gate 1, healthy telemetry, and no collision.
- Passive `n192_gate3_terminal_level_n191_shadow_053` passed while sending
  zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence was
  `602/10.0 = 60.1 Hz` with zero violations. All `294/294` trace samples
  replayed at `29.353 Hz` with maximum action error
  `2.522184371911429e-7`; every deployment hash matched. It observed `2431`
  telemetry messages and `124/124` camera detections, with zero collision,
  dropout, drain hit, malformed message, decode failure, or missing quad.
- UI-ready/reset/reset-ready/shadow/parity SHA-256 values are
  `90af19972fa8580291e777997cf9aff12196983ccf3423fb8b7b0d0c2ff85239`,
  `25aa8ce73568901c808790802310a3f432fb74704060cd6b8d7e12dd78f065f2`,
  `f38664acca9b8980e1e841d3b1c3a631a9f9f2221f99094fa91ab5c7d16833ea`,
  `f90be3a8552cb58e09b098c6a35169a8af8bf093db839d94c3ea41cd97906d7b`,
  and `114e715ac0bb317e78b133f6caa8c54b71aea95db72e1d3f3fdc94ac0c7a9e5d`.
- Authorize exactly one Candidate 052 `BoundedGate3` under tag
  `n192_gate3_terminal_level_n191_bounded_052`: frozen N192 hashes, actual
  Windows Python, reused PID `24476`, one detected normal-disarm-then-command-
  `31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop official index `3`. Accept
  only clean index `3`, finish `-1`, cadence `[50,100) Hz`, exact hashes,
  zero rate/collision/invalid/dropout/drain fault, one exit disarm, and a
  command-free post-stop proof. Reject failure without unchanged retry.
  FullLap remains unauthorized.



## N192 Candidate 052 — Gate-2 Collision Before N192 Activation

- Candidate 052 ran exactly once on reused responsive PID `24476`; no process
  launch. The tracked runner sent normal disarm before exactly one learned-target
  MAVLink `COMMAND_LONG 31000` with confirmation/params 1--7 zero. Reset was
  detected in `0.5 s`: boot rolled `130379 -> 3388 ms`, fresh race start
  was `3305 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- Only Gate 1 passed officially. Two camera-range pass detections were counted,
  but the authoritative index remained `1`; they are not two official gates.
  Gate 2 then produced collision `1001` (threat `2`, impact
  `8.721649169921875`) at `7.39 s`. The run stopped at index `1`, finish
  `-1`. Transport was healthy: `449/7.265 = 61.666 Hz`, zero rate,
  dropout, drain, or malformed-message faults, `2571` telemetry messages,
  `94` frames, and `91` detections. Command-free post-stop proof sent no
  reset/control command and confirmed base/status `65/3`, index `1`,
  finish `-1`.
- Neither N191 Gate-2 floor activated or changed a command, and there were zero
  Gate-3 samples, so N192 did not activate. This run therefore supplies no live
  evidence for or against terminal leveling. The last visible Gate-2 state was
  `2.183 m` forward, right `+0.481 m`, down `-0.100 m`, closing
  `7.924 m/s`, right rate `-0.696 m/s`, and down rate `-3.332 m/s`.
  Candidate 051 passed Gate 2 from a nearby lateral/yaw state but a materially
  smaller vertical rate (`-2.577 m/s`), isolating a fast terminal vertical
  deficit for offline study.
- Smoke/attempt/summary/post-stop/Gate-2-diagnostic/transition-comparison
  SHA-256 values are
  `12327cb380fd4b81b245a689432c2d192f9d9e0f345b164edb3db928ffb6a186`,
  `c761d22d3b2023092bd16548ec7d266121906cd3a8ad49d788a4d984ae7e0f6e`,
  `bd7262bc42d98e68df5f4fbb4f30f7b76a21cc67e99ade874e40bfa3287a0929`,
  `6389bc8aa428d16a6aa128f5d1a2a6dd0570d3a4a816da03b463bf270f474e6f`,
  `33327335f1be632b5952b4ddee26f72235c61de2f3494f09d68fd62223c70dce`,
  and `a91d97a92d4b7956ab3e5702469ab5587d6db5f20ad997752aff49665cac6cd0`.
- Candidate 052 is rejected without unchanged retry. Candidate 053 and FullLap
  are unauthorized pending an observable, offline-separated Gate-2 mechanism,
  hashes, focused tests, exact reset proof, and passive zero-command shadow.


## N193 Rapid Gate-2 Vertical-Deficit Latch — Offline Freeze

- N193 evaluates exact source-pinned N192 once and adds one Gate-2 thrust-only
  response to the failure state unique to Candidate 052. While visible,
  closing, and between `3--7 m`, it triggers only when down rate is below
  `-2.9 m/s`, plane-projected down is below `-1.0 m`, and inherited
  normalized thrust is still at or below `+0.05`. It then latches normalized
  thrust to `+1.0` until projected down recovers to `-0.4 m`, visibility is
  lost, or the official gate changes. Pitch, roll, and yaw are untouched; all
  N191/N192 safeguards remain exact.
- Replay across Candidates 037--052 passes with exact activation separation.
  Candidates 037--051 have zero activations and zero changes. Candidate 052
  triggers once at `6.859 s`, `5.385 m` forward, down rate
  `-2.941 m/s`, projected down `-1.070 m`, and inherited thrust near zero.
  The latch changes exactly five consecutive thrust commands before the
  recorded collision, with zero non-Gate2, non-thrust, or invalid changes.
  This gives about `0.53 s` of counterfactual correction authority but does
  not prove a crossing.
- Actual Windows focused N187--N193 controller/verifier/composition/runner
  tests pass `35/35`; the PowerShell runner parses. Policy/verifier/offline-
  report/runner SHA-256 values are
  `9da6cde4ed3c5a25f2c16872427ab6aab15ed57e379399ab725fee28b34869f2`,
  `5f1ace160d4a752a34a959fa5bdb3bd1e99713ecf966875e4138fdf37f3e2913`,
  `f5df695093a59d2ffae5a766c2b92681ab97b5322de90bc0bbb0078118920fec`,
  and `4391572c67f7bc4b6c332c06a3daf4c14c0e682eee90b8858e4eb9b6f431ba8d`.
  The broader vertical-deficit study and its analyzer hash to
  `ebbbb36023b77566bf11953c6cac8bf2d761871362626851cfbebd893ea9ff8d`
  and `d5a2efd02ee752cb74e55f91983e23994088924c14ac869f05234f76b7c25934`.
- No simulator control command occurred during N193 development. Candidate 053
  and FullLap remain unauthorized. Next only: keep PID `24476`, recover sole
  VQ1/R1 with `-NoRelaunch -SkipLoginClick` because Candidate 052 stopped
  inactive, then run the current tracked normal-disarm plus learned-target
  MAVLink `COMMAND_LONG 31000` reset. Require boot rollback over `1000 ms`,
  fresh nonnegative race start, index `0`, finish `-1`, and visible Gate 1.
  Then run passive zero-command Shadow 054 with
  `-Gate2RapidVerticalDeficitN192`; it must pass exact hash/parity, cadence,
  telemetry, camera, and fault checks before a separately preregistered flight.


## N193 Live Reset Proof and Passive Shadow 054 — Passed

- PID `24476` remained responsive and was never relaunched. Candidate 052 had
  stopped inactive, so `-NoRelaunch -SkipLoginClick` recovered the sole
  VQ1/R1 session on the same process. Command-free readiness showed base/status
  `193/4`, index `0`, finish `-1`, visible Gate 1, and old boot/race epoch
  `588707/568695 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000` with confirmation and
  params 1--7 zero. Boot rolled `588707 -> 2963 ms`, fresh race start was
  `3283 ms`, and a later command-free proof showed boot `21754 ms`,
  base/status `193/4`, index `0`, finish `-1`, visible Gate 1, healthy
  telemetry, and no collision.
- Passive `n193_gate2_rapid_vertical_deficit_n192_shadow_054` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `603/9.984 = 60.296 Hz`, zero violations. All `277/277` samples
  replayed at `27.659 Hz` with maximum action error
  `2.1852722167925442e-7`; every deployment hash matched. It observed `2427`
  telemetry messages and `124/124` detections with zero collision, dropout,
  drain hit, malformed message, decode failure, or missing quad.
- Ready/reset/reset-ready/shadow/parity SHA-256 values are
  `51b173069af8ccd25e199cef22261385281988805ea31b36604117340dd0ed48`,
  `c5bcb4931e31a139349b8e130bb908635a1e0401686d8455c547c57f0955935e`,
  `15e4fc1e10c8b265c0611aece169ef0fc9801dee854d19e9b6587f19c6fe7897`,
  `661951f8a22f7a73db6694edf9fcc8d9e2f9bdaba4833cfc8a9c4d2bb48ea711`,
  and `656ffe5eb255670a7d9372b939799a98e153e3c6d79398bc2d128659e87ba9ef`.
- Authorize exactly one Candidate 053 `BoundedGate3` under tag
  `n193_gate2_rapid_vertical_deficit_n192_bounded_053`: frozen N193 hashes,
  actual Windows Python, reused PID `24476`, one detected disarm-then-command-
  `31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-valid
  `1/1`, and target/minimum/authoritative stop index `3`. Accept only clean
  index `3`, finish `-1`, cadence `[50,100) Hz`, exact hashes, zero rate/
  collision/invalid/dropout/drain fault, one exit disarm, and command-free
  post-stop proof. Reject failure without unchanged retry. FullLap remains
  unauthorized.


## N193 Candidate 053 — Clean Gate-3 Stale-Observation Stall

- Candidate 053 ran once on reused responsive PID `24476`; no process launch.
  The disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset was
  detected in `0.5 s`: boot rolled `114460 -> 3414 ms`, fresh race start
  was `3295 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- Gates 1 and 2 passed officially. Gate 3 did not pass and the run ended at the
  `45 s` bound with authoritative index `2`, finish `-1`, and no in-run
  collision. Transport was healthy: `2771/44.875 = 61.727 Hz`, zero rate,
  dropout, drain, or malformed-message faults, `11114` telemetry messages,
  `574` frames, and `183` detections. N193's rapid Gate-2 branch did not
  activate because this path never reproduced Candidate 052's rapid deficit;
  N192's terminal-level rule also did not activate.
- The camera pass detector counted three range events, including a Gate-3 event
  at `10.453 s`, but official index remained `2`; detector events are not
  authoritative gate passes. The last real detection aged to `30.703 s` by
  run end. Meanwhile the observable Gate-3 motion state froze for `271`
  consecutive inference samples from about `15.609 s`: forward `2.059 m`,
  right `+0.972 m`, down `-0.780 m`, closing `8.520 m/s`, right rate
  `+1.546 m/s`, down rate `-3.261 m/s`, while inherited roll stayed
  `-1.0`. The same five-sample frozen condition occurs in no protected
  Gate-3-pass trace or Candidate 037--052. This is a stale held-observation
  control loop after a missed crossing, not a live target approach.
- The command-free post-stop capture sent no reset or control command. After
  exit disarm, v3385 had already cleared the race epoch to start `-1`/index
  `0` and reported low-threat contact `1001` with impact `0.03224`; this
  is passive post-exit ground contact and is not substituted for the smoke
  report's zero in-run collisions.
- Smoke/attempt/summary/post-stop/transition/terminal-level/control-sequence/
  frozen-study SHA-256 values are
  `3e5ade2e0b9fb9d37d295673d0aefe43de021b00750bb3f06b065643f5a5539b`,
  `b8c39393b715ada7e014e16dfd0c48078dd46d2adc19c3717d75707f1f9a82a1`,
  `99b8eecd122a26570c9edba4bdd357b5f6a35fe49eeee54a34ad0a738899fc96`,
  `e54d0ff92f1417ada5d73c12604ec3298f2c5a5bb785d0658f35f1c15db36277`,
  `3ce1794e37d4b5623cf9f3b92228a26f431e7b3e617ac78b19f3b5e1d848b877`,
  `5de54f9a4eb942cd3466085c9cb3fda6b23d2d6a06d32b43212d8ee60d333c9f`,
  `c445747a70f76a5af4af408a010dea70d3f22b3e8b91b26a93f14c4d184e0cbc`,
  and `a0b6506edbb03e1817ef4affe70f8ba4700b8aa523817c21e2ab094243ce2ded`.
- Candidate 053 is rejected without unchanged retry. Candidate 054 and FullLap
  remain unauthorized pending an offline-separated stale-observation recovery,
  hashes, tests, exact reset proof, and passive zero-command shadow.


## N194 Gate-3 Frozen-Observation Recovery — Offline Freeze

- N194 evaluates exact source-pinned N193 once and adds one recovery for the
  Candidate-053 failure state. On official Gate 3 only, while visible,
  closing at least `1.0 m/s`, and within `3.0 m`, it watches features
  `0,1,2,11,12,13,14`. Five identical consecutive samples within
  `1e-7` activate recovery: normalized roll becomes `0.0` and yaw scans
  `0.25 rad` toward the last measured bearing. Inherited pitch and thrust
  remain exact. Any fresh feature motion, visibility loss, invalid approach
  state, or official gate transition releases the recovery immediately.
- Offline replay passed with exact separation. Protected Gate-3-pass Candidates
  016, 018, 020, 021, and 022 plus Candidates 037--052 have zero activations
  and zero changes. Candidate 053 activates once at `15.609 s`, on the fifth
  frozen sample, and changes exactly `267` roll/yaw commands through the end
  of its 271-sample frozen run. There are zero non-Gate3 changes, zero pitch or
  thrust differences, and zero invalid outputs. This is counterfactual recovery
  authority, not proof of Gate-3 reacquisition or passage.
- Actual Windows focused N189--N194 controller/verifier/runner tests pass
  `26/26`; the PowerShell runner parses. Policy/verifier/offline-report/runner
  SHA-256 values are
  `a68c07a2a55f61fc1b9ad6522a27e3f6b0ba206dab6dd554392f8c5da417a6a7`,
  `a92d2159698b9f19336a160d99d11f5eb865d4f32aeaebd5d3b0f4451a563178`,
  `42f0bed223539702b5109954451a523803d30051c25ec1ca34526ced77d8b620`,
  and `92f14881dfe55e57c61badeff2ffe752e77d6d30abc15d02d9f0079783e23737`.
- No simulator control command occurred during N194 development. Candidate 054
  and FullLap remain unauthorized. Next only: preserve reused PID `24476`,
  recover the sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` if inactive,
  then send normal disarm followed by exactly one learned-target MAVLink
  `COMMAND_LONG 31000` with confirmation and params 1--7 zero. Require boot
  rollback over `1000 ms`, a fresh nonnegative race epoch, index `0`, finish
  `-1`, and visible Gate 1. Then run passive zero-command Shadow 055 with
  `-Gate3FrozenObservationRecoveryN193`; only a clean hash/parity, cadence,
  telemetry, camera, and fault pass may authorize a separately preregistered
  bounded flight.


## N194 Live Reset Proof and Passive Shadow 055 — Passed

- PID `24476` remained responsive and was never relaunched. Candidate 053 had
  stopped inactive at base/status `65/3` with race start `-1`; the sole
  VQ1/R1 UI was recovered using `-NoRelaunch -SkipLoginClick` on the same
  process. Command-free readiness then showed base/status `193/4`, index
  `0`, finish `-1`, visible Gate 1, and old boot/race epoch
  `854041/840827 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000` with confirmation and
  params 1--7 zero. Boot rolled `854041 -> 3163 ms`; fresh race start was
  `3303 ms`. A later command-free proof showed boot `21198 ms`,
  base/status `193/4`, index `0`, finish `-1`, visible Gate 1, healthy
  telemetry, and no collision.
- Passive `n194_gate3_frozen_observation_recovery_n193_shadow_055` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `607/9.985 = 60.691 Hz`, with zero violations. All `307/307` trace
  samples replayed at `30.7 Hz` with maximum action error
  `2.627761840900966e-7`; every deployment hash matched. It observed `2424`
  telemetry messages and `125/125` camera detections, with zero collision,
  dropout, drain hit, malformed message, decode failure, or missing quad.
- UI-ready/reset/reset-ready/shadow/parity SHA-256 values are
  `2c960abeff7389128b3fe8a2ceff3ca95bbf38e2b80611d25d306adfba8a56bc`,
  `71f187a5f5916ff0e8f81e187d029c4c7473c8f683cb3be6516eb2660cf2ba07`,
  `704a30c02ca8d7ba34fec862762da0a13f71b6d25dfd796b8162ff7df700bd73`,
  `efb732f201d7fbd2da75effd48c938ced36bd748581499c2fc988af3c28d3b1a`,
  and `641710fdd64cbd683b3e47fcb6e313d59ec4fef8cca12da5635a3de0bbbbf06c`.
- Authorize exactly one Candidate 054 `BoundedGate3` under tag
  `n194_gate3_frozen_observation_recovery_n193_bounded_054`: frozen N194
  hashes, actual Windows Python, reused PID `24476`, one detected normal-
  disarm-then-command-`31000` reset, requested `80 Hz`, maximum `45 s`,
  repeats/min-valid `1/1`, and target/minimum/authoritative stop official
  index `3`. Accept only clean index `3`, finish `-1`, cadence
  `[50,100) Hz`, exact hashes, zero rate/collision/invalid/dropout/drain
  fault, one exit disarm, and a command-free post-stop proof. Reject failure
  without unchanged retry. FullLap remains unauthorized.



## N194 Candidate 054 — Clean Gate-2 Frozen-Observation Stall

- Candidate 054 ran exactly once on reused responsive PID `24476`; no process
  launch. The disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset
  was detected in `0.5 s`: boot rolled `232191 -> 3406 ms`, fresh race
  start was `3291 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- Gate 1 passed officially at race time `4.847601890 s`. Gate 2 did not pass;
  the attempt reached its `45 s` bound at authoritative index `1`, finish
  `-1`, with no in-run collision. Transport remained acceptable:
  `2698/44.906 = 60.059 Hz`, zero rate violations, `11089` telemetry
  messages, `560` frames, `106` detections, one allowed telemetry dropout,
  and zero drain-limit or malformed-message faults.
- The last real detection aged to `36.328 s` by run end. Despite that, the
  policy observation continued to claim Gate 2 visible and froze exactly for
  `312` consecutive logged inference samples from `9.891--44.860 s` at
  forward `4.183 m`, right `+0.821 m`, down `-1.351 m`, closing
  `4.296 m/s`, right rate `-0.690 m/s`, and down rate `-3.531 m/s`.
  The action also froze near pitch `-0.2`, roll `-0.251`, thrust `+1.0`,
  and yaw `-0.00095`. The five-sample Gate-2 frozen condition occurs only in
  Candidates 050 and 054 among Candidates 037--054; both are clean Gate-2
  stalls, while all protected Gate-2 passes remain inactive. N194's Gate-3-
  only recovery never activated and therefore received no live test.
- The command-free post-stop proof sent no reset or control command and showed
  base/status `65/3`, official index `1`, finish `-1`, and no collision.
  Smoke/attempt/summary/post-stop/vertical-study/frozen-study SHA-256 values are
  `7901d9eaaae4d8bb01a70b076aba5306845c0635313ee5f16092b1666e79513d`,
  `f74d95d1113f29ba531fa0e10607c4a8d6c148e35cd98262e284ea3be5637da9`,
  `555fef017744375a8121056b49dc715dba5cf1deef8be834c4057517bc78613e`,
  `54f74d1e9586268e1a4a87f8ece50a4785783ae43ff91c04e4570f60e3eaffcd`,
  `fbec238a75b79d066f6b5b8cc1e3d6ad9f1ec7562af50c5d987b8afefd403850`,
  and `819d5e1e0a58ad6709ed8eb82e021066990dfeb9c6b17e9d05f460cb03f402f3`.
  The Gate-2 frozen analyzer hashes to
  `578cad0bfb39eb1a7547f09c058666caa3daa9e96ecf848b3b72714191b84f17`.
- Candidate 054 is rejected without unchanged retry. Candidate 055 and FullLap
  remain unauthorized pending an observable, offline-separated Gate-2 stale-
  observation recovery, pinned hashes, focused tests, exact reset proof, and
  passive zero-command shadow.


## N195 Gate-2 Frozen-Observation Dropout — Offline Freeze

- N195 evaluates exact source-pinned N194 and changes no actuator channel
  directly. On official Gate 2 only, while the observation claims visibility,
  closes at least `1.0 m/s`, and is within `6.0 m`, it watches features
  `0,1,2,11,12,13,14`. After eight equal samples within `1e-7`, it changes
  only `gate_visible` from `1.0` to `0.0` before N194 inference. This
  releases N193's visibility-gated thrust latch and invokes the recurrent
  prefix's existing trained vision-dropout behavior. Fresh feature motion,
  raw visibility loss, an ineligible approach, or a gate transition releases
  the sanitizer immediately.
- Offline replay across Candidates 037--054 passes with exact separation.
  Candidates 037--049 and 051--053 have zero activations and zero changes,
  including every protected Gate-2 pass. Candidate 050 activates once at its
  eighth frozen logged sample (`11.594 s`) and sanitizes `303` samples;
  Candidate 054 activates once at `10.547 s` and sanitizes `306` samples.
  Every changed sample is Gate 2 and field 10 only; all other observation
  fields remain bit-identical and all outputs are finite. This demonstrates
  stale-input removal, not counterfactual Gate-2 passage.
- Actual Windows focused N189--N195 controller/verifier/runner tests pass
  `32/32`; the PowerShell runner parses. Policy/verifier/offline-report/runner
  SHA-256 values are
  `46a075eafd49a1e44a14e32af458a26e8d411e4cb507b510c6aaaa7ee6b97b4a`,
  `ef8a513c421870436f65d0c2e9f725cab3443610573b64a15c2470a68e1b45d8`,
  `d10fc3364865d2c66a45d12af028a71542a80b1f1daabe731d44fdce1d99d2b0`,
  and `4465f0e915a70510d63ec0c09f307b1cd0405b2f42431e811cca6b9b37259a30`.
- No simulator control command occurred during N195 development. Candidate 055
  and FullLap remain unauthorized. Next only: preserve PID `24476`, recover
  the sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` because Candidate 054
  stopped inactive, then send normal disarm followed by exactly one learned-
  target MAVLink `COMMAND_LONG 31000` with confirmation/params 1--7 zero.
  Require boot rollback over `1000 ms`, a fresh nonnegative race epoch, index
  `0`, finish `-1`, and visible Gate 1. Then run passive zero-command Shadow
  056 with `-Gate2FrozenObservationDropoutN194`; only clean hash/parity,
  cadence, telemetry, camera, and fault checks may authorize a separately
  preregistered bounded flight.


## N195 Live Reset Proof and Passive Shadow 056 — Passed

- PID `24476` remained responsive and was never relaunched. Candidate 054 had
  stopped inactive at base/status `65/3`; the sole VQ1/R1 UI was recovered
  using `-NoRelaunch -SkipLoginClick` on the same process. Command-free
  readiness then showed base/status `193/4`, index `0`, finish `-1`,
  visible Gate 1, and old boot/race epoch `954041/934020 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000` with confirmation and
  params 1--7 zero. Boot rolled `954041 -> 3035 ms`; fresh race start was
  `3298 ms`. A later command-free proof showed boot `18571 ms`,
  base/status `193/4`, index `0`, finish `-1`, visible Gate 1, healthy
  telemetry, and no collision.
- Passive `n195_gate2_frozen_observation_dropout_n194_shadow_056` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `614/10.0 = 61.3 Hz`, with zero violations. All `280/280` trace
  samples replayed at `27.955 Hz` with maximum action error
  `2.1868190765161888e-7`; every deployment hash matched. It observed `2427`
  telemetry messages and `125/125` camera detections, with zero collision,
  dropout, drain hit, malformed message, decode failure, or missing quad.
- UI-ready/reset/reset-ready/shadow/parity SHA-256 values are
  `8d8122f797cf4c42d411ee6d6d41d8ae82da0d3429430825f8d0a6b6f9fd948f`,
  `af73ed4f804e724737f6eadd6fde105f3fbd92293afb2cd2cc8527832b7f738e`,
  `4246c29885f38e863934e42e0e9326ed21d3ee024e75791cb4ce1fb7b3e1dda7`,
  `805626a5fc146b7060cd57713c6774be9589e4d4046dec228ce571bfcb5c379f`,
  and `94b19c2a1616347449a3e4dae898d12551a6ee626dacf1411ed38081add091a6`.
- Authorize exactly one Candidate 055 `BoundedGate3` under tag
  `n195_gate2_frozen_observation_dropout_n194_bounded_055`: frozen N195
  hashes, actual Windows Python, reused PID `24476`, one detected normal-
  disarm-then-command-`31000` reset, requested `80 Hz`, maximum `45 s`,
  repeats/min-valid `1/1`, and target/minimum/authoritative stop official
  index `3`. Accept only clean index `3`, finish `-1`, cadence
  `[50,100) Hz`, exact hashes, zero rate/collision/invalid/dropout/drain
  fault, one exit disarm, and a command-free post-stop proof. Reject failure
  without unchanged retry. FullLap remains unauthorized.


## N195 Candidate 055 — Clean Official Gates 1--3 Milestone

- Candidate 055 ran exactly once on reused responsive PID `24476`; no process
  launch. The disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset
  was detected in `0.5 s`: boot rolled `139093 -> 3336 ms`, fresh race
  start was `3284 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- The vehicle passed official Gates 1, 2, and 3 cleanly. Authoritative index
  reached `3` with official last-gate race time `10.537808418 s`; the
  configured index-3 stop triggered at controller elapsed `10.546 s` and
  finish remained `-1`. Camera-range pass events at `4.375/7.078/10.359 s`
  are diagnostic only. There was no in-run collision or invalid condition.
- Transport and perception were healthy: `616/10.421 = 59.015 Hz`, zero
  command-rate violations, `3267` telemetry messages, `131` frames and
  `131` detections, and zero telemetry dropout, drain hit, malformed message,
  decode failure, or missing quad. Offline replay of the exact attempt shows
  zero N195 Gate-2 dropout activations and zero N194 Gate-3 frozen-recovery
  activations; this successful path did not require either watchdog.
- The command-free post-stop proof sent no reset or control command. After exit
  disarm, v3385 had cleared the race epoch to start `-1`/index `0` and
  reported passive low-threat ground contact `1001`, impact `0.02107`; it
  is not substituted for the smoke report's zero in-run collision result.
  Smoke/attempt/summary/post-stop/Gate-2-diagnostic/Gate-3-diagnostic hashes are
  `b44b0b441e3382c4d2722a23f7ae00fae1a05d95e5816a8962595d9bdf212109`,
  `5d4514ace948067347852759dc7a4d92095c17131aadfc1894741a65d946f194`,
  `ce8408bba4da14fe76b04674de3a761da9d61efd77a552261093750550d33a67`,
  `4e8ff2468d7c96ab538497667e9e10e390fc260d88e8691a816d1da32d12935d`,
  `167b0a31efefde52826c4657627e02138f5962213f57e16333e875e1e470cb19`,
  and `d7d6abb6f99f883f0b79955c3c5eaf868073d6404dc498733d1915889239c4d5`.
- Candidate 055 is accepted only as the bounded Gates-1--3 milestone, not as a
  six-gate lap. Authorize exactly one Candidate 056 `BoundedGate4` under tag
  `n195_gate2_frozen_observation_dropout_n194_gate4_bounded_056`: no code,
  policy, checkpoint, or hash change from passed Shadow 056/Candidate 055;
  actual Windows Python, reused PID `24476`, one detected normal-disarm-then-
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-
  valid `1/1`, and target/minimum/authoritative stop official index `4`.
  Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`, exact
  hashes, zero rate/collision/invalid/dropout/drain fault, one exit disarm, and
  command-free post-stop proof. Reject failure without unchanged retry.
  FullLap remains unauthorized.


## N195 Candidate 056 — Gate-3 Collision Before Gate 4

- Candidate 056 ran exactly once on reused responsive PID `24476`; no process
  launch. The disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset
  was detected in `0.484 s`: boot rolled `417871 -> 3390 ms`, fresh race
  start was `3307 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- Gates 1 and 2 passed officially. A third camera-range pass event at
  `10.109 s` was diagnostic only: official index remained `2`. Gate 3 then
  collided with frame `1001`, threat `2`, impact `9.248054504394531 m/s`;
  finish stayed `-1`, so Gate 4 was never active or evaluated. Candidate 055's
  clean official index-3 result remains the retained milestone.
- Transport was healthy: `601/10.250 = 58.537 Hz`, zero rate violations,
  `3212` telemetry messages, `128` frames/detections, and zero telemetry
  dropout, drain hit, malformed message, decode failure, or missing quad. N195
  Gate-2 dropout and N194 Gate-3 frozen recovery each had zero activations.
- The discriminating current branch is N192 terminal leveling. It was inactive
  throughout clean Candidate 055 but active for Candidate 056's last five
  Gate-3 samples. It latched at `9.781 s`, `5.367 m` forward, with plane-
  projected right/down `-0.342/-0.703 m`, then held normalized roll at
  `0.0` through collision while the lower promoted Gate-3 mode called for a
  negative counter-bank. Candidate 051's original three protected terminal-
  level samples begin much later at `3.664 m`; this separates an offline
  hypothesis that delays only the latch admission range without retiring it.
- The command-free post-stop proof sent no reset or control command. After exit
  disarm, v3385 cleared race start to `-1`/index `0` and reported passive
  low-threat ground contact `1001`, impact `0.02022`; this does not replace
  the in-run collision record. Smoke/attempt/summary/post-stop/control/
  transition/terminal-level/Gate-2/Gate-3-frozen hashes are
  `db797be5729a248b26841be1de81fd7ed34143a85ab369eb716cde5bbea0d9fc`,
  `970fbc2ee7ef90e6158a46e8c28c466c536379ab5c5280eb0eeff47f53d7505a`,
  `f9456a5425b19f9ed9ccd39929f7b8af0a03a714ccc88884d435d38fe6638ab0`,
  `bf1390342eb69e8a402d01797ec65969a47230d3a294ffcdfdf5dac6b92e0226`,
  `a53241039a24a18823d4a9507d02f8027fb9a81a2f9a767bde34051651a66174`,
  `adc0fe651c1d91824ba4afd4f87e263d8165cb8e650ed49b3237fa94ebe4511e`,
  `1b7250a551886dd50d35143a503e9b06016bdbf9181b0ac8ff29d7b20188fa28`,
  `42d7fe27c39b76ab7ec5eb2cc5208c39c6e20370490083fba376f910ea83fba0`,
  and `b553a445222dcdfb593278fb34736b0f237e2434b51de9b311191f66d10993e9`.
- Candidate 056 is rejected without unchanged retry. Candidate 057 and FullLap
  remain unauthorized pending an offline-separated later terminal-level
  admission, pinned hashes, focused tests, exact reset proof, and passive zero-
  command shadow.


## N196 Later Gate-3 Terminal-Level Admission — Offline Freeze

- N196 keeps the exact source-pinned N195 controller composition and changes
  only the Gate-3 terminal-safe-level admission range from `6.0 m` to
  `3.7 m`. The projection plane remains `2.0 m`; maximum absolute projected
  right/down errors remain `0.5/1.0 m`; minimum closing speed remains
  `1.0 m/s`; the branch still latches only on official Gate 3, changes only
  normalized roll to `0.0`, and releases on official transition. Gate-2 stale
  dropout, rapid vertical correction, Gate-3 frozen recovery, recurrent state,
  pitch, thrust, and yaw are unchanged.
- Source-hashed replay passes on protected clean Gate-3 paths 016/018/020/021/
  022 plus Candidates 045/048, with zero action changes. Candidate 051 retains
  its exact three proven level corrections beginning at `3.664 m`; clean
  Candidate 055 stays inactive. Candidate 056's old five-sample latch is reduced
  to two samples beginning at `2.447 m`, retiring exactly the premature
  `5.367/4.873/3.875 m` admissions. This is offline branch separation only,
  not counterfactual proof of a Gate-3 pass.
- Linux focused N189--N196 controller/verifier/runner tests pass `37/37`;
  actual Windows focused controller/runner tests pass `30/30`; Python
  compilation and the PowerShell deployment parser pass. Policy/verifier/
  offline-report/runner SHA-256 values are
  `3e2fb526ae0115b36c4dfbdfe1c5a5bb5a6265caabd3a45a3c9ea8ed1077c2df`,
  `f0978810ba51fcfdbd1a2f031049a73398648e2e5179027b8053499ee7e697e7`,
  `c53ddc068a47a1cc1f3b296ad4fd40c94916ab95c796a184cddc2713939bee5d`,
  and `82668d40b44f1991e199251f2b3109ebe37a6b38e076281ac4cc31f9b4043f79`.
- No simulator command occurred during N196 development. Candidate 057 and
  FullLap remain unauthorized. Next only: preserve responsive shipping PID
  `24476`; recover the sole VQ1/R1 UI on that same process with
  `start_windows_aigp_race.ps1 -NoRelaunch -SkipLoginClick` if the inactive
  base/status `65/3` state is present; then send normal disarm followed by
  exactly one learned-target MAVLink `COMMAND_LONG` command `31000`,
  confirmation `0`, parameters 1--7 all zero. Count reset only after boot
  rollback greater than `1000 ms`, a fresh nonnegative race-start epoch,
  official index `0`, finish `-1`, and visible Gate 1. Then run passive
  zero-command `n196_gate3_later_terminal_level_n195_shadow_057` with
  `-Gate3LaterTerminalLevelN195`; only a passing hash/parity/cadence/stream/
  fault report may authorize one separately preregistered bounded Candidate
  057.


## N196 Reset-31000 Proof and Passive Shadow 057 — Passed

- Responsive shipping PID `24476` was preserved throughout and never
  relaunched. A command-free precheck found Candidate 056's inactive
  base/status `65/3` state with race start `-1`. The documented
  `-NoRelaunch -SkipLoginClick` recovery restored the sole VQ1/R1 session on
  the same PID to base/status `193/4`, official index `0`, finish `-1`,
  visible Gate 1, and old boot/race epoch `1118326/1096556 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000`, confirmation `0`,
  parameters 1--7 zero. Boot rolled `1118326 -> 3276 ms`; a fresh race start
  appeared at `3282 ms`. A later command-free proof showed boot `31091 ms`,
  base/status `193/4`, index `0`, finish `-1`, `136` Gate-1
  detections, and no collision.
- Passive `n196_gate3_later_terminal_level_n195_shadow_057` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `589/10.016 = 58.706 Hz`, with zero rate violations. All `251/251`
  visible samples replayed at `25.022 Hz` with maximum action error
  `2.3526687623065534e-7`; every deployment hash matched. It observed
  `2426` telemetry messages, `1248` IMU samples, and `115/115` camera
  detections, with zero collision, telemetry dropout, drain hit, malformed
  message, decode failure, or missing quad.
- Inactive-precheck/UI-ready/UI-ready-snapshot/reset/reset-ready/shadow/parity
  SHA-256 values are
  `0295072468ac6270eaca45fe75341dd28475630018a333bf6f7d786148a65477`,
  `fca195732348c4b796579a73e4520ff1e9c6fddbd9b32303b4664ab014527796`,
  `a93f6259adede0a7b2ee9f4414c5dc9fab6cb705d8a597fd1c743fd0affb0940`,
  `5ad9b57579de8d94bf7a49cdff33c3638387b1bbbdcac475e73c29db5e439cfe`,
  `46cb97682a12cdaaac0948c3ed20b1943fed7427f26381d4984f1ae27a8cc2e4`,
  `58b49e1f3c94231eda5bc6d67c328ad64fc02691609ba8b80e6a013e92cdbb28`,
  and `28a174a92a47507cf7ae65dee865e25f7bf9c5b9ab33abe3ec622b8d1686a7d2`.
- Authorize exactly one Candidate 057 `BoundedGate3` under tag
  `n196_gate3_later_terminal_level_n195_bounded_057`: frozen N196 hashes,
  actual Windows Python, reused PID `24476`, one detected normal-disarm-then-
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-
  valid `1/1`, and target/minimum/authoritative stop official index `3`.
  Accept only clean official index `3`, finish `-1`, cadence `[50,100) Hz`,
  exact hashes, zero rate/collision/invalid/dropout/drain fault, one exit
  disarm, and command-free post-stop proof. Reject failure without unchanged
  retry. Candidate 058 and FullLap remain unauthorized.


## N196 Candidate 057 — Collision-Free Gate-3 Vision-Loss Timeout

- Candidate 057 ran exactly once on reused responsive PID `24476`; no process
  launch. The disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset
  was detected in `0.516 s`: boot rolled `179671 -> 3410 ms`, fresh race
  start was `3284 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- Gates 1 and 2 passed officially; authoritative last-gate race time was
  `7.482846260 s`. Gate 3 did not pass. The attempt reached its exact
  `45 s` bound at official index `2`, finish `-1`, with no collision or
  invalid state. Delivery was healthy: `2714/44.844 = 60.499 Hz`, zero rate
  violations, `11094` telemetry messages, `5602` IMU samples, `569`
  frames, `169` detections, zero telemetry dropout/drain/malformed/decode
  fault, and one exit disarm.
- The last detection aged to `30.938 s` by run end. The legal raw-confidence
  observation slot 30 became exactly zero at `14.313 s` and remained zero
  for `281` consecutive eligible Gate-3 samples. Its fifth sample was
  `14.750 s` at forward/closing/right/down
  `2.822/7.052/+0.995/-0.859 m` and yaw error `+0.339 rad`; deployed
  action remained approximately pitch `+0.093`, roll `-1.0`, thrust
  `+0.761`, yaw `-0.00052`.
- N196 later terminal-level admission had zero activations, so Candidate 057
  does not falsify the N196 threshold. N194's frozen-observation recovery also
  had zero activations: sub-micro estimator drift repeatedly exceeded its
  `1e-7` equality test even though detector confidence stayed exactly zero.
  This separates raw confidence persistence as the simpler observable fault
  signal.
- The command-free post-stop proof sent no reset or control command and showed
  base/status `65/3`, official index `2`, finish `-1`, and no collision.
  Reuse-check/probe/smoke/attempt/summary/CSV/post-stop SHA-256 values are
  `96d3c52ebdb5e4d1d7b4a4ce516fa56529ff0feca417821c163a8a62f6bc543b`,
  `a3d8e9ffb763a8893ea14ec10c22a957d58d862557ded128281e3f4236ecbf5d`,
  `f12418f224a9f02140b7cc6e502e4905d42fde9315c9d9543a1ed5aed8905e3a`,
  `34058c04566520e7d2e962008f66ae43c29b78371d0b2c94dd3f2f63ebaa0e16`,
  `aaa255b024ca128b3d36be2ab61357d15563ae4dd348c58b371249113fe21e29`,
  `915a22b622d9f1bd1ecacc952102f850ff472d4648689958b4c2f999f17ee6b8`,
  and `c8db2071ff882096255c3803f81c3ee3697b2f0842d8b6ee44a05c1ef2d7567f`.
- Candidate 057 is rejected without unchanged retry. Candidate 058 and FullLap
  remain unauthorized pending a source-hashed, offline-separated raw-
  confidence recovery, focused tests, exact reset proof, and passive shadow.

## N197 Gate-3 Persistent Zero-Confidence Recovery — Offline Freeze

- N197 evaluates exact source-pinned N196 once, then adds one Gate-3-only
  controller. While official Gate 3 is visible, within `3.0 m`, closing at
  least `1.0 m/s`, and legal raw-confidence field 30 is exactly zero for five
  consecutive inference samples, it sets only normalized roll to `0.0` and
  commands the existing bearing-signed `0.25 rad` yaw scan. Pitch and thrust
  remain exact. Any positive confidence, ineligible pose, visibility loss, or
  official transition releases the recovery immediately.
- Source-hashed replay covers every Candidate 016--057 trace. Only the three
  long Gate-3 vision-loss stalls activate: Candidate 034 once for `259`
  samples beginning `16.437 s`, Candidate 053 once for `272` samples
  beginning `15.047 s`, and Candidate 057 once for `277` samples beginning
  `14.750 s`. Every other trace remains inactive, including protected clean
  Gate-3 paths 016/018/020/021/022/055 and Candidates 045/048/051/056.
  Every active sample changes only roll/yaw, preserves pitch/thrust exactly,
  and remains finite/in range. Candidate 057's command-free official post-stop
  proof is embedded.
- Linux focused N189--N197 controller/verifier/runner tests pass `42/42`;
  actual Windows focused controller/runner tests pass `34/34`; compilation
  and the PowerShell deployment parser pass. Policy/verifier/offline-report/
  runner SHA-256 values are
  `ca2ceb12ecbc62a721bf3f34b7cd6b2675592626604899153c3ab7b92754b210`,
  `d2fc0816f3fe7e615d7d56e263e3368e8fbbf7ec0b221760928adcd7845c0ac9`,
  `a54361e1d2108cf9171086abde3e75d3f68ac8d5213dd2f73cf1a6c7794974d8`,
  and `7365cd84565d52f225834cc5252c79bf5ae6a9ce04a0325dd6e25ef8d409f526`.
- No simulator command occurred during N197 development. Candidate 058 and
  FullLap remain unauthorized. Next only: preserve PID `24476`; recover the
  sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` because Candidate 057
  stopped at base/status `65/3`; then send normal disarm followed by exactly
  one learned-target MAVLink `COMMAND_LONG 31000`, confirmation `0`,
  parameters 1--7 zero. Require boot rollback greater than `1000 ms`, fresh
  nonnegative race start, index `0`, finish `-1`, and visible Gate 1. Then
  run passive zero-command
  `n197_gate3_zero_confidence_recovery_n196_shadow_058` with
  `-Gate3ZeroConfidenceRecoveryN196`. Only a fully passing hash/parity/
  cadence/stream/fault report may authorize a separately preregistered bounded
  Candidate 058.


## N197 Reset-31000 Proof and Passive Shadow 058 — Passed

- Responsive shipping PID `24476` was preserved and never relaunched.
  Candidate 057's base/status `65/3` post-stop was recovered through the sole
  VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` on the same PID. Command-free
  readiness showed base/status `193/4`, index `0`, finish `-1`, visible
  Gate 1, and old boot/race epoch `537019/514210 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000`, confirmation `0`,
  parameters 1--7 zero. Boot rolled `537019 -> 3077 ms`; fresh race start
  was `3302 ms`. A later command-free proof showed boot `10092 ms`,
  base/status `193/4`, index `0`, finish `-1`, `126` Gate-1
  detections, and no collision.
- Passive `n197_gate3_zero_confidence_recovery_n196_shadow_058` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `597/9.968 = 59.791 Hz`, zero violations. All `308/308` samples
  replayed at `30.8 Hz` with maximum action error
  `2.2283401487910304e-7`; `307` were visible and every deployment hash
  matched. It observed `2434` telemetry messages, `1254` IMU samples, and
  `124` frames/`121` detections, with zero collision, dropout, drain hit,
  malformed message, or decode failure.
- UI-ready/UI-ready-snapshot/reset/reset-ready/shadow/parity SHA-256 values are
  `1b5925a554846b13a41effbbeb381f3f24c9872fe4f7db48bb83108a7dba68e7`,
  `d70708e4b7636dd0d61e752bd65fa4cabeaf4056494fb615bfa094812ac6ad71`,
  `907af66f4fb19c556a0be5ac109203794e56acb2b126404c724ef66ea089f57e`,
  `75a663bfd9465378090231d854ebc1f9d97a918ece9e069c4d792b604410f4e9`,
  `704f52f37238e3888ff584e823d0c2247e556400a4d1ea154af25d012c8f6cfd`,
  and `080938e3a37bbe6bb8655efcb2dd47c5562d21585015cb39a82b1cee8eb997fb`.
- Authorize exactly one Candidate 058 `BoundedGate3` under tag
  `n197_gate3_zero_confidence_recovery_n196_bounded_058`: frozen N197 hashes,
  actual Windows Python, reused PID `24476`, one detected normal-disarm-then-
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-
  valid `1/1`, and target/minimum/authoritative stop official index `3`.
  Accept only clean index `3`, finish `-1`, cadence `[50,100) Hz`, exact
  hashes, zero rate/collision/invalid/dropout/drain fault, one exit disarm, and
  command-free post-stop proof. Reject failure without unchanged retry.
  Candidate 059 and FullLap remain unauthorized.


## N197 Candidate 058 — Clean Official Gates 1--3 Milestone

- Candidate 058 ran exactly once on reused responsive PID `24476`; no process
  launch. Its disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset
  was detected in `0.5 s`: boot rolled `127600 -> 3412 ms`, fresh race
  start was `3299 ms`, index `0`, finish `-1`, and calibration completed
  `60/60` samples.
- The vehicle passed official Gates 1, 2, and 3 cleanly. Authoritative index
  reached `3` with official last-gate race time `10.867674827 s`; the
  configured index-3 stop fired at controller elapsed `10.812 s`, and finish
  remained `-1`. There was no collision or invalid state.
- Transport and perception were healthy: `648/10.718 = 60.366 Hz`, zero
  command-rate violations, `3341` telemetry messages, `1688` IMU samples,
  `137` frames/`131` detections, and zero telemetry dropout, drain hit,
  malformed message, or decode failure. Offline replay shows zero N197
  zero-confidence activations, so this clean run used the unchanged N196 path
  and supplies no live treatment evidence.
- The command-free post-stop proof sent no reset/control command. After exit
  disarm, v3385 cleared the race epoch to start `-1`/index `0`, retained
  finish `-1`, and reported no collision. Reuse-check/probe/smoke/attempt/
  summary/CSV/post-stop SHA-256 values are
  `d3d10646cc68698a18bd454a1280816bad4e688333ef8d0fea9871b651f03b6b`,
  `5adf98828160a043b08afc3dd9472fd9c968403f1893f152e7016845c2c8d932`,
  `94651873c6db9abbe4ea02fce7755115ae02f22f1f61585a2bc8e31c9965f6d9`,
  `cffcdaa916ebe2ff79fa517c8d9d60110b30b1e648e55d0a3ed8da659ec14ce3`,
  `192059e1849366260cca0c1220a33e0c81f852eaa8d528a543fbb57b38d15606`,
  `a8ed667531fc0465f540df3a6d514d0efa7035e22aa11dda57c23c5d1d1287b1`,
  and `059b103ea3450793be326e1367bbe4da9928b7e80a45b68502f628c4d55606c8`.
- Candidate 058 is accepted only as a bounded official Gates-1--3 milestone.
  Authorize exactly one Candidate 059 `BoundedGate4` under tag
  `n197_gate3_zero_confidence_recovery_n196_gate4_bounded_059`: no policy,
  checkpoint, code, or deployment-hash change from passed Shadow 058/Candidate
  058; actual Windows Python, reused PID `24476`, one detected normal-disarm-
  then-command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/
  min-valid `1/1`, and target/minimum/authoritative stop official index `4`.
  Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`, exact
  hashes, zero rate/collision/invalid/dropout/drain fault, one exit disarm, and
  command-free post-stop proof. Reject failure without unchanged retry.
  Candidate 060 and FullLap remain unauthorized.


## N197 Candidate 059 — Rejected at Official Gate 3

- Candidate 059 ran exactly once as the authorized `BoundedGate4` attempt on
  reused responsive shipping PID `24476`; no process launch. Its disarm-first
  learned-target MAVLink `COMMAND_LONG 31000` reset was detected in `0.5 s`:
  boot rolled `122560 -> 3423 ms`, fresh race start was `3304 ms`, index
  was `0`, finish was `-1`, and calibration completed `60/60` samples.
- The vehicle passed official Gates 1 and 2, with last official gate time
  `7.355759143 s`. A camera-side detection did not become an official Gate-3
  pass: authoritative index remained `2`, and collision `1001`/threat `2`
  occurred at controller elapsed `9.776438713 s`. Finish remained `-1` and
  Gate 4 was never active. Candidate 059 is rejected with no unchanged retry.
- Transport remained healthy: `640` commands over `10.375 s = 61.59 Hz`,
  zero rate violations, `3289` telemetry messages, `1661` IMU samples, and
  `133/133` frames/detections, with zero dropout, drain hit, malformed
  message, decode failure, or no-quaternion sample. Exit disarm count was one.
- N197 zero-confidence recovery never activated. N196 terminal roll leveling
  activated for exactly two samples: first at `10.250 s`, forward `3.221 m`,
  projected right/down `+0.0048/-0.444 m`; then at `10.375 s`, forward
  `2.232 m`, projected right/down `+0.0873/-0.236 m`. Both forced roll
  to zero. The lower controller's exact counterfactual output was not logged,
  so no stronger causal claim is made.
- Command-free post-stop sent no reset/control command and showed base/status
  `65/3`, race start `-1`, index `0`, finish `-1`. Its small passive ground
  contact after disarm does not replace the in-run impact. Reuse-check/probe/
  smoke/attempt/summary/CSV/post-stop SHA-256 values are
  `3b9fa82c0103c917cb42fb4662d167f054186552fc152f1a07f03bc7b039f3f8`,
  `3832c67eb3e59e9736e3e273036b21f49522865d895cc66306f1ac47173aa071`,
  `40f7ee4a867b6207e35d3eefedd6da6f7319ee26dfd07f2a96195dffb0486aa0`,
  `f21c652b5debfa808d4e28540b1d1e7b9fce7331f9f31caac988bb1d546b0d6f`,
  `71debead7f6ce41d00d3ca1869446d4268623e086fce31a4d74134cb6b1bf017`,
  `1a3bf9f78ff3d97e094c09c961db0b86cf9e24d45485238fd5fc9976de2a4548`,
  and `06aa6932eabb3ab3866826448a79bf2f69d768b56d6da2c5acba38d2fc3be161`.

## N198 Gate-3 Final-Only Terminal Level — Offline Freeze

- N198 is an exact source-pinned re-composition of N197 with one change:
  Gate-3 terminal roll leveling may first latch only at forward distance
  `<= 2.0 m`, instead of `<= 3.7 m`. Projection bounds, minimum closing
  speed, level-roll value, latch behavior, all other controllers, checkpoints,
  and the six-gate course contract remain unchanged.
- Source-hashed replay over every Candidate 016--059 flight trace passes with
  no blockers. The clean Candidate 021 behavior remains active for three
  samples beginning at `1.771 m`; clean Candidates 055 and 058 remain
  inactive. Failed Candidates 051, 056, and 059 change from `3/2/2` terminal
  samples to `0/0/0`. New activations are a strict subset of old activations,
  all protected clean recorded actions remain exact, only roll can change,
  and every output is finite/in range.
- Focused dependency/policy/runner tests pass `30/30` on Linux and `30/30`
  under the actual Windows Python. The PowerShell deployment parser passes.
  Policy/verifier/offline-report/runner SHA-256 values are
  `cfd9308910dbe63a857a67a3b96d53204a2ad929c9ad9b1a93787ba9392e0724`,
  `48ceb6b6e36a6c56501b799e27e9689e016cf274a2ab06c75bdcbad3a10f2f58`,
  `8c31a2e2463dd14afc8c1fd3bb8a422890c07943086abe7c7041e4cb1b9ef278`,
  and `3b446631b4cce0a5eb9ed2c8264fcb33a5f10a56c844484ac8f0466d18e7e770`.
- No simulator command occurred during N198 development. Next only: preserve
  PID `24476`; recover the sole VQ1/R1 UI on that PID with
  `scripts/start_windows_aigp_race.ps1 -NoRelaunch -SkipLoginClick`; send
  normal disarm, wait `100 ms`, then exactly one learned-target MAVLink
  `COMMAND_LONG 31000`, confirmation `0`, parameters 1--7 all zero. Require
  boot rollback greater than `1000 ms`, a fresh nonnegative race start,
  official index `0`, finish `-1`, and visible Gate 1. Then run passive
  zero-command `n198_gate3_final_terminal_level_n197_shadow_059` with
  `-Gate3FinalTerminalLevelN197`. Only a fully passing hash/parity/cadence/
  stream/fault report may authorize one separately preregistered Candidate
  060 `BoundedGate3`. Candidate 060 and FullLap remain unauthorized.


## N198 Reset-31000 Proof and Passive Shadow 059 — Passed

- Responsive shipping PID `24476` was preserved and never relaunched.
  Candidate 059's inactive base/status `65/3` state was recovered through the
  sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` on the same PID.
  Command-free readiness showed base/status `193/4`, index `0`, finish
  `-1`, visible Gate 1, and old boot/race epoch `822084/798447 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000`, confirmation `0`,
  parameters 1--7 zero. Boot rolled `822084 -> 3192 ms`; fresh race start
  was `3297 ms`. A later command-free proof showed boot `16217 ms`,
  base/status `193/4`, index `0`, finish `-1`, `234` Gate-1 detections,
  and no collision.
- Passive `n198_gate3_final_terminal_level_n197_shadow_059` passed while
  sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op cadence
  was `599/10.016 = 59.704 Hz`, zero violations. All `253/253` visible
  samples replayed at `25.219 Hz` with maximum action error
  `1.9466705319937105e-7`; every deployment hash matched. It observed
  `2425` telemetry messages, `1249` IMU samples, and `122/122` frames/
  detections, with zero collision, dropout, drain hit, malformed message,
  no-quad result, or decode failure.
- UI-ready/UI-ready-snapshot/reset/reset-ready/shadow/parity SHA-256 values are
  `511e7864d82d0f73ac1348a8f8555ad8b3d23b56f899d07e0576be84ee4858f2`,
  `e22f409e6429532f8db54570cc0bcd65414fe11f08d9093d143c7967a264b302`,
  `190cfd6e11cac446d8e1d013f738f8cbb3d3fbd058af5945657f4333ce1c71d1`,
  `2caf105ac2843e18c814cc0b8c02385e9059d8e08e0ba5844dc1432d1d0367ca`,
  `b522ec24b6c8f0bfe7af8e1e0bb73f76b1f1136da692a63ad60ca249e74ec1de`,
  and `339eda10bb544565184a815da96b16d1f90cda3243d815ca68128d5ff52405d9`.
- Authorize exactly one Candidate 060 `BoundedGate3` under tag
  `n198_gate3_final_terminal_level_n197_bounded_060`: frozen N198 hashes,
  actual Windows Python, reused PID `24476`, one detected normal-disarm-then-
  command-`31000` reset, requested `80 Hz`, maximum `45 s`, repeats/min-
  valid `1/1`, and target/minimum/authoritative stop official index `3`.
  Accept only clean index `3`, finish `-1`, cadence `[50,100) Hz`, exact
  hashes, zero rate/collision/invalid/dropout/drain fault, one exit disarm, and
  command-free post-stop proof. Reject failure without unchanged retry.
  Candidate 061 and FullLap remain unauthorized.


## N198 Candidate 060 — Clean Official Gates 1--3 Milestone

- Candidate 060 ran exactly once on reused responsive shipping PID `24476`;
  no process launch. Its disarm-first learned-target MAVLink `COMMAND_LONG
  31000` reset was detected in `0.516 s`: boot rolled `191126 -> 3399 ms`,
  fresh race start was `3296 ms`, index `0`, finish `-1`, and calibration
  completed `60/60` samples.
- The vehicle passed official Gates 1, 2, and 3 cleanly. Authoritative index
  reached `3` with official last-gate race time `10.836339950 s`; the
  configured index-3 stop fired at controller elapsed `10.782 s`, and finish
  remained `-1`. There was no collision or invalid state.
- Transport and perception were healthy: `641/10.672 = 59.970 Hz`, zero
  command-rate violations, `3351` telemetry messages, `1693` IMU samples,
  and `136/136` frames/detections, with zero telemetry dropout, drain hit,
  malformed message, no-quad result, or decode failure. Trace replay shows
  N198 terminal leveling active on the final recorded Gate-3 sample at
  `1.849711 m`; N197 zero-confidence recovery remained inactive.
- The command-free post-stop proof sent no reset/control command and retained
  base/status `65/3`, official index `3`, last-gate time `10.836339950 s`,
  finish `-1`, and no collision. Reuse-check/probe/smoke/attempt/summary/CSV/
  post-stop SHA-256 values are
  `2db194c60686ef6c8e464ce69d5d46092584f4ae1da995c7076fe5ede9e583cc`,
  `db78ed65ec16bb6794c6639f63b3a7b30cee5788f8693755e5d49035bd87161b`,
  `55e9ef64a6147d8b1613c5a2ba5e4d9970916e87eb791631dc18e012718cde4f`,
  `f680b61973621dc43fd95d2cd972885e9de01f801d5c078a3d764571c9a63543`,
  `32cfbcbdff5ef286f09de3e78d8dd30a1c2d78f7754a9950755584dc6e94e38c`,
  `fa890d6152edf5d22a048657d45275035656ba6b2b4db500b9ba93b5e6afa40b`,
  and `a038b3ff236e2527b4c0f63ce5178c7fb2ae7cac78c235ffaeff8357ea1f940d`.
- Candidate 060 is accepted only as a bounded official Gates-1--3 milestone.
  Authorize exactly one Candidate 061 `BoundedGate4` under tag
  `n198_gate3_final_terminal_level_n197_gate4_bounded_061`: no policy,
  checkpoint, code, or deployment-hash change from passed Shadow 059/
  Candidate 060; actual Windows Python, reused PID `24476`, one detected
  normal-disarm-then-command-`31000` reset, requested `80 Hz`, maximum `45 s`,
  repeats/min-valid `1/1`, and target/minimum/authoritative stop official
  index `4`. Accept only clean index `4`, finish `-1`, cadence `[50,100) Hz`,
  exact hashes, zero rate/collision/invalid/dropout/drain fault, one exit
  disarm, and command-free post-stop proof. Reject failure without unchanged
  retry. Candidate 062 and FullLap remain unauthorized.


## N198 Candidate 061 — Rejected at Official Gate 3

- Candidate 061 ran exactly once as the authorized `BoundedGate4` attempt on
  reused responsive shipping PID `24476`; no process launch. The runner
  recovered the inactive VQ1/R1 UI on the same PID without relaunch. Its
  disarm-first learned-target MAVLink `COMMAND_LONG 31000` reset was detected
  in `0.500 s`: boot rolled `145737 -> 3428 ms`, fresh race start was
  `3286 ms`, index `0`, finish `-1`, and calibration completed `60/60`.
- The vehicle passed official Gates 1 and 2, with last official gate time
  `7.537002563 s`. The camera-side pass counter recorded a third crossing at
  controller elapsed `10.453 s`, but official index remained `2`. No collision
  occurred; the run timed out at the preregistered `45 s` cap with finish
  `-1`, so Candidate 061 is rejected without unchanged retry.
- The failure is a stale-association/weak-detection stall, not a transport
  fault. After the camera-side crossing, the motion track stayed near
  `2.44--2.50 m` while new detections jumped tens of metres away at raw
  confidence around `0.02--0.03`; detections ended near `14.53 s` and final
  detection age was `30.453 s`. The exact-zero N197 watchdog therefore
  repeatedly released on weak nonzero false detections; trace replay shows
  `13` exact-zero activation episodes beginning only at `15.266 s`.
- Transport remained healthy: `2742/44.890 = 61.060 Hz`, zero rate
  violations, `11072` telemetry messages, `5593` IMU samples, and `564`
  frames, with zero collision, dropout, drain hit, malformed message, or
  decode failure. Exit disarm count was one.
- Command-free post-stop sent no reset/control command and showed base/status
  `65/3`, cleared race start `-1`/index `0`, finish `-1`, and no collision.
  Reuse-check/probe/smoke/attempt/summary/CSV/post-stop SHA-256 values are
  `e1529747303c1ad67b18eb48df078892476ace1a29cbbc26f1af84e3b1f586b8`,
  `c6424ff6d604609db9c22625e1d01048435f7ffaad703e9b4a9169b089bdec2d`,
  `1042407989c2321d57300740ee18e636f38f8ed671b21dea64586695106bf845`,
  `6f0e91bb854de5c81b887a977d9ac408d03101072aef846d8d9ec103c01504b4`,
  `51ae6294cc18d3e9423353e678d441e6e4cc118eae14500d480df6f021cf1fd7`,
  `6e5eb52c7c9600f80f91df80cbb134bedf46435967682cadc8ba6bf90d44075a`,
  and `c241708cbee039e4dafd7aeebeb67a5da2593c355a3818092c824383199c1f47`.
- Candidate 062 and FullLap remain unauthorized. Next only: implement N199
  as exact N198 plus one Gate-3 low-confidence recovery latch, prove its
  separation offline, then require a new exact reset proof and passive shadow.


## N199 Gate-3 Low-Confidence Latched Recovery — Offline Freeze

- N199 evaluates exact source-pinned N198 once, then adds one Gate-3-only
  recovery. While official Gate 3 is visible, within `3.0 m`, closing at
  least `1.0 m/s`, and legal raw-confidence field 30 is `<= 0.025` for five
  consecutive inference samples, it latches until the official gate index
  changes. While latched it sets only normalized roll to `0.0` and commands
  the existing bearing-signed `0.25 rad` yaw scan. Pitch and thrust remain
  exact. Weak nonzero false detections no longer release the recovery.
- Source-hashed replay covers every Candidate 016--061 trace. Only four failed
  stale/weak-confidence Gate-3 paths activate, exactly once each: Candidate
  034 for `305` archived samples beginning `11.375 s`, Candidate 053 for
  `308` beginning `11.031 s`, Candidate 057 for `309` beginning `11.047 s`,
  and Candidate 061 for `299` beginning `11.344 s`. Every other trace remains
  inactive, including protected clean paths 016/018/020/021/022/055/058/060.
  Candidate 061 recovery now begins after its `10.453 s` camera-side crossing
  but before the old fragmented exact-zero recovery at `15.266 s`.
- All active samples change only roll/yaw, preserve pitch/thrust exactly, and
  remain finite/in range. Candidate 061's command-free post-stop proof is
  embedded. Focused dependency/policy/runner tests pass `35/35` on Linux and
  `35/35` under the actual Windows Python; the PowerShell deployment parser
  passes. Policy/verifier/offline-report/runner SHA-256 values are
  `9deaecd3957d02b11dabdc7b1fb36ce649f1e5575fd11fc1e190332508742d30`,
  `3bb48ce664d8591747479a95a77ef08245ac5643ffccb49b5173ff1b475ac5fa`,
  `420074d73a2879b587c63895007906c6daf1065423f632890c36c1c992397495`,
  and `b9b3486802a45fafc9ed59767e53d9acdf2bbbf62919444bbe51b25aa1a5010e`.
- No simulator command occurred during N199 development. Candidate 062 and
  FullLap remain unauthorized. Next only: preserve PID `24476`; recover the
  sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick`; send normal disarm, wait
  `100 ms`, then exactly one learned-target MAVLink `COMMAND_LONG 31000`,
  confirmation `0`, parameters 1--7 zero. Require boot rollback greater than
  `1000 ms`, fresh nonnegative race start, official index `0`, finish `-1`,
  and visible Gate 1. Then run passive zero-command
  `n199_gate3_low_confidence_latched_recovery_n198_shadow_060` with
  `-Gate3LowConfidenceLatchedRecoveryN198`. Only a fully passing hash/parity/
  cadence/stream/fault report may authorize one separately preregistered
  Candidate 062 `BoundedGate3`.


## N199 Reset-31000 Proof and Passive Shadow 060 — Passed

- Responsive shipping PID `24476` was preserved and never relaunched.
  Candidate 061's inactive state was recovered through the sole VQ1/R1 UI
  with `-NoRelaunch -SkipLoginClick` on the same PID. Command-free readiness
  showed base/status `193/4`, index `0`, finish `-1`, visible Gate 1, and
  old boot/race epoch `568832/560784 ms`.
- The tracked reset learned target IDs, sent normal disarm, waited `100 ms`,
  and sent exactly one MAVLink `COMMAND_LONG 31000`, confirmation `0`,
  parameters 1--7 zero. Boot rolled `568832 -> 3127 ms`; fresh race start
  was `3321 ms`. A later command-free proof showed boot `25169 ms`,
  base/status `193/4`, index `0`, finish `-1`, `236` Gate-1 detections,
  and no collision.
- Passive `n199_gate3_low_confidence_latched_recovery_n198_shadow_060` passed
  while sending zero reset, arm, MAVLink setpoint, and disarm commands. No-op
  cadence was `613/10.000 = 61.200 Hz`, zero violations. All `327/327` visible
  samples replayed at `32.700 Hz` with maximum action error
  `2.3379898070330363e-7`; every deployment hash matched. It observed `2448`
  telemetry messages, `1260` IMU samples, and `125/125` frames/detections,
  with zero collision, dropout, drain hit, malformed message, no-quad result,
  or decode failure.
- UI-ready/UI-ready-snapshot/reset/reset-ready/shadow/parity SHA-256 values are
  `c20fc71f536469c5f3383353005abfadc0ef6c4c9a92bb2ebdab5da815a0d2c1`,
  `c6845f9f28a4b00a94798489af21665587bc832378f93dc8ffd19ac7ade5e002`,
  `983ff37316b293f4a03b4fb6693894f9eb6bcdd2506a25a2b631cb854663c2cc`,
  `12646f641bfdbb01e8c9965213002d17beac492e7708833af076f5e9ddcec834`,
  `27cbdb3a9bdc9745a18c92835230b76933d11b421b2bc5b01a5e85b3fa3b1b44`,
  and `c3b21347334e02987d061fffb9b2aab3f70884c8ef24a5840b030ec81da265c9`.
- Authorize exactly one Candidate 062 `BoundedGate3` under tag
  `n199_gate3_low_confidence_latched_recovery_n198_bounded_062`: frozen N199
  hashes, actual Windows Python, reused PID `24476`, one detected normal-
  disarm-then-command-`31000` reset, requested `80 Hz`, maximum `45 s`,
  repeats/min-valid `1/1`, and target/minimum/authoritative stop official
  index `3`. Accept only clean index `3`, finish `-1`, cadence `[50,100) Hz`,
  exact hashes, zero rate/collision/invalid/dropout/drain fault, one exit
  disarm, and command-free post-stop proof. Reject failure without unchanged
  retry. Candidate 063 and FullLap remain unauthorized.


## N199 Candidate 062 — Rejected at Official Gate 3

- Candidate 062 ran exactly once as the authorized `BoundedGate3` attempt on
  reused responsive shipping PID `24476`; no process launch. Its disarm-first
  learned-target MAVLink `COMMAND_LONG 31000` reset was detected in `0.516 s`:
  boot rolled `128409 -> 3408 ms`, fresh race start was `3297 ms`, index `0`,
  finish `-1`, and calibration completed `60/60` samples.
- The vehicle passed official Gates 1 and 2, with last official gate time
  `7.426924228 s`. The camera-side counter recorded its third crossing at
  controller elapsed `10.359 s`, but official index remained `2`. Collision
  `1001`/threat `2`, impact `9.807114601`, ended the run near `10.562 s`;
  finish remained `-1`. Candidate 062 is rejected without unchanged retry.
- N199 did not activate before the collision: weak-confidence persistence was
  absent and the last raw confidence was `0.042188`. N198's `2.0 m` terminal
  admission also remained inactive; counterfactual replay of the prior
  `3.7 m` safe-projection controller activates for two recorded samples at
  `10.422/10.531 s`, forward `2.452/2.420 m`, replacing recorded roll
  `-0.467/-0.867` with level roll. Candidate 061 would similarly activate
  before its stale-track divergence at `10.297 s`.
- Transport was healthy: `625/10.469 = 59.605 Hz`, zero rate violations,
  `3305` telemetry messages, `1669` IMU samples, and `134` frames, with zero
  dropout, drain hit, malformed message, or decode failure. Exit disarm count
  was one.
- Command-free post-stop sent no reset/control command and retained
  base/status `65/3`, official index `2`, last-gate time `7.426924228 s`,
  finish `-1`, and no continuing collision event. Reuse-check/probe/smoke/
  attempt/summary/CSV/post-stop SHA-256 values are
  `7aa62a0651ca899f37dda365797b5f0b29faf72fa813343c8359fae871028223`,
  `9e281b9bd784596360c29d227c05b83e41f64a4d3823e6ee73074a9db3c37868`,
  `4695658dcbfc2cf78ae551f82bcad8aa0ffd920349f97352ad0b6bf6278821e4`,
  `0fc67b546d16500832575809143cb5a5cf7d803fa58c12cc560c70bb3ef6edbc`,
  `3c98c6ad008f3d7c9d88e6d799366c2b7b89c04e802469d1c45d728b383f5ad1`,
  `8a673f2a71192a16c3ba5a57a360d5a4eba6fd1c72b903f1eccd0ee87275e835`,
  and `e52289244072dd2c0e9514c24436fa7cc9fb4043926f7bb3615ace23cc7883f4`.
- Candidate 063 and FullLap remain unauthorized. Next only: implement N200
  as exact N199 plus restoration of the source-pinned `3.7 m` safe-projection
  terminal roll leveler, document the corrected Candidate-059 causality, prove
  separation offline, then require a fresh exact reset and passive shadow.


## N200 Restored 3.7 m Gate-3 Terminal Level — Offline Freeze

- N200 evaluates exact source-pinned N199 once, then reapplies the exact
  source-pinned N196 Gate-3 safe-projection terminal leveler. The only
  behavioral difference from N199 is restoration of first admission from
  `2.0 m` to `3.7 m`; projection bounds, minimum closing speed, level-roll
  value, latch behavior, N199 recovery, all other controllers, checkpoints,
  and course contract remain unchanged.
- This restoration corrects the prior Candidate-059 causal interpretation.
  Candidate 059 collided at `9.776438713 s`; the archived controller's first
  terminal activation was later at `10.250 s`, so it could not have caused
  that collision. Candidate 062 supplies the opposite evidence: restored
  leveling changes its two late hard-bank samples at `10.422/10.531 s` from
  roll `-0.467/-0.867` to level. Candidate 061 restoration starts at
  `10.297 s`, before its `10.453 s` camera crossing and stale-track divergence.
- Source-hashed replay covers every Candidate 016--062 trace with no blockers.
  Every protected clean recorded action remains exact on Candidates
  016/018/020/021/022/055/058/060. Only roll can change, every output is
  finite/in range, and Candidate 062's command-free post-stop proof is
  embedded.
- Focused dependency/policy/runner tests pass `39/39` on Linux and `39/39`
  under the actual Windows Python; the PowerShell deployment parser passes.
  Policy/verifier/offline-report/runner SHA-256 values are
  `acf17a600e25e2c4c43e928052789c03dfa934e865e7bbd7e7d39ddd9a056885`,
  `0565d47bf18dc29863d64473ddfdda739ba24cfac8eca6c3b2ebaf54f0813d5d`,
  `538fb805f92704446b205f00f74dd1ccf561e3d204f0033c6118533ac7fab2ff`,
  and `27ffde607c8772246d25c49b85f57f744a31200ecd4936db272fe94410f60c5d`.
- No simulator command occurred during N200 development. Candidate 063 and
  FullLap remain unauthorized. Next only: preserve PID `24476`; recover the
  sole VQ1/R1 UI with `-NoRelaunch -SkipLoginClick`; send normal disarm, wait
  `100 ms`, then exactly one learned-target MAVLink `COMMAND_LONG 31000`,
  confirmation `0`, parameters 1--7 zero. Require boot rollback greater than
  `1000 ms`, fresh nonnegative race start, index `0`, finish `-1`, visible
  Gate 1, then passive zero-command
  `n200_gate3_restored_terminal_level_n199_shadow_061` with
  `-Gate3RestoredTerminalLevelN199`. Only a passing shadow may authorize one
  separately preregistered Candidate 063 `BoundedGate3`.


## N200 Reset-31000 Proof and Passive Shadow 061 — Passed

- The sole VQ1/R1 simulator remained the existing `FlightSim.exe` PID `24476`;
  no relaunch occurred. Command-free readiness showed boot/race
  `541086/524495 ms`, base-mode/system-status `193/4`, official index `0`,
  finish `-1`, and `217` visible detections.
- After normal disarm and the required `100 ms` wait, exactly one learned-target
  MAVLink `COMMAND_LONG 31000` was sent with confirmation `0` and parameters
  1--7 all zero. The resulting boot rollback was `541086 -> 3095 ms` and the
  new race epoch was `3318 ms`; official index was `0`, finish was `-1`, no
  collision was present, and Gate 1 remained visible. The later command-free
  readiness snapshot retained race epoch `3318 ms`, index `0`, finish `-1`,
  base-mode/system-status `193/4`, and `218` detections.
- Passive tag `n200_gate3_restored_terminal_level_n199_shadow_061` passed with
  the frozen N200 callable: `278/278` visible samples replayed, maximum action
  error `3.9267263413078624e-07`, and `27.8 Hz` inference. Its cadence probe
  counted `587` evaluations over `9.969 s` (`58.782 Hz`) with zero rate
  violations and zero MAVLink setpoints. It received `2435` telemetry messages,
  including `1254` HIGHRES_IMU messages; all `117/117` detector frames were
  detections, with zero collisions, dropouts, drain-limit hits, malformed
  messages, arm commands, disarm commands, reset commands, or control commands.
- UI-ready/start, UI snapshot, reset proof, reset-ready snapshot, shadow, and
  parity SHA-256 values are respectively
  `b6cb0aa780a7908f03201a8bf1e4ebdc6ad016aad4fb8738023852a691c95bf8`,
  `8607b74be429046aa476be081c3506676089613d525f9d07a9e1259f999ed93d`,
  `9f1c8a09c595d381602a98b82341cd35e3e5215f968a474fbf86ab28b4047567`,
  `366b3a1731f58e8077ef7cd8c660bfe010f7d51ee6e035f812ad802eb5000209`,
  `3a973821a930549c6064c7878b7ae3e27d554ee396919eef24ac698bbfdebbee`,
  and `9a9895b0767d2961dfdc7fbb6c730131851023a98da4353c19e82adda4e8c898`.
- Exactly one Candidate 063 is now authorized: tag
  `n200_gate3_restored_terminal_level_n199_bounded_063`, mode `BoundedGate3`,
  frozen `-Gate3RestoredTerminalLevelN199`, `80 Hz`, `45 s`, repeats `1`,
  minimum valid runs `1`, official stop index `3`, and the same PID. It must
  receive another exact reset-31000 proof immediately before arming. Candidate
  064 and `FullLap` remain unauthorized; pass or fail, disarm and capture a
  command-free post-stop snapshot without resetting.


## Candidate 063 N200 BoundedGate3 — Failed at Gate 1

- Candidate 063 used the sole existing PID `24476`, tag
  `n200_gate3_restored_terminal_level_n199_bounded_063`, exact N200 callable,
  `BoundedGate3`, `80 Hz`, `45 s`, repeats `1`, minimum valid runs `1`, and no
  relaunch. Its embedded reset proof sent the required pre-reset disarm and one
  MAVLink `COMMAND_LONG 31000`; boot rolled back `375215 -> 3415 ms`, the new
  race epoch was `3292 ms`, reset detection took `0.610 s`, and calibration
  used `60` samples.
- The run failed acceptance and stopped after `4.594 s`: camera counter pass 1
  occurred at `4.391 s` and `2.318841 m`, but the in-run official snapshot was
  still index `0`. Collision `1001`, threat `2`, impact `1.9213672876358032`
  then made the run invalid. It sent `258` setpoints over `4.469 s`
  (`57.507 Hz`) with zero rate violations and received `2020` telemetry
  messages, including `1022` HIGHRES_IMU messages, with zero dropouts,
  drain-limit hits, or malformed messages.
- The command-free, no-reset post-stop snapshot later exposed the authoritative
  delayed transition: official index `1`, last-gate time `4.675325393 s`, race
  epoch still `3292 ms`, and finish `-1`. It was disarmed at base-mode/status
  `65/3`; there was no continuing collision. Thus Gate 1 was officially passed,
  but the vehicle struck the gate immediately afterward. Its closest accepted
  pose had the gate `0.518116 m` down at `2.318841 m` range; this is a Gate-1
  vertical-terminal miss, before any Gate-3-only N200/N199/N198 controller can
  activate. An unchanged N200 retry is forbidden.
- Reuse-check, probe, smoke, attempt, summary, CSV, and post-stop SHA-256 values
  are respectively
  `916f30383e11fc24f763ab0d0a85847d9ec0594021efd8139679536a87e0fb03`,
  `4635f809070cf47bf3002eb7c893d654eb911a5d48451074bc8470ada559f162`,
  `1112c264fe995c94df74184276adcdd061935fcdf0fda072eba63966554df778`,
  `12ddbfe259dd4eaf46a7d23b46648f81e30fb65059398fb33ac654c5ae63e7ad`,
  `8871f2643ab7f3c5f2155565957b7561b07205b08aa5033c712cbbaf8f295d4d`,
  `32167184fb78a5e2d852633469018d00448cdf64047903a89b6c492fdf29ddbd`,
  and `ab1a66d16b2e0fa70c40eec20a0ca0e4da2e4bace84c87337a73066640e57b8f`.
- Candidate 064 and `FullLap` remain unauthorized. Next only: use the archived
  trace corpus to design the smallest source-pinned N201 Gate-1 terminal
  vertical correction, prove that it activates on Candidate 063 while leaving
  protected clean Candidates 055/058/060 exact, then require a fresh reset-31000
  proof and passive shadow before any new live command.


## N201 Gate-1 High-Down Thrust Ceiling — Offline Freeze

- N201 evaluates exact source-pinned N200 once, then applies one stateless
  Gate-1-only rule: while the official gate index is `0`, the gate is visible,
  range is in `(0, 6.0] m`, closing speed is at least `1.0 m/s`, and the gate
  is at least `0.35 m` down, normalized thrust is capped at `-0.70`. Pitch,
  roll, yaw, N200, every Gate-2/Gate-3 controller, checkpoints, and the six-gate
  course contract remain unchanged.
- Source-hashed replay covers every Candidate 016--063 with no blockers. The
  rule activates only on Candidate 026 (`6` stored samples), Candidate 036
  (`1`), and causal Candidate 063 (`5`). On Candidate 063 it begins at
  `4.000 s`, before the `4.391 s` camera crossing and `4.675325393 s` official
  transition, and changes only thrust on all five samples. Protected clean
  Candidates 055/058/060 remain bit-exact; all outputs are finite and bounded.
  Candidate 063's command-free, no-reset official-index-1 post-stop proof is
  embedded in the report.
- Focused dependency/policy/runner tests pass `47/47` on Linux and `47/47`
  under the actual Windows Python; the PowerShell deployment parser passes.
  Policy/verifier/offline-report/runner SHA-256 values are
  `1c555692497ebb2e0ec35c309158ee55813d5caad299709483439bf636c4c1e9`,
  `cdd0fb38570c24f36a23beb485a82969c80ece28f619a0bc84d72700dcae9155`,
  `5ac86c7d905539e0978c3ff9094a2574f5901846318bfa6d0f3a41a833accf1a`,
  and `3edbfb3df8d15c35f5c8bc3871aa9a2345dde9f0d310b3a4f1f87f1e10e3d143`.
- No simulator command occurred during N201 development. Candidate 064 and
  `FullLap` remain unauthorized. Next only: preserve PID `24476`; recover the
  sole VQ1/R1 UI without relaunch if base-mode/status remains `65/3`; perform
  normal disarm, wait `100 ms`, then exactly one MAVLink `COMMAND_LONG 31000`
  with confirmation `0` and parameters 1--7 zero. Require boot rollback greater
  than `1000 ms`, a fresh nonnegative race epoch, index `0`, finish `-1`, and
  visible Gate 1, then run passive zero-command tag
  `n201_gate1_high_down_thrust_ceiling_n200_shadow_063` with
  `-Gate1HighDownThrustCeilingN200`. Only a passing shadow may authorize one
  separately preregistered Candidate 064 `BoundedGate3`.


## N201 Reset-31000 Proof and Passive Shadow 063 — Passed

- The sole VQ1/R1 simulator remained the existing `FlightSim.exe` PID `24476`;
  the inactive `65/3` post-crash state was recovered through the UI with
  `-NoRelaunch -SkipLoginClick`, and no process restart occurred. Command-free
  readiness showed boot/race `1029754/1010353 ms`, base-mode/status `193/4`,
  official index `0`, finish `-1`, no collision, and `182` detections.
- After normal disarm and the required `100 ms` wait, exactly one learned-target
  MAVLink `COMMAND_LONG 31000` was sent with confirmation `0` and parameters
  1--7 all zero. Boot rolled back `1029754 -> 3236 ms`, the fresh race epoch
  was `3265 ms`, official index was `0`, finish was `-1`, no collision was
  present, and Gate 1 remained visible. The later command-free ready snapshot
  retained epoch `3265 ms`, index `0`, finish `-1`, base-mode/status `193/4`,
  no collision, and `211` detections.
- Passive tag `n201_gate1_high_down_thrust_ceiling_n200_shadow_063` passed with
  the frozen N201 callable: `304/304` visible samples replayed, maximum action
  error `2.0600662231640143e-07`, and `30.35446829755367 Hz` inference. Its
  cadence probe counted `619` evaluations over `10.015 s` (`61.707 Hz`) with
  zero rate violations and zero MAVLink setpoints. It received `2427` telemetry
  messages, including `1250` HIGHRES_IMU messages; all `126/126` detector
  frames were detections, with zero collisions, dropouts, drain-limit hits,
  malformed messages, arm commands, disarm commands, reset commands, or
  control commands.
- UI-ready/start, UI snapshot, reset proof, reset-ready snapshot, shadow, and
  parity SHA-256 values are respectively
  `e8f617b993480b3eb3ab4a8cdbc5b932bc8d0ce5eaff0a39bce9b437242f3261`,
  `fb167e7d522673a0d5447f6ae03ae281c06668acf788dbf3229af92958cffe2f`,
  `ffcf0dd508541ce8429277453fc0892219bb379cf3674bf8d191be5a974613ac`,
  `d0540228e5f169511b7ec86c609376203487c35dc82053bd8548a47a63ca5733`,
  `3a64fe1dbec9c82929445926969d7a0022e4a615a935ef8b164fdc8c2f77a535`,
  and `e54e57cdf2a56b7451507d7e04005c9ba2306cdcdf04c35e72de4f5d84843592`.
- Exactly one Candidate 064 is now authorized: tag
  `n201_gate1_high_down_thrust_ceiling_n200_bounded_064`, mode `BoundedGate3`,
  frozen `-Gate1HighDownThrustCeilingN200`, `80 Hz`, `45 s`, repeats `1`,
  minimum valid runs `1`, official stop index `3`, and the same PID. It must
  receive another exact reset-31000 proof immediately before arming. Candidate
  065 and `FullLap` remain unauthorized; pass or fail, disarm and capture a
  command-free post-stop snapshot without resetting.


## Candidate 064 N201 BoundedGate3 — Official Gate 3, Collided and Invalid

- Candidate 064 used the sole existing PID `24476`, tag
  `n201_gate1_high_down_thrust_ceiling_n200_bounded_064`, exact N201 callable,
  `BoundedGate3`, `80 Hz`, `45 s`, repeats `1`, minimum valid runs `1`, and no
  relaunch. Its embedded reset proof sent the required pre-reset disarm and one
  MAVLink `COMMAND_LONG 31000`; boot rolled back `262331 -> 3381 ms`, the fresh
  race epoch was `3300 ms`, reset detection took `0.531 s`, and calibration
  used `60` samples.
- Gates 1--3 advanced officially; Gate 3 recorded at `10.561188697 s` and the
  stop fired at index `3` after `10.532 s` control. This is not a valid pass:
  collision `1001`, threat `2`, impact `4.9736151695251465` was present at the
  stop, two collision messages were received, and smoke acceptance explicitly
  failed on `crash_detected` and `invalid_run`. The camera counter was ordered
  through three, but its explicit events contained only Gates 1 and 2. N201's
  Gate-1 ceiling was inactive, so the new Gate-1 correction did not perturb
  this trajectory.
- The run sent `607` setpoints over `10.407 s` (`58.230 Hz`) with zero rate
  violations and received `3270` telemetry messages, including `1650`
  HIGHRES_IMU messages, with zero dropouts, drain-limit hits, or malformed
  messages. Near Gate 3, N200's restored terminal leveler zeroed roll while the
  gate remained `0.584034 m` left at `2.689076 m`; the official crossing and
  collision followed. This is the next causal boundary to analyze offline.
- The command-free, no-reset post-stop snapshot showed inactive base-mode/status
  `65/3`, cleared race epoch/index `-1/0`, finish `-1`, and continuing collision
  `1001` (threat `1`, impact `0.021709894761443138`; `90` collision messages).
  No reset was sent.
- The validation wrapper exposed a fail-open aggregation bug: its child exit
  code was `1`, status was `smoke_failed_acceptance`, and smoke acceptance was
  false, yet official index `3` alone set `gate_progress_passed=true`, producing
  aggregate promotion success and outer exit `0`. This result is manually and
  authoritatively rejected. Future aggregation must require both clean smoke
  acceptance and official progress.
- Reuse-check, probe, smoke, attempt, summary, CSV, and post-stop SHA-256 values
  are respectively
  `c772678bb8c53d1b914452ee9145f69d0c6805813c4b87306aca7623126934c6`,
  `31b473bed37d73c667026967c4c28122379acd347febac80b004f67e3c833d37`,
  `b9fbb0d3e5dfcbc669be9be1c020b3c474310cf8f1fede3562f06915e5b9ca82`,
  `98685bd9690cc77dfb19ab454bdff4cf3ee4c36a5ec6bd04e3482dc345a66f27`,
  `1be5fcd7aa083c8a86643c0cbdbd36996d962cee1a239bf623197f4efe68689d`,
  `92db9c670fd84c438b1d0c8d11e56de23df8958fd07a410493ef11fe34bd2a97`,
  and `816fb5a86f1198f3674f041fd52a1ad9c43062273947eee8c965005b6c38cdd9`.
- Candidate 065 and `FullLap` remain unauthorized. Next only: make official
  validation fail closed on child exit/smoke acceptance/crash validity, then
  use source-hashed replay to replace N200's `0.5 m` Gate-3 terminal projection
  admission with the smallest separator that preserves C060 and the intended
  C061/C062 corrections but does not level Candidate 064's unsafe lateral path.


## N202 Fail-Closed Validation and Tight Gate-3 Terminal Corridor — Offline Freeze

- Official policy validation now fails closed. Official progress counts only when
  the child exits `0`, smoke `acceptance_passed` is explicitly true,
  `crash_detected` and `invalid_run` are not true, and the official index
  reaches the requested minimum. Candidate 064's archived child-exit-`1`,
  rejected, collided fixture therefore evaluates false and cannot promote from
  its official index `3` alone.
- N202 evaluates exact source-pinned N199 once, applies the restored Gate-3
  terminal roll leveler with only the projected-lateral admission tightened
  from `0.50 m` to `0.30 m`, then applies the exact N201 Gate-1 thrust
  ceiling. The `3.7 m` forward limit, `2.0 m` projection plane, `1.0 m/s`
  minimum closing speed, `1.0 m` projected vertical bound, zero-roll output,
  latch, all non-roll actions, checkpoints, and six-gate contract are unchanged.
- Source-hashed replay covers every Candidate 016--064 trace with no blockers.
  Candidate 061 remains admitted at projected-right `+0.257 m` at
  `10.297 s` and latches for `308` samples; Candidate 062 remains admitted at
  `+0.093 m` at `10.422 s` for both samples. Candidate 064's prior N200
  admissions at `-0.340628 m`/`-0.373181 m` at `10.313/10.454 s` are both
  rejected. Protected clean Candidates 055/058/060 remain action-exact, no
  non-roll output changes, and every output is finite and bounded. The report
  embeds Candidate 064's rejected attempt and command-free collision post-stop
  evidence.
- Focused dependency/policy/validator/runner tests pass `59/59` on Linux and
  `59/59` under the actual Windows Python; the PowerShell deployment parser
  passes. Validator/policy/verifier/offline-report/runner SHA-256 values are
  `2be1b44399a5b19dce2204567efbd2cad38099026ff765eabc01207fb9cef311`,
  `892f38e34da75f29d7b20a42638ff26a19773518ecab751e4f408842733538dd`,
  `d57b0a8b5e1a3842f756e8bae7bca6e40048f023a5423c05b04242a46c3463ee`,
  `bc46ba1bdfeb0424b041af6c3517f3a04caccaed356000cb34fa024bb493f82b`,
  and `4c37ce1bc21dadf2e475471cb591e0035e2e45a74100a95c433df84938c72233`.
- No simulator command occurred during N202 development. Candidate 065 and
  `FullLap` remain unauthorized. Next only: preserve sole `FlightSim.exe` PID
  `24476`; recover the VQ1/R1 UI with `-NoRelaunch -SkipLoginClick` if needed;
  send normal disarm, wait `100 ms`, then exactly one learned-target MAVLink
  `COMMAND_LONG 31000`, confirmation `0`, parameters 1--7 all zero. Require a
  boot rollback greater than `1000 ms`, fresh nonnegative race epoch, official
  index `0`, finish `-1`, and visible Gate 1, then run passive zero-command tag
  `n202_gate3_tight_terminal_level_n201_shadow_064` with
  `-Gate3TightTerminalLevelN201`. Only a passing shadow may authorize one
  separately preregistered Candidate 065 `BoundedGate3`.


## N202 Reset-31000 Proof and Passive Shadow 064 — Passed

- The sole shipping simulator remained responsive as PID `24476`; Windows also
  exposed launcher PID `25552`, but no process was relaunched. The inactive
  `65/3` collision state was recovered through the UI with
  `-NoRelaunch -SkipLoginClick`. Command-free readiness then showed
  boot/race `1193933/1182994 ms`, base-mode/status `193/4`, official index
  `0`, finish `-1`, no collision, and `196` detections.
- After normal disarm and the required `100 ms` wait, exactly one learned-target
  MAVLink `COMMAND_LONG 31000` was sent with confirmation `0` and parameters
  1--7 all zero. Boot rolled back `1193933 -> 3221 ms`, the fresh race epoch
  was `3283 ms`, official index was `0`, finish was `-1`, no collision was
  present, and Gate 1 remained visible. The later command-free ready snapshot
  retained epoch `3283 ms`, index `0`, finish `-1`, base-mode/status `193/4`,
  no collision, `16250 ms` boot time, and `234` detections.
- Passive tag `n202_gate3_tight_terminal_level_n201_shadow_064` passed with the
  frozen N202 callable: `311/311` visible samples replayed, maximum action error
  `2.493780136107737e-7`, and `30.908 Hz` inference. Its cadence probe counted
  `605` evaluations over `10.0 s` (`60.4 Hz`) with zero rate violations and
  zero MAVLink setpoints. It received `2423` telemetry messages, including
  `1248` HIGHRES_IMU messages; all `126/126` detector frames were detections,
  with zero collisions, dropouts, drain-limit hits, malformed messages, arm
  commands, disarm commands, reset commands, or control commands.
- UI-ready/start, UI snapshot, reset proof, reset-ready snapshot, shadow, and
  parity SHA-256 values are respectively
  `e75df46c68a522330c5052dea6d13b703f759988129210803c3947687ca57458`,
  `c6871330efafeadd994ea9fa3b52303902c10bd9930962800ad9e8ad9032de93`,
  `7999da78a30970549a0905bcf9f5d8a0bc91b16a270f5d6b3e9b0872d2cc59bc`,
  `7f651542f8aa3d6b05913c734c2e4c564838926d28fbcf4b18bd1b8c2c1a992d`,
  `504af65c772ce64afb648d3397556d71f6d9985bec40501989080fc9b29bf635`,
  and `6a1542ca2792375edbbe54c42811a3cbea9cb74dc84074b890c402b6896b752f`.
- Exactly one Candidate 065 is now authorized: tag
  `n202_gate3_tight_terminal_level_n201_bounded_065`, mode `BoundedGate3`,
  frozen `-Gate3TightTerminalLevelN201`, `80 Hz`, `45 s`, repeats `1`,
  minimum valid runs `1`, official stop index `3`, and the same shipping PID.
  It must receive another exact reset-31000 proof immediately before arming.
  Candidate 066 and `FullLap` remain unauthorized; pass or fail, disarm and
  capture a command-free post-stop snapshot without resetting.


## Candidate 065 N202 BoundedGate3 — Gates 1–2 Official, Collided and Rejected

- Candidate 065 used the sole shipping PID `24476`, tag
  `n202_gate3_tight_terminal_level_n201_bounded_065`, exact N202 callable,
  `BoundedGate3`, `80 Hz`, `45 s`, repeats `1`, minimum valid runs `1`, and
  no relaunch. Its embedded reset proof sent the required pre-reset disarm and
  one learned-target MAVLink `COMMAND_LONG 31000`; boot rolled back
  `177116 -> 3381 ms`, the fresh race epoch was `3297 ms`, reset detection
  took `0.515 s`, and calibration used `60` samples.
- The attempt stopped invalid after `7.703 s`. Its camera counter reached two,
  with the explicit Gate-2 event at `7.546 s` and `2.165414 m`, while the
  in-run official snapshot still showed index `1` and last-gate time
  `5.066904544 s`. Collision `1001`, threat `2`, impact
  `6.18809700012207` then triggered immediate rejection. The hardened wrapper
  correctly retained child exit `1`, `gate_progress_passed=false`, zero valid
  passes, `promotion_passed=false`, and outer exit `1`.
- The command-free, no-reset post-stop snapshot exposed the delayed official
  Gate-2 transition: index `2`, last-gate time `7.863622665 s`, unchanged
  race epoch `3297 ms`, and finish `-1`. It was disarmed at base-mode/status
  `65/3`, the collision event had cleared, and no reset was sent. Thus Gates 1
  and 2 passed officially, but the vehicle struck immediately after Gate 2.
  N202's changed Gate-3 branch never received official index `2` during live
  control and did not activate; this run does not validate the N202 separator.
- Transport remained healthy: `472/7.547 = 62.409 Hz`, zero rate violations,
  `2633` telemetry messages, `1330` HIGHRES_IMU messages, `97` detector
  frames/detections, and zero telemetry dropouts, drain-limit hits, or malformed
  messages. Reuse-check/probe/smoke/attempt/summary/CSV/post-stop SHA-256 values
  are respectively
  `16899e9aa45176ea3f9682640d167d45d563732ccabccb72ac3c84e88c6452c8`,
  `ab498dd94d29dc96e65ae742943deeca6ffc88a85568e0e1fefc784de96a01cc`,
  `d3054c5eb60a1e5ab78972ba62de3fb07761d0e37f5b12d3b04b5c62dd7460d5`,
  `e3b525c5beacea1a1bfb4353a26657480b2b833410775dc3509e867de8397feb`,
  `86b09516a71b8b6b643fb1b155a580c0095656665b0f563b94018c8bdb64dacc`,
  `7888ffe734dede2c4a754241c484402abfab7c476af2e4d3aa571feea6257063`,
  and `7540e8973f147d5e11d96ecc04edd5968f40ba47f478e195763fa0367f8e11fb`.
- Candidate 066, unchanged retries, and `FullLap` are unauthorized. The
  single-trace patch/one-attempt loop is closed: Candidate 065 failed before the
  changed branch could run, demonstrating prefix outcome variation. Next only:
  define and implement a repeated-run six-gate optimizer whose lexicographic
  objective is valid finish, clean official gate count, then lap time; automate
  the same MAVLink-reset/same-process recovery contract; and promote only from
  aggregate batch evidence rather than one crash trace.


## Six-Gate Optimizer Stages 1–3 and N205 Gate-4 Hold — 2026-07-18

- The active objective is a competitive VQ1/R1 lap through all six official
  gates. No candidate is promoted without official gate index 6, a nonnegative
  finish time, no collision, and repeated evidence. The only simulator is
  `C:\\Users\\anon\\Desktop\\AI-GP Simulator v1.0.3385\\AIGP_3385\\FlightSim.exe`;
  reuse the responsive shipping process (PID `24476` in these stages) and do
  not relaunch it. Every attempt uses normal disarm, a `100 ms` wait, and one
  learned-target MAVLink `COMMAND_LONG 31000` with confirmation `0` and
  parameters 1–7 all zero. Reset acceptance still requires a boot rollback
  greater than `1000 ms`, a fresh nonnegative race epoch, official index `0`,
  and finish `-1`. Collision-inactive `65/3` is recovered through the UI in
  the same process with `-NoRelaunch -SkipLoginClick`.
- The new resumable, dry-by-default tournament runner is
  `scripts/run_windows_six_gate_batch_optimizer.ps1`. It gives each attempt
  exactly one reset/arm/flight/disarm lifecycle, captures a command-free
  five-second post-stop proof, and stops on the first valid finish. The scorer
  `scripts/score_official_six_gate_batch.py` ranks by valid finishes, minimum
  official gate count, median official gate count, collision-free count,
  collision count, then time. Scorer SHA-256 is
  `538161ed73bf27fff14c2cd8eb6b59dbfc75981477c10e060073112d3724a48c`.
- Stage 1, `competitive_six_gate_batch_001`, ran N195, N197, and N198 once
  each. All three ended at official index `2`; N195 collided after `8.344 s`,
  while N197 and N198 remained collision-free for the full `45 s` but
  deadlocked on Gate 3. There was no finish, so no candidate was promoted.
  Evidence is in the batch manifest, score, and Gate-3 diagnosis JSON under
  `logs/sitl/competitive_six_gate_batch_001_*`.
- Stage 2 introduced N203, an exact N202 parent plus a Gate-3-only counter-bank
  hysteresis with fixed `0.8 m` release gap and frozen entry variants H08,
  H12, and H16. All three advanced from the N202 baseline's official index
  `2` to official index `3`, proving the changed branch in the simulator.
  H08/H12/H16 then collided at Gate 4 after `15.219/15.141/17.937 s`;
  baseline N202 collided at index `2` after `16.984 s`. H12 ranked first and
  is retained only as the experimental parent, not as a valid lap. N203 source
  SHA-256 is `b357536767a62beb85ba707a297b67ef5191998a425f6e0912a9882e07ee5cbe`;
  its focused tests passed `8/8`. Aggregate evidence is
  `logs/sitl/competitive_gate3_hysteresis_batch_002_*`.
- Stage 3 tested the isolated Gate-4 yaw-sign hypothesis as N204/YP. The exact
  repeat batch `competitive_gate4_yaw_sign_batch_004` produced official gate
  counts `3,3,2`: two collision-free `45 s` Gate-4 stalls and one Gate-3
  collision at `11.062 s`, with no finish. On the two Gate-4-evaluable runs,
  the positive sign made bearing error grow to about `0.77 rad` and the target
  left the field of view; safety came from the subsequent hover, not correct
  steering. N204/YP is rejected and must not be promoted or used as the next
  parent. Its source SHA-256 is
  `a06c41623d7dd154d63450ff028f3de274d107e59a8c2d7407d88b39c2628db9`.
- Association replay isolated the next contradiction. In the original H12
  trace, Gate 4 acquired a coherent anchor, but later visible poses were
  rejected as discontinuous; the parent association correctly rejected them
  while the yaw acquisition path still overwrote its yaw target from those
  same rejected detections. N205 fixes only that inconsistency: after an anchor
  exists, a visible rejected pose retains the last coherent associated yaw
  target with the existing `0.35 rad` bound. Pre-anchor acquisition,
  associated/re-anchored poses, dropouts, yaw sign, and pitch/roll/thrust are
  exact passthrough. Source is
  `scripts/policy_callable_gate4_coherent_target_hold_n203.py`, SHA-256
  `23100ecad829add77d9de88f4778d8ec9d920d3a2d03ce93c460b47ff484d511`.
  The focused Linux suite passed `32/32`, the exact Windows runtime subset
  passed `21/21`, and both PowerShell entry points parse. The deployment
  runner SHA-256 is
  `c22f36ca037efa76abc67a188a1eb69c5f0143d7996773eb0ff18a51f4d8b904`;
  the batch runner SHA-256 is
  `43269e175d3f16d0fc42d4c0daf8d8e60dc1bb2c298860f1bcac4373e091fbb3`.
  The preregistered official batch is
  `competitive_gate4_coherent_hold_batch_005`, variant `CH`, three exact
  `45 s` full-course attempts at `80 Hz`. It is evidence gathering only;
  promotion still requires an actual valid six-gate finish.


## Stage 4 N205/CH Official Batch — Rejected; Learned Gate-4 Tail Next

- `competitive_gate4_coherent_hold_batch_005` completed all three exact
  attempts on the reused responsive shipping PID `24476`; every attempt used
  the required disarm/`100 ms`/single MAVLink `COMMAND_LONG 31000` reset
  proof and a command-free post-stop capture. Official gate counts were
  `3,3,1`, all three attempts collided, no finish timestamp existed, and no
  candidate was promoted. Run 1 reached post-stop index `3` but hit ID `1001`
  after `10.656 s` at impact `1.7049003`; run 2 reached index `3` then hit
  ID `1002` after `15.343 s` at impact `5.5965281`; run 3 hit ID `1001`
  at index `1` after `7.282 s` at impact `8.6681356`. Aggregate minimum/
  median/maximum official count is `1/3/3`, with `0/3` collision-free and
  `0/3` valid finishes. Manifest/score SHA-256 values are
  `d4245e5b6f2fbba474cb97d0613ac526b7d9641533a5ce4c721e410272b4b39b`
  and `3a2f8b9cd7c73167409c2204eab53534a2bb42c46b0f38fd63ecdca5e519d310`.
- N205's coherent rejected-visible yaw hold did activate in the sole Gate-4-
  evaluable run, but it did not solve the collision and is rejected. The trace
  isolates a stronger blocker: Gate 4 became active at `11.062 s`; continuous
  `0.22` thrust began at `12.109 s` while the accepted family was about
  `27.43/3.94/10.76 m` forward/right/down, and persisted until ID `1002`
  contact. This repeats the ledger's Candidate-139/159 ground-contact result:
  far `21–25 m` descent is unsafe, while current-associated close/bounded
  descent was the retained safe form. The present N158-era handoff's one-second
  continuous vertical branch conflicts with that later evidence.
- Do not make another hand-controller scalar child. The historical manual Gate-4
  bracket was already closed at Candidate 207 after no index-4 pass. The next
  smallest distinct mechanism is the already-trained N142 phase-reset tail:
  retain exact H12/N203 for Gates 1–3, but at official Gate 4 bypass the manual
  safety handoff and execute the frozen learned Gate-4/N112 checkpoint directly,
  with a fresh recurrent reset at phase entry. That checkpoint passed its native
  fixed Gate-4 segment `128/128`; it has never received official Gate-4 evidence
  because its earlier N142 live candidate failed in the prefix. Test it as a
  bounded, source-pinned variant and reject it on any collision or lack of
  aggregate official improvement. It is not promoted by native evidence alone.


## N206 Learned Gate-4 Tail Direct — Implemented and Running Stage 5

- `scripts/policy_callable_gate4_learned_tail_n203.py` implements the distinct
  learned-tail experiment. It source-pins H12/N203 for official Gates 1–3.
  Only while official Gate 4 is active, it clears the retired manual Gate-4
  state and calls the frozen N142 composite exactly once using the standard
  coordinate-free 32-value observation with reserved slots `30/31` cleared.
  The N142 composite owns the phase-entry recurrent reset and selects frozen
  N112 step `294912` for Gate 4. Gates 5–6 remain the same learned phase-reset
  composite. No gate coordinate, future-gate state, GPS, or privileged simulator
  state is used. Source SHA-256 is
  `4a976df88dc95af46d2bd022fd8497608ff9cbe3a47c2265c4215e8243e775af`.
- The focused Linux suite passed `86/86`; the exact Windows Python 3.12 runtime
  subset passed `30/30`; both updated PowerShell entry points parse. Deployment
  runner SHA-256 is
  `e85aca0f87438ff531dd8d48801b29bbadc7c94bdaa8436931a941e34dde5fcc`;
  batch runner SHA-256 is
  `21fecd9fba9914e1e8bfb964c600da5d0b91b35e4f8b3dd0e0a3b3fd5172e90a`.
- Stage 5 is the fixed tag `competitive_gate4_learned_tail_batch_006`, variant
  `LT`, three exact full-course attempts, `45 s`, `80 Hz`, on the reused
  responsive v1.0.3385 VQ1/R1 simulator at
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe`.
  Each attempt retains the normal-disarm/`100 ms`/single-command MAVLink-31000
  reset proof and command-free post-stop capture. Stop the batch on any valid
  official six-gate finish. Native success is rationale for testing only; it is
  not promotion evidence.


## Stage 5 N206/LT Official Batch — Rejected; Terminal Crossing Isolated

- `competitive_gate4_learned_tail_batch_006` completed all three exact
  full-course attempts on the reused responsive shipping PID `24476` with the
  required normal-disarm/`100 ms`/single MAVLink `COMMAND_LONG 31000` reset
  lifecycle and command-free post-stop captures. Official gate counts were
  `3,1,3`; every run collided, no finish timestamp existed, and N206 is not
  promoted. Run 1 reached index `3`, then hit ID `1001` at impact
  `3.0251808` after `20.531 s`; run 2 was a prefix failure at index `1`,
  ID `1001`, impact `8.5861702`, after `7.531 s`; run 3 reached index
  `3`, then hit ID `1002` with threat `1`, impact `0.56966996`, after
  `14.469 s`. Aggregate minimum/median/maximum official count is `1/3/3`,
  with `0/3` collision-free and `0/3` valid finishes. Manifest/score
  SHA-256 values are
  `68d5650bef4b52b84391dc078469ace1adc0887f02abe2e1a1944d585c5554d9`
  and `ac117148de934c8b1a8c6813203f48db151f02cf8276e66dea593f1a828524ca`.
- Run 1 supplies the first useful official Gate-4 terminal evidence from the
  learned tail. It drove the visible gate to about `1.471891 m` forward,
  `-0.121891 m` right, `0 m` down, and `-0.082624 rad` yaw error while
  outputting decoded pitch `-0.052567 rad`, roll `0.011956 rad`, thrust
  `0.266508`, and yaw rate `0.001383 rad/s`. The last recorded policy
  sample was likewise centered at `1.576642/-0.226642/0 m` with yaw error
  `-0.142772 rad`, but the official counter stayed at `3` before gate
  contact. The learned tail therefore solved the far approach in this run and
  then commanded a terminal brake instead of a forward crossing.
- Do not reopen the retired manual Gate-4 controller ladder. The next and only
  admitted child is a narrow observable-pose residual on the N206 learned tail:
  once per Gate-4 phase, when a current visible pose is within `3.0 m`,
  `|right| <= 0.5 m`, `|down| <= 0.5 m`, and `|yaw error| <= 0.15 rad`,
  retain the learned roll/thrust/yaw and floor pitch at the already-established
  `+0.06 rad` forward command (normalized `+0.12`) for at most `0.5 s`.
  The bounded pulse may bridge detector dropout but may not retrigger in the
  same Gate-4 phase. These bounds admit the exact run-1 near pass and reject
  run 3's last coherent approach at about `11.85/3.50/-0.33 m`. Test this
  single mechanism as Stage 6; judge it only by official gate count, collision,
  and finish evidence across repeated full-course trials.


## N207 Terminal-Crossing Residual — Implemented and Running Stage 6

- `scripts/policy_callable_gate4_terminal_crossing_n206.py` source-pins N206
  and changes only learned Gate-4 pitch inside the preregistered terminal
  envelope. The one-shot pulse triggers from current visible observation values
  at `0 < forward <= 3.0 m`, `|right| <= 0.5 m`, `|down| <= 0.5 m`, and
  `|yaw error| <= 0.15 rad`; for at most `0.5 s`, normalized pitch is floored
  at `+0.12` (`+0.06 rad`) while learned roll, thrust, and yaw pass through.
  It can bridge a detector dropout only while that pulse is already active,
  cannot retrigger in the same Gate-4 phase, resets at a phase transition, and
  uses no course coordinate or privileged state. Source SHA-256 is
  `b4ccbcb90f30ce866c8a9f827c98d457708ca925345cd1847a6d3ef3ebf80f63`.
- The focused Linux deployment suite passed `90/90`; the actual Windows Python
  3.12 subset passed `37/37`; both PowerShell entry points parse. Unit evidence
  includes the exact Stage-5 run-1 terminal pose, every threshold boundary,
  dropout expiry, no same-phase retrigger, action-channel preservation, and
  phase/reset behavior. Deployment runner SHA-256 is
  `25fe052bed03504f2b0b8ea749db045aa55a2ee73fa006ed3fee4580601245e8`;
  batch runner SHA-256 is
  `0343814d2cca3ab8f9fa79ceb675b435a31667544bce9e0e5303d63a1cb083b2`.
- Stage 6 is fixed as tag `competitive_gate4_terminal_crossing_batch_007`,
  variant `TP`, three `45 s`, `80 Hz` full-course attempts on the already
  running v1.0.3385 VQ1/R1 simulator. Preserve the same shipping PID whenever
  responsive, use the documented MAVLink-31000 reset acceptance proof once per
  attempt, and stop immediately on a valid official six-gate finish. This is
  an experimental child, not a promoted policy.


## Stage 6 N207/TP Official Batch — No Trigger; Raw Tail Contract Rejected

- `competitive_gate4_terminal_crossing_batch_007` completed all three exact
  attempts on responsive shipping PID `24476` with the documented reset and
  post-stop lifecycle. Official counts were `2,3,3`; all three runs collided,
  no finish existed, and N207/TP is not promoted. Run 1 was a prefix collision
  at ID `1001`, threat `2`, impact `3.8700616`, after `7.266 s`. Run 2
  reached Gate 4 and hit ID `1002`, threat `2`, impact `5.7731671`, after
  `15.407 s`. Run 3 reached Gate 4 and hit ID `1002`, threat `1`, impact
  `0.02998717`, after `15.140 s`. Aggregate minimum/median/maximum official
  count is `2/3/3`, with `0/3` collision-free and `0/3` valid finishes.
  Manifest/score SHA-256 values are
  `1500673f6d3642fdb73faf972470a41d45b55ca02abb05f4889a7894585f6c24`
  and `77ffc79ae1ecf6e0dcb4f0f75851b4876133c84910f7d249b366acbea051acbf`.
- The terminal pulse had zero eligible Gate-4 samples in all runs. Run 2's
  raw Gate-4 trace began at about `38.40/4.44/4.62 m` forward/right/down and
  ended at `5.52/2.90/-1.76 m`, yaw error `0.483 rad`. Run 3 began near
  `36.92/6.40/6.63 m`; its closest raw forward sample was
  `3.91/3.74/-0.85 m`, yaw error `0.764 rad`, followed by far-target jumps.
  Neither entered the fixed centered envelope. Do not widen its bounds from
  this evidence; Stage 6 did not causally exercise the changed branch.
- The causal contract error is upstream of the pulse. The hybrid deployment
  globally forced raw Gate-4 observations because the retired manual handoff
  owned its own target association. N206 bypassed that handoff but accidentally
  inherited the raw setting. Consequently, during learned Gate-4 control the
  standard `ObservableGateMotionFilter` was reset and then never initialized;
  the N112 checkpoint received unassociated detector jumps even though its
  learned observation contract expects associated pose and observable rates.
  Stage-6 snapshots show the Gate-4 motion filter `initialized=false`, with
  `23/26` rejected-sample counters inherited from earlier phases but no tracked
  Gate-4 identity. This is a deployment-contract mismatch, not a learned-policy
  hyperparameter.
- The next distinct experiment is filtered-tail N208/FT: exact N207 policy and
  all fixed terminal-pulse bounds, but disable raw observation only for this
  variant so Gate 4 uses the existing observable association/rate filter.
  Preserve the raw setting for historical manual variants. The standard filter
  rejects first aliases beyond `35 m`, rejects discontinuous target jumps,
  holds the last officially associated gate through brief dropout, and resets
  only on official identity change. Validate this configuration offline and in
  the Windows runtime, then run a fixed repeated full-course batch. Do not tune
  filter or pulse thresholds within the batch.


## N208 Filtered Learned-Tail Contract — Implemented and Running Stage 7

- N208/FT uses the exact N207 policy source and fixed terminal-pulse parameters;
  the only change is deployment observation routing. New runner switch
  `Gate4FilteredTerminalCrossingN206` removes the raw Gate-4 environment
  interval, restoring the existing observable motion filter for Gate 4. Raw
  routing remains unchanged for every historical TP/manual variant. Variant
  `FT` is separately named in the batch runner, so raw and filtered evidence
  cannot be conflated.
- The focused Linux suite passed `92/92`; the actual Windows Python 3.12 subset
  passed `41/41`; both PowerShell entry points parse. Policy source remains
  `b4ccbcb90f30ce866c8a9f827c98d457708ca925345cd1847a6d3ef3ebf80f63`.
  Updated deployment runner SHA-256 is
  `8820e500ef3cd9ca48bc06b15af7abca7821174c07dae55daa909acdbee46367`;
  updated batch runner SHA-256 is
  `f2759cacc7f85ae4f06f37eadc5deb8c3c001cec4bc71c2fb2f4e21423fe3c9c`.
- Stage 7 is fixed as `competitive_gate4_filtered_tail_batch_008`, variant
  `FT`, three exact `45 s`, `80 Hz` full-course attempts on responsive
  shipping PID `24476`, with the same reset acceptance and post-stop proof.
  There is no within-batch change. Promotion still requires a valid official
  six-gate finish; otherwise retain or reject only from aggregate official
  counts, collisions, filter initialization, and trace evidence.


## Stage 7 N208/FT Official Batch — Rejected; Train the Six-Gate Policy

- `competitive_gate4_filtered_tail_batch_008` completed all three fixed
  attempts on shipping PID `24476` with the documented reset and post-stop
  proof. Official gate counts were `2,3,2`; every run collided and no finish
  existed. Run 1 collided at ID `1001`, threat `2`, impact
  `9.6425323`, after `10.453 s`. Run 2 reached Gate 4, then collided at
  ID `1002`, threat `1`, impact `0.42715925`, after `20.991 s`.
  Run 3 collided in the prefix at ID `1001`, threat `1`, impact
  `0.06559401`, after `7.297 s`. Aggregate minimum/median/maximum is
  `2/2/3`, with `0/3` collision-free and `0/3` finishes. Manifest and
  score SHA-256 values are
  `c452c97172f4e5d55eb94c3f2226f0e7ff56a5ae24ed9fd042701cf554554af1`
  and `32d37e2fa5173771dccfcf55b8b13894b28c1ed7cb9350428f509d58397c1224`.
- The filtered contract was causally exercised in run 2: Gate-4 identity was
  associated and discontinuous target jumps were rejected. It still failed
  because a long detector dropout caused the last accepted world pose and
  rates to be held indefinitely. The policy then saturated near
  `[0.855,1.0,-1.0,0.000676]` normalized action until contact. This closes
  both the raw-target tail and the indefinitely held filtered tail. N208/FT is
  rejected; do not add a timeout wrapper or continue scalar Gate-4 tuning.
- The next mechanism is actual distributional six-gate training. Extend the
  native reset contract to sample a bounded distribution around measured
  official Gate-4 entry states, mix those starts with ordinary full-course
  starts, and optimize one recurrent policy whose observation includes all six
  gate phases. Screen it first on both full-course and entry-distribution
  native suites, then use the same official simulator lifecycle. This directly
  trains the entry-state variation exposed by Stages 5-7 instead of adding
  another deployment controller.


## N209 Measured-Entry Six-Gate Training — Native Gain, Official Rejected

- The native environment now supports opt-in symmetric reset distributions
  over segment elapsed time, position, velocity, Euler attitude, and body
  rate. Zero amplitudes consume no RNG and preserve every historical fixed
  reset exactly. The native regression and focused script tests pass. N209
  reconstructs its Gate-4 entry envelope from five usable official Stage 5-7
  traces: center position about `64.3,-0.5,-6.9 m`, velocity
  `8.5,0.4,-5.0 m/s`, measured attitude/rate center and bounded jitter. It
  mixes these starts 50/50 with ordinary starts and trains the entire accepted
  N096 recurrent policy for all six gates; there is no per-gate weight freeze,
  controller wrapper, privileged coordinate input, or official simulator use
  during training.
- One `1,048,576`-step FP32 run produced 32 admissible checkpoints. Every
  checkpoint was deterministically screened for exactly 128 fixed full-course
  episodes and 128 measured-entry episodes. Step `589824`, SHA-256
  `efad0967b46d10bd30eebf03c85a6bc2cecdfb9646bab9852a0d5c701357aefb`,
  is the balanced winner: full-course `109/128` success, mean `5.375` gates,
  and measured-entry `20/128` success, mean `3.921875` gates, with zero crash
  and timeout in both. On the identical paired cohorts, unmodified N096 scored
  `103/128`, mean `5.203125`, and `17/128`, mean `3.8359375`, also with zero
  crash/timeout. N209 therefore strictly improves both paired finish counts
  (`+6` full, `+3` entry). Full/entry aggregate SHA-256 values are
  `a0dd3671b7e2cda6f8a2d77e3a1d2648a533a6acb08708a15c447b6918f18c37`
  and `af377c5d0146106413a2d5bf391124a9dfa8915ac5453db24fb6a07a384205a0`.
- The single permitted official attempt used N209 alone, FP32 input 32 with
  progress/6 at observation 23 and reserved-zero columns 24..31, 60 Hz, one
  policy-ready disarm -> MAVLink-31000 reset, and healthy reused shipping PID
  `24476`. Gate 1 passed at `5.018333435 s`; the policy then collided at
  official index `1` with object `1001`, threat `1`, impact
  `0.3188585043`, and stopped after `8.0 s`. It sent 421 commands at
  `53.225 Hz`, with zero command violations, telemetry dropouts, or drain
  limit hits. Official count is `1`, no finish exists, and N209 is rejected
  as a start-to-finish official candidate. Smoke/aggregate SHA-256 values are
  `b1ecf4b19b8e26a66cd66e5b8fdeb77bb783cfb20a2b72feb3dc19156f7cb566`
  and `b0079813186289b4cce03b34980b69c868b93694ae9a4208fb93bfd8bddc1bac`.
  A passive post-stop proof reports base/status `65/3`, responsive PID
  `24476`, official index `1`, finish `-1`; SHA-256
  `259bcf891ed7a5294bf4e5e648f7338e44ccd85e8ffe42beda87f812bb44b62e`.
- Causal conclusion: measured-entry training improves native six-gate
  completion but does not repair the known native-to-official Gate-2 prefix
  transfer gap. Do not repeat N209 start-to-finish unchanged. The simplest
  next candidate is the already official-proven H12 Gates-1-through-3 prefix
  followed by one reset N209 recurrent tail for Gates 4-6. This uses the new
  policy exactly on the measured entry distribution it improved, replaces the
  rejected per-gate/manual tail, and adds no action residual.


## N210–N217 Unified Tail, Teacher Intervention, and Distillation — 2026-07-18

- The active objective remains one competitive, collision-free official VQ1/R1
  lap through all six ordered gates. There is still no valid six-gate finish.
  Continue to reuse responsive shipping PID `24476`; the exact simulator,
  Windows Python, six-gate semantics, and normal-disarm/`100 ms`/single
  MAVLink `COMMAND_LONG 31000` reset proof at the top of this file remain
  mandatory. No unchanged N209–N217 retry is authorized.
- N210 combined the exact official-proven H12/N203 Gates-1-through-3 prefix with
  one zero-state N209 recurrent controller for the complete Gates-4-through-6
  tail. The topology is
  `scripts/policy_callable_n209_unified_tail.py`, SHA-256
  `5a00e97df818477e18424cbccb9c21fa53c435a05a8602572390419c420f7769`.
  Passive shadow `n210_h12_n209_unified_tail_shadow_002` passed with 269
  inference ticks, maximum replay error `1.6844e-7`, all manifests exact,
  zero setpoints, and `60.403 Hz` cadence; shadow/parity hashes are
  `8ef3d1fc...`/`2ca0b6a8...`.
- N210 had two real official attempts and one void startup. Attempts 001 and 003
  both passed official Gates 1–3 and then failed Gate 4. Attempt 001 stayed
  collision-free until the 45 s timeout at index `3`; attempt 003 hit object
  `1002`, threat `2`, impact `9.3623`, at index `3`. Transport was
  healthy at `60.502/60.011 Hz` with zero rate violations, dropouts, or drain
  limits. Attempt 002 sent no controller action and is void: command 31000 was
  not acknowledged because the startup reused a stale inactive race. UI
  recovery returned the same PID to `base_mode=193/system_status=4` and a
  fresh race epoch. Real-attempt smoke hashes are `1195de4f...` and
  `93058a9c...`; the void summary hash is `a8fb0fc1...`.
- N211 reward-only spline-teacher shaping did not improve the N209 tail. Its
  best exact common-256 checkpoint was `40/256=15.625%`, mean
  `3.9140625` gates, with `2/256` crashes; evaluation SHA-256 is
  `4e7caee2...`. It was rejected without live deployment.
- N212 introduced a default-off teacher-intervention mechanism in the native
  plant. Before each attitude-mode step, selected pitch/roll/thrust channels
  can be convexly blended with the teacher action; the policy's raw sampled
  action remains the imitation target, while the observation's last-action
  fields record what the plant actually executed. Blend zero preserves the
  historical dynamics exactly. Native regression and focused tests pass.
  Source hashes are `drone_race.c=e0a0b815...`,
  `drone_race.h=28b7112a...`, `binding.c=1f680ec9...`, and the config
  `2303645c...`.
- A 100% roll/thrust intervention probe completed `56/128=43.75%` measured
  six-gate tails with zero crash; probe hash is `de8df645...`. N212 training
  at blend 1.0 produced only `18/128=14.06%` when evaluated unassisted. N213
  continued at blend 0.5 and produced `21/128=16.41%`, zero crash. N214 then
  consolidated at blend zero; screening every admissible checkpoint selected
  early step `65536`, SHA-256
  `999ab9061ca04f7d4b66d33991afd7fa14d1ca0049d9398a66f361419a15fa75`.
  It scored `23/128=17.97%`, zero crash on the selection cohort, then
  `53/256=20.70%`, mean `4.0390625` gates, `2/256` crashes on confirmation.
  Confirmation JSON SHA-256 is `1848720a...`. Later N214 checkpoints degraded;
  promote only step 65536 as a native experimental artifact, not as a valid
  official policy.
- N215 packaged that exact N214 checkpoint behind hash pins while preserving
  the N210 topology. Callable
  `scripts/policy_callable_n214_unified_tail.py` has SHA-256
  `b13b36d7c22af83eb6f2c0e1faf8f8df340fc02e49df8d555064b8aa1751d4ba`.
  The first passive shadow correctly failed in stale collision state
  `65/3`; same-PID UI recovery restored R1. Shadow 002 then passed: 192
  replayed ticks at `19.141 Hz`, maximum error `2.2291e-7`, all manifest
  artifacts exact, zero setpoints, and `54.411 Hz` cadence. Shadow/parity
  hashes are `85515e46...`/`eacd769b...`.
- The single N215 official attempt used a verified MAVLink-31000 reset and was
  valid. It passed Gates 1–3, stayed collision-free, but timed out after 45 s at
  official index `3`; finish remained `-1`. It sent 2723 commands at
  `60.616 Hz`, with zero rate violations, telemetry dropouts, or drain-limit
  hits. Smoke/attempt/summary SHA-256 values are `192fbaba...`,
  `8a9bf0fe...`, and `3641c0dc...`. At Gate-4 activation (`10.641 s`),
  the filter correctly zeroed the first three observations while rejecting
  `64–37 m` aliases. The first accepted family appeared near
  `31 m forward / 7 m right / 6 m down`; roll saturated negative, later
  counter-banked too late, overshot laterally, and the held dropout state drove
  indefinite saturation. This is the same live transfer failure as N210, not a
  transport/reset failure.
- N216 showed that forcing the teacher's fixed pitch is not a solution:
  full-channel intervention scored only `10/128=7.8125%`. N217 then captured
  a six-gate roll/thrust-intervention cohort (`59/128=46.09%`, zero crash) and
  built 128 recurrent episodes/128,727 labels solely from the next observation's
  executed last-action fields. Trace/dataset hashes are `7b06d396...` and
  `deca47cb...`; converter source is
  `scripts/build_teacher_intervention_dataset.py`, SHA-256 `021444ec...`.
  Five decoder-ridge children and one full recurrent child were evaluated on
  disjoint episode offset 128. The unchanged N214 parent scored
  `22/128=17.19%`; the best ridge child scored `4/128`, and the recurrent
  child `3/128`, all with new crash/timeout regressions. N217 is rejected:
  low teacher imitation error does not survive the partially observable
  closed-loop dropout dynamics.
- Current causal boundary: native entry training and privileged teacher
  intervention can improve synthetic completion, but the deployed observation
  stream does not identify the teacher's correction during long Gate-4
  association dropouts. Do not run another reward-only teacher, blend schedule,
  decoder fit, recurrent distillation, N209/N214 unified-tail retry, scalar
  Gate-4 wrapper, or live attempt from this lineage. The next admitted work must
  change observability or state estimation and prove that change on held-out
  closed-loop native evidence before FlightSim.
- N218 made that state-estimation change without changing N214's checkpoint or
  policy topology. A new default-off gate-motion option advances the last
  associated world-frame gate pose by its observable filtered motion during
  detector dropout. The matching live contract is enabled only by
  `PUFFER_POLICY_PREDICT_GATE_DROPOUT=1`; default-off behavior remains
  historical. Python/native regression passed, the CUDA FP32 extension was
  rebuilt, and 61 focused tests plus the native regression passed after runner
  integration.
- On two disjoint measured Gate-4 native cohorts, the unchanged N214 weights
  improved from `23/128` to `43/128` and from `22/128` to `30/128`, for an
  aggregate `45/256=17.58%` to `73/256=28.52%`, with no added crashes.
  Evaluation JSON SHA-256 values are `d30d2871...` and `9137331d...`.
- N218 shadow 001 correctly failed against stale collision state `65/3` and
  emitted zero inference/setpoints. Same-PID recovery restored R1 to
  `base_mode=193/system_status=4`. Shadow 002 passed with 251 replay ticks,
  `25.022 Hz` inference, maximum replay error `1.8254e-7`, exact manifests,
  zero setpoints, and `59.3 Hz` cadence. Shadow/parity SHA-256 values are
  `c50e2397...`/`178af23b...`.
- The sole N218 official attempt used a verified MAVLink command-31000 reset,
  passed Gates 1-3, stayed collision-free, and timed out after 45 seconds at
  official index `3`; finish remained `-1`. It sent 2664 commands at
  `59.343 Hz` with no transport fault. Smoke/attempt/summary SHA-256 values are
  `51752dac...`, `5ce49643...`, and `7f594266...`.
- N218 was causally active but is rejected for an unchanged retry. At Gate 4 it
  first acquired about `31 m forward / 6 m right / 5 m down`, then oscillated
  across the aperture under alternating saturated banks and had already
  overshot before the long dropout. After detections ceased, indefinite motion
  extrapolation ran to an impossible world target near
  `[-137.0, 96.0, 42.4] m` and saturated the observation/action. Prediction
  improves held-out native completion, but unbounded prediction does not solve
  the observed live Gate-4 control/association failure. Do not launch another
  unchanged N218 attempt.
- N219 trained the N214 recurrent tail under the N218 predictor with teacher
  reward retained but intervention explicitly zero. The predictor plus full
  roll/thrust teacher intervention supplied a controllability ceiling of
  `58/128=45.31%`, zero crash. Seventeen checkpoints were screened; exact step
  `32768`, SHA-256
  `20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec`,
  ranked first. It completed `47/128` offset-0 and `33/128` true offset-128
  tails, aggregate `80/256=31.25%`, versus N218's `73/256=28.52%`; both had
  three crashes in the second cohort. The initially duplicated offset-128
  screen was detected and discarded before promotion; the retained JSON hashes
  are `be1120d4...` and `8e50412f...`.
- N219 packaging uses callable
  `scripts/policy_callable_n219_unified_tail.py`, SHA-256 `28cce611...`, and
  requires the predictor. Seventy-five focused tests passed. Passive shadow 001
  was void because a prior moving race yielded only one stationary calibration
  sample and emitted no setpoints. Same-PID UI recovery restored R1. Shadow 002
  passed 192 replay ticks at `19.110 Hz`, maximum action error `1.8508e-7`,
  exact manifests, zero setpoints, and `53.946 Hz` cadence; shadow/parity hashes
  are `2a2967d6...`/`9741dee8...`.
- N219 official attempt 001 used a valid reset but the frozen H12 prefix struck
  Gate-2 object `1001` at official index `1`, impact `4.9878`; N219 had zero
  Gate-4 samples. It is retained only as prefix-failure evidence. The one
  replacement attempt recovered the same PID without relaunch, verified a new
  command-31000 epoch, passed Gates 1-3, and then timed out collision-free at
  Gate 4. It sent 2729 commands at `60.791 Hz` with zero transport fault;
  finish remained `-1`. Smoke/attempt/summary hashes are `3fd4e184...`,
  `30b0d2d2...`, and `19fa3213...`.
- N219's live Gate-4 trace began with the first accepted target near
  `32 m forward / 9 m right / 5 m down`. The policy saturated negative roll,
  crossed to about `4 m forward / 3 m left`, counter-saturated positive roll,
  then reacquired about `16 m forward / 3 m right` and saturated negative roll
  again. Detections ended near `9 m forward / 8 m right`; the unbounded
  predictor then diverged while the policy saturated roll/thrust until timeout.
  Predictor-specific PPO improved native cohorts but did not change the live
  oscillation. N219 is rejected; no unchanged retry is authorized.

## N220-N221 authoritative execution snapshot - 2026-07-18

- The active objective is still a competitive, valid, collision-free official
  VQ1/R1 lap through all six ordered gates. No N220 or N221 run finished; the
  goal is not complete. The exact simulator remains
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe`
  with Windows Python
  `C:\Users\anon\code\pufferlib-drone\.venv-win\Scripts\python.exe`.
  Work reused responsive shipping PID `24476` without relaunch. Normal attempt
  reset remains disarm, 100 ms wait, then learned-target MAVLink
  `COMMAND_LONG 31000`, confirmation `0`, parameters 1-7 all zero; count it
  only after boot rollback greater than 1000 ms plus a fresh race epoch at
  index `0` and finish `-1`.
- N220 added a default-off roll-aware dropout predictor. The live switch uses
  `4.295 m/s^2 per tan(roll)`; native observation sampling is 15 Hz
  (four 60 Hz steps). Nine PPO checkpoints trained under that harder contract
  all scored `0/128` unassisted, as did their parent under the same condition.
  A full teacher remained controllable at `43/128`. Expert-only ridge
  distillation, seven DAgger ridge children, and a recurrent last-layer child
  all scored `0/128`; these learned branches are rejected.
- N220's observation-only tail controller was the first material native
  improvement in this branch. From measured Gate-4 entry distributions it
  completed `90/128=70.31%` development episodes and
  `82/128=64.06%` frozen holdout episodes. It owns roll/thrust only at official
  Gates 4-6, uses N219 for pitch/yaw, and reads only the 32-value deployed
  observation plus a fixed development-course prior. Its callable SHA-256 is
  `dda536d07b2548b50ca3d433a222539898d9e7c147f9d29fc054be1a12cbd746`.
  Passive shadow passed with no setpoints; shadow/parity hashes are
  `3f12f038...` and `e3ef4e19...`.
- N220 official attempt 001 reset correctly, passed Gate 1, then collided while
  approaching Gate 2: official index `1`, collision ID `1002`, impact
  `6.111586`, finish `-1`. Smoke SHA-256 is `432df241...`. A causally
  changed replacement restricted the roll-aware predictor to index `>=3`;
  it again passed only Gate 1, then timed out collision-free at index `1`
  after `45.008 s`, `2716` commands, and `60.418 Hz`. Smoke SHA-256 is
  `409225a0...`. The trace showed a Gate-2 association frozen near
  `25.31 m / 2.41 m / -6.64 m` with zero rate and 115 rejected samples.
  N220 is rejected; do not retry it unchanged.
- N221 removes the brittle learned-prefix/controller-tail handoff. One
  observation-only fixed-course controller owns pitch, roll, and thrust at all
  six gates; N219 supplies yaw only. The deployable pitch loop uses observable
  gate-rate speed, not simulator velocity, with a `4.00 m/s` target. Gates 5-6
  coordinates in this controller remain synthetic development proxies, not
  measured official coordinates. Callable SHA-256 is
  `468eb14d2234d640d644a06c9877759374553905c3116bc47d6e69e7211285c7`.
- An initial `128/128` screen was invalidated before promotion because pitch
  used privileged native velocity. The corrected exact-deployment screen uses
  the same observable speed estimate as N221. It completed
  `126/128=98.4375%` perturbed full-course episodes with zero crashes and
  timeouts; two episodes missed zero-based gate 4 (official Gate 5). The
  fixed-origin run completed in `2463/60=41.05 s`. Dataset path
  `logs/drone_race_full_policy_six_gate_bootstrap/n220_control_aware_predictor/observable_controller/n221_exact_jitter128_speed_4p0.bin`,
  SHA-256 `01c0772a...`. Eight-episode, 19,356-record Python/native controller
  replay has pitch/roll/thrust RMS
  `1.032e-6 / 0.002113 / 2.790e-6`; report SHA-256 `1344e609...`.
  Ninety-five focused tests passed.
- N221 passive live shadow passed at official index `0`: `214` replay ticks,
  `21.300 Hz` inference, maximum action error `6.736e-5`, all manifest
  matches true, and no MAVLink setpoints. The no-op publisher measured
  `55.469 Hz` with zero violations. Shadow/parity SHA-256 values are
  `961d57c3...` and `d6e0bb86...`.
- N221 official attempt 001 used a fresh command-31000 epoch and healthy
  streams, but made zero official passes. It approached Gate 1 to
  `10.0 m forward / 3.266 m right / 1.422 m up`, then the association
  diverged while roll/thrust saturated. The run timed out collision-free at
  index `0`, finish `-1`, after exactly `45.0 s`, `2750` commands, and
  `61.259 Hz`. Smoke/attempt/summary SHA-256 values are `d434eeb2...`,
  `c7014011...`, and `6a10723a...`. N221 is rejected for live deployment;
  do not retry it unchanged.
- The current blocker is specifically live target association, not another
  unconstrained policy-training round. Native dynamics and controller porting
  are sufficient to complete the proxy course, while live camera association
  can accept an alias, reject subsequent correct samples, and propagate a
  false target indefinitely. The next candidate must first demonstrate a
  course-consistent, bounded association/reacquisition mechanism on the N220
  Gate-2 and N221 Gate-1 traces. It must fail closed when consistency is
  unavailable and pass passive shadow/parity before another official command.
## N222-N224 association-clock execution snapshot — 2026-07-18

- The active objective remains one competitive, valid, collision-free official
  VQ1/R1 lap through all six ordered gates. No N222-N224 run finished, so the
  goal is not complete. The exact simulator, Windows Python, responsive reused
  PID `24476`, and disarm/100 ms/one MAVLink `COMMAND_LONG 31000` reset
  contract remain unchanged.
- The N220 Gate-2 trace isolated a real clock-conflation defect in
  `ObservableGateMotionFilter`: every 60 Hz prediction advanced `_time_s`,
  so the innovation horizon collapsed to one inference tick and rejected the
  next physically valid camera pose. A separate last-accepted-camera clock now
  governs association. The failed trace counterfactual changes Gate 2 from a
  frozen `[25.3118,2.4141,-6.6441]` m track to accepting the next valid
  22.3 m observation at 5.055 s and following the gate inward. Focused tests
  cover the last-good horizon, nearer-only optional reacquisition, and farther
  alias rejection.
- N222 enabled the bounded reacquisition mechanism with N221's all-course
  observable controller. It was passive shadow only; no official flight was
  run. Shadow/parity passed with zero setpoints, but live actions were more
  saturated than N221 (roll saturation 63.32%, pitch 98.25%), so N222 was
  rejected before flight. Shadow/parity SHA-256 values are
  `4e2e7818...`/`3ddca8fb...`.
- N223 paired only the corrected association horizon with the proven H12
  Gates-1-through-3 prefix and N220 Gates-4-through-6 tail. Passive shadow
  passed 224 ticks at 22.4 Hz, maximum replay error `1.5786e-7`, exact
  manifests, zero setpoints, and 57.592 Hz no-op cadence. Its one valid official
  attempt passed Gate 1 and continuously tracked Gate 2 instead of freezing,
  then hit Gate-2 object `1001` at impact `3.319516`; it sent 445 commands
  at 60.847 Hz with no transport fault. Shadow/parity/smoke/attempt/summary
  hashes are `2f88cdff...`, `5474f8d4...`, `69227537...`,
  `19c2d7af...`, and `0ccc5e8d...`. No unchanged N223 retry is authorized.
- N223 also exposed that the association horizon had inadvertently been fed
  into the pose/rate filter timestep, changing the already-proven H12 control
  signal. N224 keeps the long last-good horizon only for gate membership and
  restores the historical time-since-propagation timestep for state filtering.
  The N223 counterfactual raises terminal Gate-2 closing-rate estimation from
  about 4.8 to 6.7 m/s while preserving the corrected association. The current
  contract/runner SHA-256 values are `473773f0...`/`487195d7...`; all 66
  focused tests pass.
- N224 reset recovery proved boot rollback and fresh index 0/finish -1 with zero
  collision. Passive shadow passed 215 ticks at 21.466 Hz, maximum replay error
  `1.9576e-7`, exact manifests, zero setpoints, and 58.6 Hz no-op cadence.
  Its one official attempt made the first advancement in this branch: official
  Gates 1 and 2 passed, the run stayed collision-free, and transport delivered
  2735 commands at 60.904 Hz with no rate/dropout/drain fault. It then timed out
  at Gate 3/index 2 after the close track projected roughly 2.1 m outside the
  vertical aperture and detector loss let the predictor diverge to an
  impossible -297 m forward target. Finish remained -1. Reset/shadow/parity/
  smoke/attempt/summary hashes are `ca45bcae...`, `6dc130bf...`,
  `75affd8a...`, `45c7ee33...`, `71a873e2...`, and `1fbe3bbf...`.
- The Gate-2 association bug is closed; do not restart association research,
  retrain another reward-only policy, or repeat N223/N224 unchanged. The next
  admitted change must preserve the now-proven clean Gates-1/2 prefix and
  address only Gate 3's projected vertical overshoot and unbounded
  post-plane prediction. It must show replay/native evidence and passive
  parity before another official attempt.

## N225-N228 phase-gating and terminal-control snapshot — 2026-07-18

- The active objective remains one competitive, valid, collision-free official
  VQ1/R1 lap through all six ordered gates. Best progress in this block is
  three official gates; no finish exists, so the goal is not complete. All
  work reused responsive shipping PID `24476` and the exact documented
  disarm/100 ms/one learned-target MAVLink `COMMAND_LONG 31000` reset contract.
- N225 added a Gate-3-only vertical thrust guard. It changes strong sub-hover
  thrust only when the visible trajectory projects above the vehicle at the
  terminal plane. Offline it changed four samples on N224's miss and was exact
  on five prior Gate-3 passes. Callable SHA-256 is `c2f104c1...`; passive
  shadow/parity passed with hashes `0a486625...`/`5a5bd48d...`. Official
  attempt 001 hit Gate-1 object `1001` at impact `6.404631`; the one replacement
  timed out collision-free at index 0 after its unbounded prefix predictor ran
  the stale target to about -287 m. The guard never activated live. Smoke/
  attempt/summary hashes are `928bba5d...`/`d15ac519...`/`265c7a33...` and
  `6fe12fad...`/`bdf1050b...`/`7787f9dc...`. N225 is stopped.
- N226 introduced an explicit phase switch for dropout propagation. It restores
  the historically live-proven non-predictive H12 contract at official indices
  0-2, then enables constant-velocity and roll-aware prediction at index 3.
  The valid shadow passed 299 visible replay samples at 29.9 Hz with maximum
  error `2.3537e-7`, zero setpoints, and 59.886 Hz passive cadence; reset/
  shadow/parity hashes are `c161b10d...`/`aa9ec640...`/`6e4019e7...`.
  Its official attempt used reset `68356 -> 3384 ms`, passed Gates 1-3
  collision-free, and sent 2697 commands at 60.078 Hz. At Gate 4 the predictor
  became active as configured, but the last association propagated to about
  `[-238.9,-0.5,130.5] m`; the run timed out at index 3. Smoke/attempt/summary
  hashes are `38fd8473...`/`3332a9a7...`/`41f6d202...`. This is the first
  three-gate advance in the N220-N228 branch and proves phase gating, not a lap.
- N227 kept N226's proven prefix and replaced only Gate-4 roll/thrust with a
  coordinate-free terminal-plane PD law. Callable SHA-256 is `f0fd8ed0...`;
  77 focused tests and passive shadow/parity passed, with shadow hashes
  `4631d112...`/`f37d3bf9...`. The official run again passed Gates 1-3, but
  missed Gate 4 near an associated `2.0 m right / 4.1 m down` plane estimate
  and struck object `1002` at impact `2.400161`. Smoke/attempt/summary hashes
  are `8c33b715...`/`ec73affa...`/`dc1b33a2...`. N227 is rejected.
- N228 retained N227 lateral control and changed only Gate-4 vertical PD plus a
  bounded post-plane response. Callable SHA-256 is `3531c3be...`; 83 focused
  tests and passive shadow/parity passed, hashes `ff44aacc...`/`e4dd0682...`.
  Both authorized official attempts failed in the unchanged prefix before any
  Gate-4 sample: Gate-3 object `1001` impacts `1.929871` and `0.319780`, each
  at official index 2. Their smoke/attempt/summary hashes are
  `1488266f...`/`472340f5...`/`447707e1...` and
  `c2c2b58f...`/`6ab53499...`/`0f9d1d47...`. N228 is stopped and its target
  mechanism has no live evidence.
- Current source hashes are motion contract `473773f0...`, smoke runner
  `acaeadbb...`, and N209-tail deployment runner `7682ac1d...`. Do not repeat
  N225-N228, add another scalar PD child, or claim that three gates is a lap.
  The next admitted mechanism must be structurally distinct: reuse the existing
  learned phase-reset Gate-4 tail/terminal pulse under the corrected association
  clock and N226 phase-gated observation contract, with passive parity before
  aggregate official evidence.

## N229 learned-tail batch snapshot — 2026-07-18

- N229 implemented that structurally distinct candidate. Variant `PP` keeps the
  H12/N203 learned prefix non-predictive through official index 2, then enables
  constant-velocity dropout propagation at index 3 and dispatches the existing
  learned Gate-4/5/6 checkpoints plus the N206 learned Gate-4 terminal crossing
  pulse. The hybrid runner SHA-256 is `49f39699...`; the batch runner SHA-256 is
  `21e26040...`. Eighty focused tests and the Windows PowerShell parser passed.
- Preflight reset was clean at official index 0 with no collision. Passive
  shadow/parity passed: 274/274 visible samples, maximum action error
  `2.2516e-7`, 27.315 Hz inference, 60.341 Hz passive cadence, zero MAVLink
  setpoints, and exact deployment manifests. Reset/shadow/parity SHA-256 values
  are `d2fd82ca...`/`35bc7889...`/`cd782ccb...`.
- The three-run official batch produced post-stop official gate counts `3, 1,
  2`, median 2 and maximum 3, with zero finishes and zero collision-free runs.
  Run 001 hit Gate-3 object `1001` at impact `1.640297` after 627 commands at
  60.065 Hz. Run 002 hit Gate-2 object `1001` at impact `0.753877` and also fell
  below the 50 Hz transport floor at 48.389 Hz. Run 003 hit Gate-3 object `1001`
  at impact `10.596446` after 656 commands at 61.289 Hz. All failures occurred
  before official index 3, so the learned tail never activated and has no live
  target-mechanism evidence. Batch manifest/score SHA-256 values are
  `150f6d6d...`/`f81bc686...`.
- N229 is rejected as a six-gate deployment and must not be repeated unchanged.
  Best verified progress remains three of six official gates; no valid lap has
  been produced. The live blocker is the high-variance Gates-1-through-3 prefix.
  Before any further Gate-4 work or live run, the next candidate must improve
  prefix collision-free reliability against the accumulated passing and failing
  traces and preserve the corrected association and phase-gating contracts.

## N230-N231 fixed-clock and MAVLink discovery snapshot — 2026-07-19

- The active objective remains one competitive, valid, collision-free official
  VQ1/R1 lap through all six ordered gates. No N230/N231 run finished; best
  verified progress remains three of six. The exact v3385 simulator and reused
  responsive shipping PID `24476` remain authoritative.
- N230 added an opt-in 60 Hz local-monotonic recurrent-state clock independent
  of camera/inference arrival and the fixed-rate MAVLink publisher. The latest
  observation and previous-action slots are reused for catch-up ticks; no
  future observation is invented. Core smoke/batch-runner SHA-256 values are
  `e6f2d717...`/`70e78ee4...`. Focused Linux and actual-Windows tests passed,
  and passive shadow/parity proved exactly 600 recurrent ticks in 10 s,
  600/600 replay, maximum error `2.210226e-7`, 63.495 Hz no-op cadence, zero
  setpoints, and exact manifests. Shadow/parity hashes are
  `219431c9...`/`351abe78...`.
- The preregistered N230 `PF` batch ran three 45 s attempts at command request
  80 Hz / recurrent state 60 Hz. Official gate counts were `2,3,3` (median
  three), with two collision-free Gate-4 timeouts and zero finishes. Run 1 hit
  Gate-3 object `1001`, impact `10.403496`; runs 2/3 delivered 2700 recurrent
  ticks at exactly 60 Hz and 63.643/63.555 Hz wire cadence with no transport
  fault. The Gate-4 predictor then extrapolated last valid targets near
  `[18.226,9.436,-7.015]` and `[11.826,6.040,-5.104]` m to impossible final
  targets near `[-224,19,95]` and `[-264,0,106]` m, saturating the policy.
  Manifest/score hashes are `580a2681...`/`4e3affbc...`. N230 proves the
  clock mechanism and rejects unbounded Gate-4 prediction.
- N231 retained the fixed 60 Hz clock and removed prediction only for the
  learned Gate-4 terminal tail, using current raw/partial-frame pose. Linux
  passed 110 focused tests; actual Windows passed 70 plus launcher parses.
  Passive shadow/parity again proved exactly 600 ticks at 60 Hz, maximum error
  `2.3588e-7`, 63.401 Hz no-op cadence, zero setpoints, and exact manifests;
  hashes are `5e93180b...`/`9111ca12...`.
- The preregistered N231 `RF` batch produced official counts `1,3,3`, median
  three, one collision-free run, and zero finishes. Run 1 timed out at Gate 2.
  Runs 2/3 reached Gate 4 but the learned tail immediately commanded aggressive
  forward pitch, negative bank, and minimum thrust from entries near
  `[32,3.8,4.2]` and `[32,4.1,4.7]` m; each struck Gate-4 object `1002`
  before entering the terminal-pulse envelope. Impacts were `0.015273` and
  `6.071695`. Manifest/score hashes are
  `2cb79c15...`/`11686157...`. N231 rejects the raw/current-frame tail.
- Together N230/N231 close all three simple Gate-4 observation substitutions:
  indefinite filtered hold, unbounded constant-velocity prediction, and raw
  current-frame control. Fixed recurrence improves timing determinism but does
  not solve Gate 4; another scalar PD wrapper or unchanged tail is not admitted.
- A complete MAVLink site rescan and compliant active-state probe are recorded
  in `docs/mavlink_site_context_2026-07-18.md`. Command `31000` is the standard
  user-defined enum slot `MAV_CMD_WAYPOINT_USER_1`; its reset meaning remains
  simulator-specific. The active VQ1/R1 probe sent zero reset/arm/disarm/
  setpoint/persistent-rate/position requests and received no ACK, interval,
  capability, one-shot `ATTITUDE`, TIMESYNC, or PING response. Probe SHA-256 is
  `f9f3bc4b...`. Standard discovery cannot provide the missing Gate-4 state.
- The next admitted mechanism is a bounded, uncertainty-aware receding-horizon
  visual optimizer, not another reactive tail: fit compact relative-pose and
  attitude dynamics from accumulated official traces, optimize short body-rate/
  thrust sequences in camera-relative coordinates, execute only the first
  action, and replan on every accepted frame. Prediction must terminate at a
  plane/time/uncertainty bound and fail closed. Before flight it must turn the
  recorded N230/N231 Gate-4 entries into centered terminal states in replay,
  preserve the source-hashed Gates-1-through-3 prefix, and pass passive Windows
  shadow/parity.
## N232 bounded visual MPC admission and preregistration — 2026-07-19

- The active objective remains one competitive, valid, collision-free official
  VQ1/R1 lap through all six ordered gates. N232 is admitted for a bounded
  three-attempt batch; it has not yet flown and no finish is claimed.
- N232 fits only current raw aperture poses and observable attitude/rates from
  `118` official reports (`276` segments, `3,672` raw samples). The constrained
  five-fold model has nonnegative thrust/drag terms and no fitted forward bias.
  Model SHA-256 is
  `9aaefaae111734314353757adbd49da913f13c9bb558627e382ba3c2eb8cc275`;
  fitter/controller/evaluator hashes are `69bf3004...`/`3f0fe8f1...`/
  `d3d37212...`.
- The Gate-4 controller carries only observable Gate-3 relative velocity,
  rejects incoherent/far aliases, anchors an inertial frame on the first
  coherent raw Gate-4 pose, optimizes bounded pitch/roll/thrust knots against
  the leave-one-fold-out model ensemble, executes the first action, and stops
  prediction at the gate plane, `2.0 s`, `0.30 s` measurement age, or `0.80 m`
  uncertainty. It changes only official index `3`; unit evidence preserves
  parent outputs exactly at indices `0-2` and `4-5`.
- Deployment uses `32` candidates, two refinements, a deterministic `10 Hz`
  logical clock, one recurrent update per completed live update
  (`PolicyStateHz=0`), and fresh-pose replanning no faster than `0.18 s` with a
  `0.25 s` hard maximum. This supersedes N230/N231's infeasible wall-clock
  catch-up: the exact Windows parent benchmark sustains only about `18.5 Hz`;
  optimized/cached N232 calls measured about `172/52 ms`.
- The authoritative `0.20 s` held-action leave-one-fold-out counterfactual
  passes all `20/20` N230/N231 Gate-4 entry/member rollouts. Worst absolute
  terminal offsets are `0.478516 m` right and `0.296263 m` down; maximum
  crossing time is `12.2 s`. Evidence SHA-256 is
  `ee352fd072a5cb49f778ba3908375d805e1c513db2ba9cbbdffc17564543e6a0`.
- A latest-frame transport correction now lets the bounded UDP socket discard
  obsolete retransmitted camera chunks during the `15 Hz` cooldown instead of
  starving the control loop. Camera receiver SHA-256 is
  `bac1c5d3fc5c0b8b509ffe81f6f49af19a6c1ecb3fe9401172c3f420b169dfeb`.
  Linux and actual Windows each pass `70/70` focused tests; both PowerShell
  launchers parse.
- Passive `n232_visual_mpc_shadow_004` passed for `10.047 s`: `129` live and
  replay ticks at `12.8397 Hz`, maximum action error `2.09942e-7`, `86` camera
  frames/detections, no telemetry dropout/drain-limit hit, exact manifests, and
  `59.017 Hz` no-op cadence. It sent zero reset, arm, disarm, or MAVLink
  setpoints and left official index `0`, finish `-1`, on responsive PID
  `24476`. Shadow/parity hashes are `ed32bc52...`/`a0bd5afc...`.
- Preregister exactly variant `MO`, tag
  `competitive_n232_visual_mpc_batch_001`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `0`, target six gates, and stop on the
  first valid finish. Every attempt must reuse the responsive simulator and
  prove the normal disarm/`100 ms`/one learned-target MAVLink
  `COMMAND_LONG 31000` reset rollback, fresh index `0`, and finish `-1`.
  Dry manifest SHA-256 is
  `3a991bb73ba057b49b87d3e3720cbbd1ff0c4df4a443ff59883cb7d537d89b09`.
## N232 official batch result and N233 admission — 2026-07-19

- The preregistered N232 `MO` batch is complete and rejected. Official gate
  counts were `1,2,2` (median `2`, maximum `2`), with zero collision-free
  attempts and zero finishes. All failures were in the unchanged prefix before
  official index `3`, so the Gate-4 optimizer never activated and still has no
  live target-mechanism evidence.
- Run 1 hit Gate-2 object `1001` at `7.375 s`, impact `9.523264`, after
  `463` commands at `63.724 Hz`. Runs 2/3 passed Gates 1-2 then hit Gate-3
  object `1001` at `10.375/10.625 s`, impacts `10.404356/8.742578`, after
  `653/672` commands at `63.709/63.905 Hz`. All three had zero command-rate
  violation, telemetry dropout, or drain-limit hit.
- Final manifest/score SHA-256 values are
  `0fdee072f729c0252b1548e56d5adbae2ac2ecc66c4ec3f285ee3c21f3658d26`/
  `88c18407401074d851607ddafcccbf99a80330a35392c3d90408be130cd10e6e`.
  Do not repeat N232 unchanged.
- The causal regression is the `PolicyStateHz=0` prefix compromise, not the
  measured recurrent network cost. A direct exact-Windows benchmark runs the
  deployed FP32 recurrent checkpoint at about `6,320 Hz`; N232 instead paid
  about `50 ms` per call because it resolved, read, and SHA-256 hashed the
  dynamics JSON on every inference, including Gates 1-3.
- N233 is admitted as the smallest corrective mechanism: launcher/hash-verify
  the same model once, cache that immutable verified model in the callable,
  restore the N230/N231 `60 Hz` recurrent prefix clock, and retain N232's same
  raw-pose tracker, bounded `32 x 2` optimizer, uncertainty/plane/time stops,
  and approximately `0.20 s` held-action cadence only while Gate 4 is active.
  It must pass source-hash tests, exact prefix parity, an actual-Windows timing
  benchmark, and passive live shadow before any bounded official batch.

## N233 fixed-rate visual MPC preregistration — 2026-07-19

- N233 passed every admission gate. Its callable SHA-256 is
  `4a3f0c7a0c8b73053f6aa7da8995b4fffe1e2c8ed2f430c79135476ca8f1c4ba`;
  the pinned hybrid/batch launcher hashes are `6fc426e0...`/`2af447bb...`.
  Linux and actual Windows each pass `70/70` focused tests, both launchers
  parse, and `git diff --check` passes.
- On actual Windows, 600 Gate-1 prefix decisions took `0.24733 s`
  (`2,425.9 Hz`). The first Gate-4 optimized decision took `0.13724 s`, the
  cached following decision took `0.00994 s`, and model validation remained
  exactly once. This is safely inside the fixed `60 Hz` prefix and roughly
  `5 Hz` bounded Gate-4 replan contract.
- The existing responsive simulator PID `24476` was reused without relaunch.
  Same-process UI recovery cleared the prior terminal `base_mode=65`,
  `system_status=3` state and restored VQ1/R1 at official index `0`, finish
  `-1`, with `base_mode=193`, `system_status=4`.
- Passive tag `n233_visual_mpc_fixed_shadow_005` passed: exactly `600` live
  and replayed decisions at `60 Hz`, maximum action error `2.340714e-7`, `142`
  completed camera frames (`126` detector frames), zero telemetry dropout or
  drain-limit hit, zero MAVLink setpoints, and zero reset/arm/disarm commands.
  Shadow/parity hashes are `d781b6f4...`/`1c3d3812...`.
- Preregister exactly variant `MC`, tag
  `competitive_n233_visual_mpc_fixed_batch_002`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `60 Hz`, target six official gates,
  and stop on the first valid finish. Each attempt must reuse the responsive
  simulator and preserve reset rollback, fresh index `0`/finish `-1`, health,
  collision, command-rate, and command-free post-stop evidence. Dry manifest
  SHA-256 is
  `d52519417daf7bcd0a2ebf6faf7ed053d4d4a426353ede27855e1f3a2779c641`.
+

## N233 official batch result and N234 admission — 2026-07-19

- The preregistered N233 `MC` batch is complete and rejected unchanged.
  Official gate counts were `2,3,3` (median/maximum `3`), with zero finishes
  and zero collision-free attempts. This improves N232's `1,2,2`, and runs
  2/3 activated Gate 4, so the target mechanism now has live evidence.
- Run 1 collided at official index `2` after `10.719 s` with object `1001`,
  impact `9.212901`, and `662` commands at `61.938 Hz`. Runs 2/3 reached
  official index `3`, then collided with object `1002` after
  `16.015/36.750 s`, impacts `2.621719/6.141685`, and `1013/2333`
  commands at `63.436/63.563 Hz`. Every run had zero command-rate violation,
  telemetry dropout, or drain-limit hit and proved the normal single-reset
  boot rollback/fresh-start sequence.
- Final manifest/score SHA-256 values are
  `3f577a381e1a5006d3c6f2c198349c95e5f582180913939ac5dd7e4cdc95bfa2`/
  `7c20ccb5297ad39cc5314a991c39ad47a2dcdf0a7c7f15182d72e9cc0c6ca114`.
  Do not repeat N233 unchanged.
- The new causal evidence is a Gate-4 track-identity/staleness failure. Run 2's
  observable target jumped from `[68.57,1.07,1.29] m` at `10.4 s` to
  `[33.10,6.00,2.12] m` at `10.9 s`, later reached
  `[3.39,-1.07,-0.36] m` at `15.2 s`, then jumped to
  `[19.59,-4.04,-8.23] m` one sample later without an official advance.
  Run 3 lost all target poses after `16.7 s` but held the stale saturated
  action `[1.0,-0.9,0.6523,-0.0322]` for about `20 s` until collision.
- Admit only an N234 stale-track/reacquisition correction next: do not alter the
  proven 60 Hz prefix or bounded optimizer. Expire aggressive cached actions
  after the 0.30 s observable age limit, reset the Gate-4 tracker after a
  bounded rejection/staleness interval, require coherent fresh samples before
  reacquisition, and use a neutral bounded brake/search action while no track
  is admitted. First capture the callable controller snapshot and replay the
  N233 Gate-4 traces offline; require that no stale saturated action can persist
  and that Gates 1-3 remain bit-exact before any new official run.
+

## N234 stale-track reacquisition preregistration — 2026-07-19

- Fixed-clock replay of the two live N233 Gate-4 traces confirmed the cause.
  N233 produced only `2` accepted optimizer plans in each trace but
  `56/121` failsafe plans; final reasons were `measurement_stale`, with
  saturated pitch/roll commands. Replay artifacts for runs 2/3 have hashes
  `ad0ed381...`/`b97fa758...`.
- N234 preserves N233's exact parent, 60 Hz clock, verified dynamics ensemble,
  optimizer, and Gate-3 velocity carry. It changes only observable association
  failure handling: accept initial targets out to `75 m`, require three fresh
  coherent poses, reset after `0.30 s` staleness, and command bounded
  `[0.25,0,0,current_yaw]` brake/hover whenever acquisition or planning is
  rejected. Callable SHA-256 is
  `84c08dac68be66039567173a7319831232601c22790792588a044401d7671530`.
- Counterfactual replay on N233 runs 2/3 performed `3/6` track resets,
  admitted `5/4` optimized plans, and ended with neutral cached actions
  instead of stale saturation. The run-3 maximum recorded plan-event magnitude
  fell below `0.934`; replay hashes are `60427632...`/`a7b3fbd3...`.
  The relevant suite passes `88/88` on Linux and actual Windows, both
  PowerShell launchers parse, and `git diff --check` passes.
- PID `24476` was again reused without relaunch and VQ1/R1 recovered to index
  `0`, finish `-1`, `base_mode=193`, `system_status=4`. Passive tag
  `n234_reacquiring_mpc_shadow_006` passed with exactly `600` live/replayed
  decisions at `60 Hz`, maximum action error `1.711920e-7`, `116`
  detector frames, zero telemetry dropout/drain-limit hit, zero setpoints, and
  zero reset/arm/disarm commands. Shadow/parity hashes are
  `94c006da...`/`91ebc9d8...`.
- Preregister exactly variant `MR`, tag
  `competitive_n234_reacquiring_mpc_batch_003`, three attempts, `45 s`
  each, command request `80 Hz`, policy state `60 Hz`, target six official
  gates, and stop on the first valid finish. Preserve the same reset, health,
  collision, cadence, and command-free post-stop proofs. Dry manifest SHA-256
  is `31d3fe386b257764def577503e05ac1f1186845cef6de50a3fc1c3d5772eefdd`.
+

## N234 official result and N235 admission — 2026-07-19

- The preregistered N234 `MR` batch is complete and rejected unchanged.
  Official counts were `1,3,2` (median `2`, maximum `3`), with zero
  finishes and zero collision-free attempts. Runs 1/3 failed in the unchanged
  prefix at `7.313/10.578 s`; run 2 alone reached Gate 4 and failed after
  `42.281 s`.
- Runs 1/3 hit objects `1001/1001` with impacts `2.375203/10.908347`.
  Run 2 hit object `1002` at official index `3`, impact `10.075733`.
  Effective command rates were `62.069/63.479/63.147 Hz`; every run had zero
  command-rate violation, telemetry dropout, or drain-limit hit and a proven
  normal reset rollback/fresh start. Final manifest/score SHA-256 values are
  `18b3b49fcc5ae3f86d1643f4a5aa026e3874c7b25e494461c775e3ea5c939e35`/
  `5d22e174e69a65945092ea0eed9a47d19f8d0121d8386cfd832adb6157c81c02`.
- N234 fixed N233's stale-saturation defect live, but exposed the next cause.
  In run 2 it entered Gate 4 at `10.3 s`; its only non-neutral interval was
  approximately `13.3-13.6 s`. The visible target then moved to
  `[16.0,13.1,9.8] m` with yaw error `+0.686 rad` by `17.2 s`, while
  neutral brake/hover did not steer toward it. All poses disappeared after
  `17.2 s`, and the controller correctly stayed at bounded
  `[0.25,0,0,current_yaw]` rather than saturating until the late collision.
- Fixed-clock replay of that live trace reports only `1` accepted optimizer
  plan, `18` rejected plans (`14` state-uncertain, `4`
  prediction-uncertain), `4` track resets, and `1704` acquisition-wait
  calls. Replay SHA-256 is `8eb06a63...`.
- Admit only N235 observable acquisition/search fallback next. Preserve N234's
  prefix, optimizer, coherent admission, stale reset, and neutral safety
  bounds, but when a raw current Gate-4 pose exists, use a bounded low-gain
  pitch/roll/thrust/yaw acquisition servo to keep the target centered; when it
  leaves frame, carry/search in the last observed yaw direction for a bounded
  interval. No map coordinates or privileged simulator state. Prove bounded
  offline replay, exact non-Gate-4 invariance, Windows/Linux tests, and passive
  shadow before any flight.
+

## N235 observable acquisition/search preregistration — 2026-07-19

- N235 keeps N234's exact verified parent/model, 60 Hz clock, coherent tracker,
  stale reset, and accepted optimizer path. Its only new behavior is a bounded
  no-plan acquisition action: far/medium/near pitch `-0.12/0/+0.25`, raw
  lateral gain `0.03/m` capped at `0.55`, raw vertical gain `0.04/m`
  capped at `0.50`, yaw step capped at `0.25 rad`, and a `0.10 rad`
  last-direction yaw search for at most `2.0 s` before brake/hover. Callable
  SHA-256 is
  `dd8cd8d89f070c9ca51820c18a9565bb61eb478b1dcd9cce464d4fd4674c569a`.
- Counterfactual replay of N234's live Gate-4 trace produced `221` visible
  acquisition calls, `104` bounded blind-search calls, then `1397` blind
  brake calls; maximum recorded plan-event magnitude remained `0.9`.
  Replay SHA-256 is `2c3392a7...`. The relevant suite passes `92/92` on
  Linux and actual Windows, both launchers parse, and `git diff --check`
  passes.
- PID `24476` was recovered and reused without relaunch at VQ1/R1 index `0`.
  Passive tag `n235_search_mpc_shadow_007` passed with exactly `600`
  live/replayed decisions at `60 Hz`, maximum action error `1.992399e-7`,
  `127` detector frames, zero telemetry dropout/drain-limit hit, zero
  setpoints, and zero reset/arm/disarm commands. Shadow/parity hashes are
  `dffebee3...`/`b26ffe9e...`.
- Preregister exactly variant `MS`, tag
  `competitive_n235_search_mpc_batch_004`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `60 Hz`, target six official gates,
  and stop on the first valid finish. Preserve all reset, health, collision,
  cadence, and command-free post-stop proofs. Dry manifest SHA-256 is
  `f61c389f7a0950ecfa1bd3a35b97ed52d34dc5d7d0f3e10834a1e82179117e2b`.
## N235 official result and N236 admission — 2026-07-19

- The preregistered N235 `MS` batch is complete and rejected unchanged.
  Official counts were `2,0,3` (median `2`, maximum `3`), with zero
  finishes. Run 1 collided with object `1001` at `7.171 s`, impact
  `0.211197`; run 2 remained collision-free but passed no gate in `45 s`;
  run 3 reached Gate 4 then hit object `1002` at `14.906 s`, impact
  `6.979586`. All command rates were `63.5-63.9 Hz` with zero command-rate
  violation, telemetry dropout, or drain-limit hit and valid reset proofs.
- Final manifest/score SHA-256 values are
  `c1096ed17d849a103116a2d99784ead97cdc480f377ab9f004419eef4659b05d`/
  `0e079b44075beeeffda856b13bd24f9bf5426721a5aa8c76a978a17bd53b56ee`.
  Do not repeat N235 unchanged.
- N235 succeeded at its intended acquisition mechanism: the Gate-4 target
  remained visible in `38/44` retained samples. It failed because acquisition
  pitch was forward before centering: at `10.6 s` the target was
  `[40.0,4.75,6.38] m`; at `12.7 s` it was still
  `[23.41,4.79,7.32] m`, yet pitch remained `-0.12`, and collision followed
  at `14.9 s`.
- Admit only N236 staged acquisition next: preserve all N235 lateral, vertical,
  yaw, search, tracker, and optimizer behavior, but command brake pitch while
  raw right/down/yaw errors exceed the existing center envelope. Permit gentle
  forward pitch only after centering; blind search also brakes. Validate this
  exact causal change offline and through the normal test/shadow gates.

## N236 staged Gate-4 preregistration — 2026-07-19

- N236 preserves N235's exact observable tracker, lateral/vertical/yaw search,
  optimizer, and fixed `60 Hz` policy clock. Its only causal change is staged
  pitch: brake at `+0.25` while the raw target lies outside the existing
  `1.5 m` right/down and `0.12 rad` yaw center envelope; allow N235's gentle
  forward pitch only after centering; and brake during blind yaw search.
  Callable SHA-256 is
  `fe7984c17b203ade4b3967ff0e6166be17c4d9a89a827231e2e34be6e28862f4`.
- Counterfactual replay over N235's actual Gate-4 collision trace produced
  `63` visible uncentered-brake decisions, `35` blind-search brake decisions,
  zero centered-forward decisions, and ended in brake rather than repeating
  the collision-producing forward command. Replay SHA-256 is
  `38760787cc32ca88f4f0f73996550a7fbd1debfa1be6d3756d6cfc3917186f52`.
- The relevant suite passes `96/96` on Linux and the actual Windows runtime;
  both PowerShell launchers parse. Responsive PID `24476` was recovered and
  reused without relaunch at VQ1/R1 index `0`. Passive tag
  `n236_staged_mpc_shadow_008` passed with exactly `600` live/replayed
  decisions at `60 Hz`, maximum action error `2.555741e-7`, `122` detector
  frames, zero telemetry dropout/drain-limit hit, zero setpoints, and zero
  reset/arm/disarm commands. Shadow/parity SHA-256 values are
  `310bf2b3e8c68a6e8eb6aa35b027c9a9c70a4fc1730c4c5616cfb4a296bdb73f`/
  `2a023bfecadb565c1d1cbd584f1d657950b2971905abaa4ca69204cdd7d60226`.
- Preregister exactly variant `MB`, tag
  `competitive_n236_staged_mpc_batch_005`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `60 Hz`, target six official gates,
  and stop on the first valid finish. Preserve all reset, health, collision,
  cadence, and command-free post-stop proofs. Dry manifest SHA-256 is
  `3c818c036e240e60bf2f61d09cea24fbd297339ac951b59940828c34698a0105`.

## N236 official result — rejected — 2026-07-19

- The preregistered N236 `MB` batch completed unchanged. Every attempt passed
  official Gates 1-3, but none passed Gate 4 or finished: counts were `3,3,3`.
  Run 1 was collision-free for `45 s`; run 2 hit object `1002` at
  `17.75 s`, threat `1`, impact `0.010981`; run 3 hit object `1002` at
  `17.187 s`, threat `2`, impact `10.552031`. Effective command rates were
  `61.930/63.549/63.302 Hz`, with zero command-rate violation, telemetry
  dropout, or drain-limit hit. Final manifest/score SHA-256 values are
  `ff595221184c7f710a1c24f0831023738946a9d639a94810d6f52e513bff49f4`/
  `50595bb65f3970d1ef5c32786f66d273b3236660231f1da22b16efbf811ab0fe`.
- N236 causally removed N235's immediate uncentered forward drive, but the
  target mechanism remains rejected. Gate-4 entries repeatedly appeared near
  `30-38 m` forward, `4-7 m` right, and `5-7 m` down. The staged controller
  braked forward, yet raw detections jumped between inconsistent range/bearing
  families; lateral/vertical errors grew or reversed, only `1-4` optimized
  plans were admitted per replay, and run 1 lost the target after `19.1 s`.
  Live replay SHA-256 values for runs 1-3 are `a25bd2b...`, `cfa621b7...`,
  and `1d944348...`. Do not repeat N236 unchanged.
- Standard MAVLink discovery is closed as a source of the missing state. The
  next mechanism must use legal camera/IMU/action-history observations and fix
  visual target identity/bearing consistency before optimizing control. Do not
  reopen persistent message requests, hidden-position assumptions, stale-track
  carry, or another renamed scalar Gate-4 PD wrapper.
- After the batch, responsive PID `24476` was recovered through the UI without
  relaunch. Active VQ1/R1 proof is base/status `193/4`, official index `0`,
  finish `-1`; recovery SHA-256 is
  `56d34547a470ed34157c030e958c1d4b90e3d52bdc094b10fef1d08ee3080965`.

## N237 action-observer Gate-4 admission and preregistration — 2026-07-19

- N236 failed because range-family jumps corrupted camera finite-difference
  velocity after the Gate-3 handoff. The first N237 draft retained that
  velocity estimate and was rejected offline at `0/15`; it was never flown.
  The admitted N237 instead carries the reliable Gate-3 entry velocity through
  the fitted ensemble-mean translational plant using the actual action history
  and observed attitude. Fresh camera observations correct position but never
  replace velocity with raw range differences. It requires three coherent
  fresh frames, caps initial range at `40 m`, limits observer steps to `0.10 s`
  and uncertainty to `2.0 m`, and uses projected-intercept actions only while
  the observations remain fresh and coherent. All stale, missing, incoherent,
  or unapproved cases retain N236's staged brake/search and hard `0.30 s`
  reset. No hidden simulator state is used.
- Admitted N237 source SHA-256 is
  `6fc71223f11246bc90d0c70df41cf67e5bc3e721ae83e6b66c05fcfe1720cb5d`.
  The actual Windows deployment suite passed `178/178` tests in `41.62 s`,
  both PowerShell launchers parsed, and `git diff --check` passed. All three
  N236 traces passed all five held-out causal folds (`15/15` total); worst
  projected absolute right/down error was `0.04378/0.02395 m` and maximum
  crossing time was `9.3 s`. Per-trace evaluator artifact SHA-256 values are
  `9a0b7785b6452389db2c6bb657d01f19e8c53c0b07f3ad04d51b0ae6a33874c7`,
  `d5e3b62ec90ca726c759a486ec52befbad165c554822da51d4b8d063c4f8f8b5`,
  and `340c1fa44d231eca0c8140854b92be88cfa99acd2714b787ddce61cf2397eaba`.
  Exact controller replays are
  `63b19c3eb3e0074e1412f30df3b42667fcb036aec3a4adfdd7f82b8df6e1984c`,
  `6d7817788e770608f7861140bced52d8264d0e1f27100d567e9c478438d5054e`,
  and `c22836f1a0230bf89c96eb15e9fb9b2ac73081bf8612da81f28350dce757bdd0`.
- Mandatory live passive shadow passed: `600/600` fixed `60 Hz` samples,
  maximum replay action error `2.2779426575580963e-07`, `127` detector
  frames, zero setpoints/reset/arm/disarm commands, zero collision/dropout/
  drain-limit hit, and no-op cadence `63.902 Hz`. The existing simulator
  remained active VQ1/R1 at base/status `193/4`, official index `0`, finish
  `-1`. Shadow/parity SHA-256 values are
  `9fcc4fdaf466d521b92bb37cea8ec3c381e2d436310fcf12177e0ffc73b881bb`/
  `baf9367ebd902358a2e887b52e28aeae4af8b756ca6a6ff001961f4d93920eec`.
- Preregister exactly variant `MI`, tag
  `competitive_n237_action_observer_batch_006`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `60 Hz`, target six official gates,
  and stop on the first valid finish. Use the already-running responsive
  `FlightSim.exe` PID `24476`; never relaunch it. Preserve the documented
  normal disarm/`100 ms`/exactly-one command-`31000` reset and all health,
  collision, cadence, fresh-start, and command-free post-stop proofs. Dry
  manifest SHA-256 is
  `bf6dc909e1f6e0eb5824f2a609f74ce9c5111d98cf4b991908afd40dd7062ddf`.

## N237 official result — rejected — 2026-07-19

- The preregistered N237 `MI` batch completed unchanged. All three attempts
  passed official Gates 1-3, stayed collision-free for the full `45 s`, and
  failed Gate 4: counts `3,3,3`, finish `-1` throughout. Effective command
  rates were `63.621/63.354/63.622 Hz`, with zero rate violation, telemetry
  dropout, or drain-limit hit. Every attempt proved a fresh boot rollback and
  the required reset lifecycle. Final manifest/score SHA-256 values are
  `183cc8fb7c7f413e5ec6058e1a0f8e1cc57ad1694b02ed3df3d4124e187f2442`/
  `114a6be0681cb455344a5cec9bf072f409e7e90cd7163008c6aa80700a74693f`.
- N237 is rejected live despite the offline counterfactual. Raw closest
  Gate-4 poses were only `3.77/2.42/2.74 m`, `9.60/6.53/4.02 m`, and
  `10.05/7.94/4.30 m` forward/right/down before the apparent target moved
  away. Replay admitted `28/13/16` optimized plans but invoked the new
  rejected-plan projected-intercept branch `0/0/0` times and reset its track
  `3/4/4` times. Each hard reset admitted a discontinuous range/bearing family
  (for example run 1 jumped from about `12 m` back to `24 m`; runs 2/3
  repeatedly jumped between `10-35 m`). The inherited no-plan acquisition
  also continued steering from raw rejected poses. This is the exact target-
  identity defect N236 exposed; N237 corrected velocity propagation but did
  not correct association.
- The observer/model is also not admissible unchanged: its replayed forward
  velocity ended at `-0.56/-3.91/-6.29 m/s` while associated/raw detections
  remained `10-24 m` forward, producing large family-dependent lateral and
  vertical actions. Replay artifact SHA-256 values are
  `7e1faf798ae579a8d2615b625e3b4a2ab17a95a2b5d09c001172922d9dce7450`,
  `3ebfe4db9f7dfb9dabe7cc86c26500904a80d7c5b8dc023fdb68951e8340df13`,
  and `da3fb3f198b9833004673e8bb6a4b0f7865780f10c52838752021afa0585c386`.
  Do not repeat N237 or tune its fitted dynamics.
- The next admitted mechanism must own one persistent observable Gate-4
  target family across detector gaps, reject farther/discontinuous aliases,
  and ensure rejected raw poses cannot affect pitch, roll, thrust, or yaw.
  Association failure must produce bounded level/brake behavior; only a
  coherent associated target may drive an intercept. Preserve the exact
  official-proven Gates-1-3 prefix, learned Gates-5-6 tail, reset contract,
  and existing simulator process.

## N238 persistent identity-lock admission and preregistration — 2026-07-19

- N238 replaces N237's complete Gate-4 tracker/MPC/raw-fallback path while
  preserving the exact parent actions outside official Gate 4. It acquires one
  camera-relative family only after three coherent fresh poses no farther than
  `40 m`. A current pose is accepted only inside the bounded continuity and
  `2.5 m` forward-regression envelope. A discontinuous family can re-enter
  only after three coherent samples and at least `3.0 m` closer progress;
  persistent farther families never re-anchor. Rejected poses have zero
  pitch/roll/thrust/yaw authority. After `0.25 s` without association the
  output is bounded level/brake; a `1.0 s` forward commit is possible only
  from a fresh associated pose at at most `8 m` with projected right/down at
  most `1.0/1.0 m` and yaw error at most `0.15 rad`. Runtime inputs remain
  only the legal camera pose, attitude, official phase, and action history.
- Callable SHA-256 is
  `2e2f05b25907a14ba062f2a140c1068fbc625693302165b97cf811e00d43b8fc`.
  All three N237 live traces pass the exact identity-lock evaluator. They
  exercise `40/49/45` rejected aliases and `38/47/44` identical-state
  rejected-alias versus missing-frame parity comparisons, with maximum action
  difference exactly `0.0`. Maximum accepted forward regression is only
  `1.4765/0.9767/1.5716 m`; each trace has exactly one initial family.
  Evaluator artifact SHA-256 values are
  `3e9df7d5864a8e0a28d4ecf4e00ceabc541747ed245425e1a2dfcad8da59e481`,
  `3d48f94cd4d3082e8605e7d8df574b17f9d85705422d76401be1c9cf3d85f537`,
  and `9f49477125f8281f79d6d1be4c5f4bb982d2cc7f92a24852b6ba0eb7a0068007`.
- The actual Windows deployment regression passed `210/210` in `46.03 s`;
  both launchers parse and `git diff --check` passes. Hybrid/batch launcher
  SHA-256 values are
  `660c1113a69a92d6106d873c965c0265072a6d079f775448286ae57dd6c82902`/
  `42aeefb2c4f50a26596f840bf7684625f91cfbf4073b043ab71b2bdbb609cba3`.
  PID `24476` was recovered through the UI without relaunch to active VQ1/R1,
  base/status `193/4`, index `0`, finish `-1`; recovery artifact SHA-256 is
  `95ef599eb8d935ba6a1cc17723ee8315beb82975c78227cbc8821c22276ac46c`.
- Mandatory command-free shadow passed with `599/599` fixed `60 Hz` decisions,
  maximum replay error `2.707422256387204e-07`, `127` detections, no-op
  cadence `63.702 Hz`, zero setpoints/reset/arm/disarm commands, and zero
  collision, telemetry dropout, or drain-limit hit. The race remained active
  at index `0`, finish `-1`, base/status `193/4`. Shadow/parity SHA-256 values
  are `d9a313414ea394497826f00ee879ea7c4b17dbbd24f176634e72576d18161658`/
  `cff1e2a42886a7407e4679c0506f68fb9a1bb7575845faa10f7cf74918bbbdb7`.
- Preregister exactly variant `IL`, tag
  `competitive_n238_identity_lock_batch_007`, three attempts, `45 s` each,
  command request `80 Hz`, policy state `60 Hz`, target six official gates,
  and stop on the first valid finish. Reuse PID `24476` without relaunch and
  preserve the documented reset/fresh-start, health, collision, cadence, and
  command-free post-stop proofs. Dry manifest SHA-256 is
  `1b34f95430f705f0139ce0d8e3b0215079291aa77746dc72e3454c6576546b62`.

## N238 official result — rejected — 2026-07-19

- The preregistered N238 `IL` batch completed unchanged with official counts
  `3,3,3`, zero finishes, and zero collision-free runs. All three attempts hit
  Gate-4 object `1002` at `16.781/17.078/17.250 s`; threat/impact values were
  `2/1.183825`, `1/0.051233`, and `2/1.017933`. Effective command rates were
  `63.162/63.881/63.884 Hz`, with zero rate violation, telemetry dropout, or
  drain-limit hit and valid fresh resets. Final manifest/score SHA-256 values
  are `158c1ea0179688bd0ed48c3765a551631f35557d1663ca1445f74b3015ed70ef`/
  `122bf6b103ef92fbf04ab5ec8b43002d92d806fd8e5b03de1e765b929b946f9b`.
- Identity rejection worked but the control law failed. Replays retained zero
  rejected-alias action error, `25/33/30` associated fresh samples, and only
  `1/1/2` persistently closer re-associations. The crossing commit triggered
  zero times. Closest raw poses were about `5.11/3.61/-1.53 m`,
  `4.80/2.46/-1.35 m`, and `3.79/1.88/-1.24 m` forward/right/down while
  the controller still commanded brake pitch and negative roll. Associated
  lateral rates were still `+5.3` to `+6.7 m/s` and yaw errors
  `0.46-0.61 rad`; the direct projected servo counter-banked too late and hit
  the aperture/frame. Live identity-evaluation SHA-256 values are
  `bc9b169b83c80c6e89c3f68cb727726863bec0eab1a17549b329f330c2c982a1`,
  `17643976350cd66b80283b1de69b14425f9ab7aedf6a8ced13c0bc313a112ad2`,
  and `0fe620a1902134a2980fd6e8cac1621207e3d153fb4e0b58f14417d71b079e01`.
- Do not repeat N238, relax its commit envelope, alter a scalar PD gain, or
  treat closer-family confirmation as solved identity. The traces now provide
  clean associated official input/output segments. The next admissible work is
  offline system identification and controller optimization on those segments,
  with held-out whole-run prediction/control evidence before another flight.

## N239 trained identity-body lateral MPC admission and preregistration — 2026-07-19

- N239 is the required offline system-identification/controller-optimization
  successor to rejected N238, not another scalar PD adjustment. The v2 fit
  reconstructs locally coherent N238 association segments from all six N237/
  N238 official traces, excludes every transition across a family re-anchor,
  and uses only recorded official actions. It provides `113` clean transitions
  and six leave-one-whole-run-out structured models. Selected ridge is `10`;
  mean held-out next-rate RMSE is `2.2992/1.3313/0.7128 m/s` and the aggregate
  score is `1.447774`. Only roll-to-right acceleration is deployable across
  every fold: its gain range is `3.1081-3.9474 m/s^2` per normalized roll.
  Pitch/forward and thrust/down action gains change sign across folds and are
  explicitly forbidden from optimization. Fit source/model SHA-256 values are
  `5952461d3756b2c53c5de10c6d60c1cc8ab9eab4f65ad10bb41f0bb6c11174bf`/
  `a40e9279bc2d34f69a2296350971ee27eb1393f64e5527b37f2bdc1a545677df`.
- The first counterfactual draft was rejected offline and never flown. It tried
  to center every identity-associated family and passed only `17/42` rollouts;
  all `24/24` far-family rollouts remained uncontrollable even at roll `-1`.
  N239 therefore quarantines far associations: they have zero action authority
  and command N238's bounded level/brake. The learned optimizer is unlocked
  only when a newly acquired or newly re-acquired coherent family first appears
  at at most `16 m` forward. It then runs deterministic five-knot, `384`-
  candidate, four-iteration robust lateral MPC over the fold ensemble and
  executes only the first roll. N238's association/rejection, pitch, thrust,
  yaw, dropout, and commit rules remain exact; rejected raw poses still have
  zero action authority. Runtime inputs remain legal camera-relative pose,
  attitude, official phase, and action history, with no simulator truth or map.
- The admitted counterfactual selects the three live near-family N238 entries
  at `14.328/-4.657`, `14.118/-5.449`, and `11.566/-6.452 m` forward/right.
  For each entry, every one of six dynamics members is used as the plant while
  being excluded from the planner. All `18/18` rollouts cross within the
  `1.0 m` lateral box; worst terminal absolute right error is `0.969814 m`.
  Core/evaluator/counterfactual SHA-256 values are
  `1059acff7135bbdd7bb20fc9630891d28cda7794630a3a51a590fc014d96338b`,
  `6a67110324625fa2c7bb6eb2945bb303e3ffc85f4078596a072b91d6e15e77e2`,
  and `4c631c0cbcfd688457433a8000cf7b9601a3ecf2992e62a732d9028268222ea3`.
  The source-pinned N239 callable SHA-256 is
  `716352e7b2ec603e0316e59f0d97cc6fe96ca5bdde4f5f7923b0be2fb6b75f4c`.
- A command-free deployment shadow exposed and corrected one runtime defect:
  the first callable re-hashed the UNC model on every `60 Hz` state step,
  causing a catch-up feedback loop. Shadow 011 passed only `205` decisions;
  shadow 012 reached `592` but failed with two drain-limit hits. Model hash
  validation is now once per controller construction and is regression-tested.
  The final actual-Windows deployment suite passed `561/561` in `119.20 s`;
  both launchers parse and `git diff --check` passes. Hybrid/batch launcher
  SHA-256 values are
  `6ca53a4e3945402b770175180a14818755ebce137e28ccc613b4ecaba8d5f85c`/
  `4a3aa2d2625ebfdaca54f3e9d613da9a7b62dc379f281e43594fbf54641ee5f9`.
- Responsive shipping PID `24476` was recovered through the VQ1/R1 UI without
  relaunch; recovery SHA-256 is
  `3b41554476ea69b2a651c66f62781d3871a6a5bbfcda9c74b1c8145c2574b995`.
  Final mandatory shadow 013 passed `600/600` fixed-clock decisions at
  `59.910 Hz`, `128` detections, maximum replay action error
  `2.5794487001906674e-07`, and no-op cadence `63.804 Hz`. It sent zero
  setpoints/reset/arm/disarm commands and recorded zero collision, telemetry
  dropout, or drain-limit hit. The race remained base/status `193/4`, official
  index `0`, finish `-1`. Shadow/parity SHA-256 values are
  `fc7b1d370c9b45a45c50ff5757cd73f06091a5e1443723bfb91c02936d5dcb96`/
  `7869523662a571023e5a65905c47102dfbd6c957e9f0d3e63361b77f005742c4`.
- Preregister exactly variant `LM`, tag
  `competitive_n239_identity_body_mpc_batch_008`, three attempts, `45 s`
  each, requested command `80 Hz`, policy state `60 Hz`, target six official
  gates, and stop on the first valid finish. Reuse responsive PID `24476` and
  never relaunch it. Each attempt must retain normal disarm, `100 ms` wait,
  exactly one learned-target `COMMAND_LONG 31000` with confirmation/parameters
  zero, boot rollback over `1000 ms`, fresh index `0`/finish `-1`, all health
  and cadence checks, and the command-free post-stop proof. Dry manifest
  SHA-256 is
  `301ddfa1cb0080db4f42eb222d0b7c283efd2c91b845892b6ce0f931c52da004`.

## N239 official result — rejected deployment, unexercised MPC — 2026-07-19

- The preregistered `competitive_n239_identity_body_mpc_batch_008` `LM`
  batch completed unchanged with official counts `3,3,3`, zero finishes, and
  two collision-free attempts. Run 001 hit Gate-4 object `1002` at `41.329 s`,
  threat `2`, impact `6.007645`; runs 002/003 remained collision-free for the
  full `45 s` but never advanced beyond official index `3`. Effective command
  rates were `63.151/63.688/63.710 Hz`; the fixed state schedule delivered
  `2479/2700/2700` steps at `59.982/60/60 Hz`, with zero command-rate
  violations, telemetry dropouts, or drain-limit hits. All three starts used a
  detected disarm-first, single-command-`31000` reset with boot rollbacks
  `2604520->3339`, `33834->3357`, and `20139->3439 ms` and fresh race starts.
  Final manifest/score SHA-256 values are
  `ad4742281a1078385319187fdcdf8a7310161b261a7ab3526ec495393bfad6d0`/
  `98b13a8dcadca042da34363a013ac89ea6b02ea13b53ccb91d26f4214007d246`.
- The learned lateral optimizer was never exercised. Every retained Gate-4
  action had exactly zero roll. A timestamp-preserving audit through N239's own
  association object found accepted-family minimum forward positions
  `15.781706`, `16.000003`, and `15.026607 m`, but zero trusted-family
  triggers, zero optimized plans, and `302/338/337` quarantined associated
  calls. The three-micrometre excess in run 002 is explicitly treated only as
  diagnostic encoder-boundary tolerance, not as a deployed comparison change.
  Audit script/artifact SHA-256 values are
  `88ed13e972a6db26fec3b23a08a623f539b85be0ab17026d84c6c3ab12eeda4c`/
  `ed9b2aaedf5d516e699a0dd2bbafe05182c96d3bd326d1962231d3f0a03ea6a3`.
- Root cause is the admission transition: N239 set `trusted_near_family` only
  if the `<=16 m` condition was already true on initial acquisition or a
  discrete re-acquisition. A continuously associated far family could enter
  the near region without ever enabling MPC. Reject N239 as deployed and do
  not repeat it unchanged. This result does not falsify the trained roll model.
  The next admissible candidate is the minimal semantics correction: allow a
  newly accepted, still-coherent associated sample to latch trust when it
  crosses the same `16 m` boundary, while preserving zero authority for
  rejected poses and preserving all model, pitch, thrust, yaw, dropout,
  commit, prefix, and safety contracts. It must first prove the transition and
  robust centering on all three N239 traces offline; no threshold relaxation or
  new PD gain is authorized.
- Responsive shipping PID `24476` was recovered through the VQ1/R1 UI without
  relaunch after the batch. It is again base/status `193/4`, official index
  `0`, finish `-1`. Recovery SHA-256 is
  `f2a57a14653c3cc0f9b9f735e62ecaf7d724f55fc26d745b5c910ea76b379cbb`.

## N240 offline rejection and N241 roll-identification probe — 2026-07-19

- N240 implemented the minimal N239 admission correction: an accepted,
  continuously associated family could latch trust at the unchanged nominal
  `16 m` boundary. A `1 mm` encoder-round-trip tolerance was explicit because
  one nominal boundary sample reconstructed as `16.000003 m`; rejected poses
  remained inert. N240 was rejected offline and never flown. It unlocked on all
  three N239 traces, but only `2/18` leave-one-model-out lateral rollouts
  centered; maximum terminal error was `15.802635 m`. A scan of every accepted
  near state found zero feasible plans in all three traces; per-run best worst-
  member errors were `7.516056/15.802635/2.368993 m`. Policy/evaluator/result
  SHA-256 values are `8f3b622b...`/`d3ce1637...`/`d1dba169...`; scan script/
  artifact values are `1325009a...`/`83c99ed1...`.
- Adding the three zero-roll N239 trajectories to all six N237/N238 traces
  produced `145` clean transitions and improved separation of ambient drift
  from action. The nine-whole-run fit correctly failed admission: its full
  roll gain was only `1.1210`, fold gains ranged `0.1387-2.2954`, and the
  required roll response was not identified in every fold. The artifact is
  `passed=false` and must not be loaded by a controller. Its SHA-256 is
  `6bc624b805185ca8ad5dfd091a5393378caa078e3554be00bdf250088d82378b`.
- N241 is therefore one bounded causal identification probe, not a lap
  candidate. It inherits N238's persistent accepted association and exact
  Gates-1-through-3 parent. Only while that association is fresh and
  `20-40 m` forward, it applies the balanced two-second roll sequence
  `[-.35,+.35,-.35,+.35,-.20,+.20,-.20,+.20]`, each for `0.25 s`.
  Pitch/thrust/yaw remain the level-brake values. A newly accepted sample below
  `20 m`, a stale/missing association, or completion ends excitation; rejected
  raw poses have zero action authority. Callable SHA-256 is
  `205ab6380a92c4770d42793c6b0450f16294d21ce9b8e5efabe6708fd3f8301a`.
- Actual-Windows focused tests passed `14/14` and the launcher parses. Passive
  shadow 015 passed exactly `600/600` decisions at `60 Hz`, maximum replay
  action error `2.300133e-7`, no-op cadence `63.696 Hz`, and zero setpoints,
  reset, arm, disarm, collision, dropout, or drain-limit hit. Shadow/parity
  SHA-256 values are `aa64b2473a384d2c5790b11d1f559c3541ff09e62e962b6db5016e362fc5a29d`/
  `1873c5e7cf93a39d500d81bf1751535bf7ce2153197a44e52a136836e3f6822c`.
- Preregister exactly one diagnostic tag `n241_gate4_roll_identification_probe_001`
  in `FullLap` lifecycle mode for `15 s`, requested command `80 Hz`, state
  `60 Hz`, and target six gates. Reuse PID `24476` without relaunch; require
  normal disarm, `100 ms`, exactly one learned-target command-`31000` reset,
  boot rollback over `1000 ms`, fresh index `0`/finish `-1`, health/cadence
  checks, and final disarm. The result is useful only if official index `3` is
  reached and multiple balanced probe phases are recorded before the near
  abort. Stop after this single probe and refit before any further flight.

## N241 official diagnostic result — uninformative prefix miss — 2026-07-19

- The single preregistered 15-second diagnostic completed exactly once. It was
  collision-free and transport-clean but passed only Gate 1: official index
  `1`, finish `-1`. The retained trace contains `48/102/0/0` samples at
  official indices `0/1/2/3`, so the Gate-4 controller never activated and
  recorded zero occurrences of every probe roll value. No dynamics row or
  controller conclusion may be inferred from this attempt, and it must not be
  repeated unchanged.
- The start proved normal disarm plus one reset-`31000`: boot rolled
  `1815309->3419 ms`, fresh start `3300 ms`. The run delivered exactly `900`
  state steps at `60 Hz` and commands at `59.312 Hz`, with zero collision,
  command-rate violation, telemetry dropout, or drain-limit hit. Smoke,
  validation, and summary SHA-256 values are
  `7731990e1119a6cd10dfd13681f339ff66454a1d4bb41ecfea5c1c32bdd48741`,
  `42b3b0d4b90396e37954da62bacee4d58cd0dbbc91eeaf6beb848cd1464201cb`,
  and `bf84041e3609b416dc136939f84d2d7700f5a23df66329e96b9cbe130b67f486`.
- PID `24476` was recovered through the UI without relaunch and is again
  base/status `193/4`, official index `0`, finish `-1`; recovery SHA-256 is
  `62bdc8296b522c3ffcd58dc31c998cc15ab0cddabd5dea3923ce50a6cecf7e0a`.
  Before another identification attempt, add a runner-owned bounded stop clock
  that begins only when official index `3` is first observed. This permits a
  longer prefix opportunity while still disarming shortly after the two-second
  probe. Do not extend blind flight duration without that stop contract.

## Current execution override — N252 native six-gate policy — 2026-07-26

This end-of-file section supersedes earlier next-action directives.

- The active simulator is telemetry-enabled VQ1 v3391. N250 proved that even a
  centered constant-`1 m/s` velocity path contacts Gate-1 object `1001`; do not
  repeat the unchanged waypoint controller or assume speed is the only issue.
- Preserve the old visual/attitude checkpoint with SHA-256 `ea03965f...` as the
  clean official three-gate baseline. Its observation and action semantics are
  incompatible with the new local-telemetry/local-velocity network.
- N251's teacher-driven six-gate result is not a trained-policy result. Its
  checkpoint failed teacher-off at `0/6` over exactly `512` episodes.
- N252 is the current learned candidate. Explicit behavior cloning used `512`
  successful full-course demonstrations and produced
  `checkpoints/drone_race_vq1_v3391_telemetry/n252_bc_6gate.bin` (SHA-256
  `cec95192a04ebbe0c28f61226fcd76a6851404b6548be12ca6c44165397e9d76`).
  Teacher-off deterministic native evaluation passed `512/512`, all six gates,
  valid/success `1.0`, mean `23.065336 s`, zero miss/timeout. Evidence SHA-256:
  `9d00c204ebc983974992e92495e9cb233aed1a31de6c3ecac8940abd2afe297a`.
- Proceed by implementing a command-free live v3391 shadow of the exact policy
  observation and recurrent inference contract. It must send zero reset, arm,
  setpoint, or disarm commands. Do not authorize a full live lap until the
  shadow passes and the N250 Gate-1 collider/track-coordinate discrepancy is
  explained or avoided by measured geometry.

### Current execution override — N253--N255 transfer audit

N253 already passed a fresh, command-free `60.1 Hz` live shadow with exact N252
checkpoint identity and zero outbound MAVLink messages. N254 then passively
matched the visible Gate-1 aperture (`23.4146 m` camera estimate) to the published
center (`23.3014 m`), rejecting any speculative half-height coordinate shift.

Run N255 exactly once before actuating the learned policy. Repeat the N250
constant-`1 m/s` path, stop at official index `1`, and allow continued control
only for object `1001`, threat `<=1`, within `0.5 m` of the active gate center.
Log those messages separately. Abort on any threat `2`, off-center warning,
other object, invalid state, or timeout. Do not retry. A pass resolves Gate-1
event ordering only; it does not promote N252 to an official six-gate finish.

N255 has now rejected that event-order hypothesis: `88` threat-1 gate messages
physically stopped the vehicle about `0.205 m` before the center, pushed it from
NED z `-0.034 m` to roughly `+0.408 m`, and official index remained `0` even
after it moved past the plane. Do not repeat N255.

Run N256 exactly once: the same constant-`1 m/s` bounded Gate-1 path with target
z equal to published z plus `+0.4 m` NED-down, published x/y unchanged, no
collision exception, and immediate stop/disarm at official index `1`. Any
collision or no advance rejects this geometry. Do not actuate N252 yet.

N256 has now cleanly crossed below the structure at transmitted z plus `0.4 m`
NED-down: no in-course collision, no score, index `0`. Paired with N255's frame
contact and downward deflection at transmitted z, this identifies transmitted z
as the gate bottom/base. Supersede the N254 image-only interpretation.

Run N257 exactly once at published x/y and published z minus `1.36 m` NED-up,
the center of a `2.72 m`-high gate. Keep constant `1 m/s`, strict collision
abort, and immediate stop/disarm on official index `1`. Do not retry. If it
passes, update all six native aperture centers and regenerate/retrain the policy
before any learned live actuation.

N257 passed: official index `0 -> 1` at `24.469 s`, no collision, at target NED
z equal to transmitted z minus `1.36 m`. The PufferLib config now applies this
proven transform to all six gates. Preserve N252, but do not actuate it because
its target z values were the transmitted bases.

Proceed with N258 by transfer, not from scratch: use N252 as initialization,
regenerate `512` corrected six-gate demonstrations, train the full two-layer
network, and demand a deterministic exact `512/512` teacher-off six-gate screen.
Then perform a fresh zero-command shadow before any actuated learned-policy run.

Reject N258: its trainer invocation omitted `--gate-progress-observation`, so
legacy three-phase decoding excluded all `105472` Gate-2 records. It passed only
one gate and timed out teacher-off. The trainer now detects this six-gate contract
and fails closed instead of silently training a partial course.

Run N259 from N252 with the accepted corrected `512`-episode dataset, full-network
BC, `--gate-progress-observation`, and verify that training reports all `718033`
samples. Require exact deterministic `512/512` teacher-off six-gate success.

N259 passed that gate: exact `512/512`, six gates, mean `23.088762 s`, zero
miss/timeout. N260 then passed `601` command-free live inferences at `60.1 Hz`
with zero outbound MAVLink messages.

Run N261 exactly once using N259: reset-isolated, `60 Hz` local-NED velocity,
`2 Hz` heartbeat, `0.5 s` zero-body-rate launch at thrust `0.27`, strict abort
on every non-launch collision/fault, and stop/disarm on official index `1`.
No retry. A pass authorizes a separately preregistered full-course attempt.

N261 passed: learned N259 advanced Gate 1 at `4.922 s`, collision/fault null,
control `60.138 Hz`, exact checkpoint hash and gate transform. Run N262 exactly
once with the frozen N259/N261 contract for at most `45 s`; require official
index `6`, nonnegative finish time, and no collision/fault. Stop/disarm on finish.

N262 cleanly passed Gates 1 and 2, then crossed the Gate-3 plane `1.8731 m`
off-center and oscillated because the official active index correctly remained
`2`. Do not retrain from random weights and do not rerun N262 unchanged. Preserve
N259 and execute N263 once, stopping at official index `3`. The policy continues
to infer at `60 Hz` and owns learned along-track speed; public track/local-pose
telemetry governs bounded cross-track correction and schedules crossing speed to
`4 m/s`. Abort on any non-launch collision/fault or if the unchanged active gate
is more than `1.5 m` behind the vehicle. Only a clean N263 Gate-3 pass authorizes
one identical full six-gate attempt with no tuning between runs.

N263 passed that gate: official progress reached `3` at `14.421 s`, collision
and invalid reason were null, and final sampled Gate-3 cross-track error was
`0.06233 m`. Run N264 exactly once with no parameter/source/checkpoint changes;
change only `--stop-after-gate-index 6` and use a `40 s` bound. Require official
index `6`, nonnegative finish time, null collision/fault, then stop and disarm.

N264 passed all six official gates. Treat
`logs/sitl/n264_v3391_n259_governed_full_lap_001.json` (SHA-256
`3752249b0560a165f870fc5d222d111694f10b8290b2816de644c92ae3ce6d90`)
as the immutable live baseline: official finish `29.237468719 s`, index `6`,
null collision/invalid reason, no telemetry health faults, `60.024 Hz` control,
and final stop/disarm sent. Do not restart training or overwrite this artifact.
Proceed to competitive-time optimization by sweeping the public-telemetry
governor's crossing speed/slowdown offline, preserving N259 and all N264 safety
contracts. Give any live optimization attempt a new run number and require it
to be both valid and faster before promotion.

N265 completed that offline requirement. It reconstructs N264's exact official
segment times as `4.976024/4.279658/4.999875/6.502513/4.284304/4.195095 s` and
fits the observed command response rather than assuming native dynamics. The
primary x-axis steady gain/time constant are `0.8341/0.8369 s`; N264's mean
scheduled along speed exceeds actual by `1.10--1.77 m/s`, and its cross-track
cap never activates. The full grid contains `840` deterministic candidates and
retains the raw baseline finish error of `-0.3572 s` in its evidence.

The minimal sub-27-second rung changes only governor crossing speed from `4.0`
to `7.5 m/s`; all other N264 settings, code, N259 inference, and safety guards
stay exact. Residual-corrected prediction is `26.792198 s` with maximum modeled
crossing error `0.30604 m`. N265 evidence SHA-256 is
`8642d60f8192f494791b94bc85a37ec58dc457ee0c89bae536deeb554a2cfede`.

Run N266 once under tag `n266_v3391_n259_cross7p5_gate3_bounded_001`: exact
N259 checkpoint, `7.5/8.0/10/1.5/4.0` crossing/max/slowdown/gain/cap, `60 Hz`,
`2 Hz` heartbeat, `20 s`, and stop at official index `3`. Retain the normal
disarm-plus-one-31000 reset proof, six transferred gates, strict collision and
telemetry aborts, unchanged `1.5 m` plane-miss guard, and final stop/disarm. Do
not retry N266 unchanged. Only a clean bounded result authorizes an identical,
separately numbered full-course run.

N266 passed: official Gates 1--3 at `4.707435/8.494735/13.016106 s`, index `3`,
finish `-1`, null collision/invalid reason, `60.062 Hz`, exact reset and six-gate
transfer, healthy telemetry, and final stop/disarm. Its prefix is `1.239449 s`
faster than N264. Evidence SHA-256 is
`f3dc321301e5a5cd4a770ae66f455642a4e2529fb2999e2773a7ad722a3e7bf8`.

Run N267 exactly once with tag `n267_v3391_n259_cross7p5_full_lap_001`. Keep the
entire N266 controller/lifecycle unchanged; only set stop index `6` and duration
`40 s`. Accept only official index `6`, nonnegative finish, no collision/fault,
healthy `[50,100) Hz` transport, and final stop/disarm. Do not retry N267
unchanged. A faster valid N267 still requires two separately numbered identical
valid confirmation laps before the requested repeatability claim.

N267 passed all six gates in official `26.604381561 s`, saving `2.633087158 s`
(`9.006%`) versus N264. Collision/invalid reason are null, transport is
`59.990 Hz` with no telemetry fault, and final stop/disarm passed. Evidence
SHA-256 is `212c5af1b10909b3a694c60203a3649de84ba116c3911fd7d587eff76a1c8993`.

Run N268 once under `n268_v3391_n259_cross7p5_full_lap_confirm_002` with every
N267 byte, scalar, reset/lifecycle rule, guard, and acceptance criterion exact.
Do not retry unchanged. A pass leaves one identical N269 confirmation before a
repeatable promotion; a failure requires offline reliability diagnosis.

N268 passed identically at official `26.591306686 s`: index `6`, null collision/
invalid reason, healthy `60.032 Hz`, and final stop/disarm. Artifact SHA-256 is
`51d36c1154133606b8bdefdf95818a4972c53c17efc8922a87060a68e8b950db`.
N267--N268 are `2/2` valid with `0.013075 s` spread.

Run final identical N269 once under
`n269_v3391_n259_cross7p5_full_lap_confirm_003`. Change only the JSON path;
retain all hashes, parameters, lifecycle, guards, and acceptance rules. A pass
completes the required three-lap repeatability evidence; a failure reopens
reliability diagnosis. Do not retry unchanged.

N269 passed at official `27.412017822 s`, still `1.825450897 s` faster than
N264, with index `6`, null collision/invalid reason, clean `59.833 Hz`, and
final stop/disarm. Artifact SHA-256 is
`9be363aa9c96e1623fceea287314f1241ee34269facf9ba8548b3076f319cf73`.

Promote the exact N259-plus-crossing-`7.5` controller as N270. Its three valid
official laps have best/median/worst `26.591306686/26.604381561/27.412017822 s`,
spread `0.820711136 s`, and every lap beats N264. The best saves
`2.646162033 s` (`9.051%`). Aggregate artifact SHA-256 is
`ff466a254fc829dd764484783b20bcf1fdca0c42f96fbba6045f937da2e0a247`.
The aggregate audit re-hashes every live artifact and frozen deployment input;
focused tests pass `7/7`. The sub-`27 s` and repeatability requirements are met.
Do not claim the optional near-`24 s` stretch target.
