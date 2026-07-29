# Hybrid RL Course Execution Prompt

Copy the text below into a coding-agent session.

```text
Goal: produce the first official gate-2 pass and then full-course completion in
AI-GP Simulator v1.0.3385 using the fast hybrid FSM → RL path.

Do not resume manual gate-2 controller gain sweeps. The deterministic FSM owns
startup, gate 1, post-pass braking, safety/limp abort, reporting, and reset. A
recurrent PufferLib policy takes over only after official active_gate_index >= 1
and the FSM brake_complete transition.

Verified runtime:
- Simulator: C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe
- Keep the healthy simulator process alive. Normal retries use MAVLink command
  31000; never process-restart for a race retry.
- Command at 60 Hz through the fixed publisher; require effective_command_hz >=
  50, zero drain-limit hits, and collision-triggered early abort.
- Official active_gate_index is the only pass authority.
- Track geometry telemetry is intentionally nulled.
- Gate 1 and brake are already proven in three consecutive official attempts.

Verified RL fixes:
- Native interface_mode=2 now treats policy outputs as absolute pitch, roll,
  yaw, and thrust, then closes attitude through the same gyro/body-rate plant as
  the v3385 live runner. It no longer integrates policy angles as raw rates.
- Normalized thrust zero maps to hover 0.27; [-1, +1] spans [0.18, 0.42].
- The native evaluator defaults to horizon 32 so episodes actually terminate.
- Latest-checkpoint selection orders by run timestamp, then checkpoint step.
- Hybrid live handoff is enabled by passing a checkpoint to
  scripts/run_windows_course_fsm.ps1.

Execute:

1. Preflight once:
   cd /root/pufferlib-drone
   bash scripts/test_drone_race_native_regressions.sh
   .venv/bin/python -m pytest -q \
     tests/test_drone_policy_contract.py \
     tests/test_drone_sitl_competition_smoke.py \
     tests/test_drone_course_controller.py

2. Run the two-stage native curriculum (only one training process at a time):
   REPO_ROOT=/root/pufferlib-drone bash scripts/train_gate2_rl_fast_cycle.sh

   Stage 1 must reach success_rate >= 0.90 and crash <= 0.10.
   Stage 2 must reach success_rate >= 0.80 and crash <= 0.15 on 4096
   deterministic episodes. Do not promote a checkpoint that misses either gate.

3. Validate only the promoted checkpoint from native Windows PowerShell:
   Set-Location '\\wsl.localhost\Ubuntu\root\pufferlib-drone'
   .\scripts\run_windows_course_fsm.ps1 `
     -SimRoot 'C:\Users\anon\Desktop\AI-GP Simulator v1.0.3385' `
     -PolicyCheckpoint '<checkpoint-relative-path>' `
     -PolicyActivateGateIndex 1 `
     -SmokeDuration 30 -TargetGateCount 2 `
     -Tag hybrid_gate2_001

4. Promotion requires official_active_gate_index >= 2, acceptance_passed=true,
   zero collision, effective command rate >=50 Hz, and policy_ticks > 0.

5. If a promoted checkpoint fails twice with the same official signature, stop
   live attempts. Record the last visible gate vector/image, collision ID/time,
   policy action trace, and official index. Change the native curriculum,
   observation contract, or plant randomization to reproduce that signature;
   do not tune the FSM around gate 2.

6. After gate 2 passes, extend native curriculum one gate at a time. Keep the
   FSM handoff at gate 1, require native deterministic promotion, then one
   official proof. After one full completion, obtain 8/10 valid runs before any
   speed optimization.

Every iteration must report:
- exact training/eval/official command;
- checkpoint path and native success/crash/completion metrics;
- official gate index, collision/invalid status, policy activation time/ticks;
- elapsed cycle time and exactly one next hypothesis.
```
