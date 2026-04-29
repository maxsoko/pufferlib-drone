# Drone Race Project Description for Non-ML Readers

This document explains the drone race project in plain language. It is written for someone who wants to understand what the project is doing, what the current blockers mean, and how to prompt an AI coding agent more effectively.

## One-Sentence Summary

We are building software that teaches a simulated drone to fly through race gates reliably, then later connects that trained behavior to the official simulator interface used by the competition.

## What We Are Trying To Build

The final goal is an autonomous drone system. "Autonomous" means the drone decides what to do by itself. It should:

- Take off or start from the race start position.
- Fly through gates in the correct order.
- Avoid crashing.
- Finish the course as fast as possible, but only after it is reliable.
- Eventually talk to the official simulator using MAVLink, which is a standard message protocol for drones.

The important part is reliability first. A fast run that crashes is useless. The current work is focused on making the drone consistently complete a smaller three-gate course before moving to four gates or a full race.

## What This Project Is Not Yet

This is not yet a complete competition submission.

Right now, the best results are from a fast internal training environment called `drone_race`. That environment is useful for training and debugging, but it gives the learning system easier access to internal simulator state than the real competition interface will allow.

Before this can count as competition-ready, the policy must run through a SITL/MAVLink interface and use official telemetry and camera inputs instead of privileged internal state.

## Important Terms

### Environment

An environment is the simulated world the drone flies in. It contains the drone physics, gate positions, crash rules, rewards, and metrics.

In this project, the important environments are:

- `drone`: a hover/control environment used to teach stable flight basics.
- `drone_race`: a gate-racing environment used to train gate navigation.

### Policy

A policy is the decision-making model. It receives observations and outputs actions.

In plain language: the policy is the "pilot brain" that sees the current situation and decides what motor commands to send.

### Observation

An observation is what the policy can see at each moment.

Right now, the native race environment uses a compact 23-number observation vector. These numbers describe useful drone and target information. This is good for fast training, but it is not the same as the final competition interface, which should use telemetry and camera data.

### Action

An action is what the policy outputs.

For this project, actions are four normalized motor commands. The environment turns those motor commands into realistic quadrotor movement using physics.

### Reward

A reward is the scoring signal used during training. It tells the policy whether its behavior was good or bad.

Examples:

- Moving toward the next gate gives positive reward.
- Passing a gate gives positive reward.
- Finishing gives positive reward.
- Crashing gives negative reward.
- Using too much control effort can give a small penalty.

The reward is not the final goal by itself. The final goal is reliable gate completion. Reward is only the training signal that tries to produce that behavior.

### Checkpoint

A checkpoint is a saved version of the policy.

Example:

`checkpoints/drone_race/1777415131824/0000000039387136.bin`

That file stores trained model weights. We can reload it and continue training, or evaluate it to see how well it flies.

### Evaluation

Evaluation means running a saved checkpoint without further learning and measuring how well it performs.

This project now prefers deterministic JSON/CSV eval artifacts instead of relying only on the live training dashboard. That matters because dashboard snapshots can be misleading.

### Promotion

Promotion means a checkpoint is good enough to become the official starting point for the next stage.

Example:

- A strong two-gate policy can be promoted to start three-gate training.
- A strong three-gate policy can be promoted to start four-gate training.

Promotion should require saved eval artifacts, not vibes.

## The Current Training Path

The project uses staged training. Instead of asking the drone to solve the full race immediately, we train easier tasks first.

### H1: One-Gate Bridge

The drone learns to fly through one gate.

Status: solved.

### R1: Two-Gate Bridge

The drone learns to fly through two gates with a mild turn.

Status: strong. This is the canonical restart point for many three-gate experiments.

Current important checkpoint:

`checkpoints/drone_race/1777139257574/0000000052494336.bin`

### R3: Three-Gate Reliability

The drone learns to fly through three gates.

Status: close, but not promotion-ready.

Current best candidate:

`checkpoints/drone_race/1777415131824/0000000039387136.bin`

Current issue:

- Success is near the desired range.
- Crash rate is still too high.
- Crashes are almost entirely low-altitude dives.

### R4: Four-Gate / Near-Full Race

The drone should only move to this stage after R3 is reliable.

Status: blocked.

## Current Best R3 Understanding

The best R3 candidate has shown promising results, but it is not stable enough to promote.

Observed results:

- One deterministic scan reached about `success_rate=0.8629`.
- Crash rate was still about `0.1371`, above the target of `<= 0.10`.
- A larger re-eval was lower, so this checkpoint is promising but not robust.

The latest diagnostics show:

- Crashes are not caused by flying too far left/right.
- Crashes are not caused by flying too high.
- Crashes are low-altitude dives.
- Lowering the crash floor does not fix the issue.
- Simple altitude-floor reward shaping did not produce a robust promoted checkpoint.

Plain-English interpretation:

The drone often knows how to get through the gates, but some runs end with it diving downward too hard. The issue is not just that the crash boundary is too strict. The control behavior itself needs to avoid entering that downward failure mode.

## Why R4 Is Blocked

R4 is the four-gate stage. It should not be resumed as a serious promotion attempt until R3 is reliable.

If we start R4 too early, the model can learn unstable shortcuts or regress badly. That has already happened in earlier experiments.

The correct approach is:

1. Make R3 pass deterministic eval.
2. Save the checkpoint and eval artifacts.
3. Use that checkpoint as the warm start for R4.
4. Treat any R4 run before that as a diagnostic, not a promotion attempt.

## What "Native v4" Means

Native v4 means the project is using PufferLib's fast C/CUDA-backed training path instead of a slower Python fallback.

Plain-English version:

- Python is convenient but can be slow.
- C/CUDA is much faster for running many simulated drones at once.
- Faster simulation means more training data per hour.
- More training data makes it practical to iterate on drone behavior.

For accepted training and performance work, use the native `_C` backend. The `--slowly` path is debug-only.

## What "PufferNet" Means

PufferNet is the policy architecture used here. It is the model that maps observations to actions.

The current policy is roughly:

1. An encoder turns the observation numbers into an internal representation.
2. A small recurrent memory layer, `MinGRU`, tracks recent history.
3. A decoder turns the internal state into motor actions.

Plain-English version:

The model has a small memory. It does not only react to the current frame; it can remember recent motion, which matters for flying.

## What Quantization Means

Quantization means making the trained model smaller and faster by using lower-precision numbers.

For example:

- Normal training may use 32-bit floating point numbers.
- Edge inference might use 8-bit or 16-bit numbers.

Why this matters:

- Smaller models can run faster.
- Faster inference is useful for real-time drone control.
- But lower precision can change behavior.

Current status:

Q8/mixed-precision inference tools exist, but quantized race control is not promoted yet. We need reliable FP32 behavior first, then prove that quantized behavior still flies safely.

## What MAVLink and SITL Mean

### MAVLink

MAVLink is a communication protocol used by drones. It is how external software sends commands and receives telemetry.

Competition-facing control must eventually send messages like:

- `SET_POSITION_TARGET_LOCAL_NED`
- `SET_ATTITUDE_TARGET`

### SITL

SITL means "software in the loop." It is a simulator setup where the drone stack communicates through realistic interfaces instead of directly accessing internal training state.

Plain-English version:

Native `drone_race` is the training gym. SITL/MAVLink is closer to the real exam.

## Why Native Race Success Is Not Enough

A native `drone_race` success means the policy can solve the internal training task.

It is not yet competition-equivalent because:

- The policy uses privileged state features.
- The official interface uses telemetry and camera data.
- The official control path goes through MAVLink messages.
- The submission needs heartbeat, command rates, and telemetry handling.

This is why the PRD says native metrics are necessary but not sufficient.

## Current Completed Work

Completed in practical terms:

- Native v4 branch is the active/default project branch.
- Native `drone` and `drone_race` environments exist.
- `drone_race` uses more realistic quadrotor physics.
- Race and hover observations are both 23 features.
- H1 and R1 curriculum stages have useful checkpoints.
- R3 has a promising but not promoted checkpoint.
- Deterministic checkpoint eval reports exist.
- Vast.ai GPU training has been set up and validated.
- Q8 edge-inference scaffolding exists.
- MAVLink/SITL scaffold exists at a basic level.

## Current Incomplete Work

Important incomplete items:

- R3 needs lower crash rate.
- R4 is blocked until R3 is reliable.
- Native privileged-state training must move toward telemetry/camera observations.
- MAVLink telemetry parsing is incomplete.
- The final controller output contract is not finalized.
- Local SITL fixed-course evaluation is not implemented.
- Vision ingestion is waiting on a detailed camera stream spec.
- Q8/fake-quant training is not promoted.

## How To Prompt Better

Good prompts should include:

- Which layer you want to work on: training, environment physics, eval, docs, SITL, MAVLink, quantization, or GPU operations.
- Whether the task should change code, run training, or only inspect and report.
- Which checkpoint or run ID should be treated as the starting point.
- What metric defines success.
- Whether to commit and push.

## Prompt Patterns That Work Well

### Ask For Status

Use this when you want orientation:

```text
Read PRD.md and summarize where the drone project stands.
Focus on what is complete, what is blocked, and the next highest-leverage task.
Do not edit files.
```

### Continue R3 Reliability Work

Use this when you want more training/debugging:

```text
Continue R3 native race reliability work.
Use the current best R3 checkpoint from PRD.md as the starting point.
Do not move to R4.
Run deterministic eval before and after any training.
Only promote a checkpoint if saved JSON/CSV eval passes success_rate >= 0.85 and crash <= 0.10.
Update PRD.md with results.
```

### Ask For A Controlled Experiment

Use this when you want one careful change:

```text
Run one R3 experiment changing only [specific variable].
Start from [checkpoint path].
Evaluate the resulting checkpoints with scripts/eval_drone_race_checkpoint.py.
Report whether the change improved success_rate, crash, and gates_passed.
Do not treat it as promoted unless it passes the PRD threshold.
```

### Avoid Accidental R4 Work

Use this when you want to prevent premature full-course runs:

```text
Do not train R4 yet.
First confirm R3 passes deterministic eval.
If R3 does not pass, explain the blocker and propose the next R3-only experiment.
```

### Work On MAVLink/SITL

Use this when you want competition-interface progress:

```text
Work on the MAVLink/SITL adapter.
Do not train policies.
Implement or inspect telemetry parsing for attitude, local velocity, status flags, and simulator navigation reference data.
Define what the policy/controller should output and how it maps to MAVLink messages.
Add a small dry-run or test if possible.
```

### Work On Documentation

Use this when you want project memory updated:

```text
Update PRD.md with the latest run results.
Include checkpoint paths, eval artifact paths, metrics, what was learned, and whether the checkpoint is promotable.
Do not overwrite old lineage unless it is explicitly marked as regressed.
```

### Work On Quantization

Use this when you want edge inference work:

```text
Inspect the Q8 PufferNet path.
Do not promote quantized race control.
Compare float vs Q8 latency and action drift.
Explain what closed-loop metric would be needed before Q8 can be trusted.
```

### Ask For A Commit

Use this when you want changes saved to GitHub:

```text
Commit and push the current project changes on native-v4-drone-port.
Only include files related to [task].
Do not include unrelated local changes.
Use a concise commit message.
```

## Prompt Mistakes To Avoid

Avoid vague prompts like:

```text
Make it better.
```

Better:

```text
Improve R3 reliability.
Start from the current best R3 checkpoint.
Target crash reduction while preserving success_rate.
Run deterministic eval and update PRD.md with results.
```

Avoid mixing unrelated goals:

```text
Train R3, build vision, quantize the model, and make a submission.
```

Better:

```text
Focus only on R3 reliability.
Do not work on vision, quantization, or SITL unless needed to explain the blocker.
```

Avoid promoting from dashboard-only results:

```text
The dashboard looked good, use that checkpoint.
```

Better:

```text
Run deterministic JSON/CSV eval for the checkpoint before promoting it.
Use the PRD promotion thresholds.
```

## How To Read The Metrics

### `success_rate`

The fraction of eval episodes that completed the required gates.

Higher is better.

### `crash`

The fraction of eval episodes that crashed.

Lower is better.

### `gates_passed`

Average number of gates passed per episode.

For R3, the maximum is 3.

### `completion_time`

How long successful runs took.

This only matters after reliability is good.

### `timeout`

The fraction of episodes that neither succeeded nor crashed before the time limit.

### `crash_low`

The fraction of episodes that crashed by going too low.

This is currently the main R3 failure mode.

### `crash_xy`

The fraction of episodes that crashed by leaving horizontal bounds.

Current R3 issue is not mainly this.

### `crash_high`

The fraction of episodes that crashed by going too high.

Current R3 issue is not mainly this.

## Current Best Mental Model

The drone is no longer failing because it cannot find gates at all. It often gets most or all of the three-gate course. The main problem is that a meaningful fraction of runs end with a downward dive below the crash floor.

This suggests the next good work is not simply "make the reward bigger" or "lower the crash floor." The next good work is to understand why the policy enters that dive:

- Does it dive after a specific gate?
- Does it dive during turns?
- Does it dive when trying to recover from speed or angle errors?
- Does the observation lack a useful altitude/vertical-velocity cue?
- Is the reward encouraging progress at the expense of safe altitude?
- Is recurrent state reset/eval behavior consistent?

The best prompts should ask the agent to answer one of those questions with evidence.

## Suggested Next Technical Prompt

If you want to continue from here, a strong next prompt is:

```text
Continue R3 reliability debugging.
Use the current best R3 checkpoint from PRD.md.
Do not train R4.
Investigate why failures become low-altitude dives.
Add diagnostics or a trace that identifies when in the three-gate route the low crash happens.
Run deterministic eval before and after any change.
Update PRD.md with evidence and do not promote unless success_rate >= 0.85 and crash <= 0.10.
```

## Source Of Truth Files

Important files to mention in prompts:

- `PRD.md`: project status and promotion rules.
- `ocean/drone_race/drone_race.c`: native race environment logic.
- `ocean/drone_race/drone_race.h`: native race data structures and metrics.
- `ocean/drone_race/binding.c`: exposes env config and metrics to Python/PufferLib.
- `config/drone_race.ini`: default race environment/training config.
- `scripts/train_drone_race_curriculum.sh`: staged H1/R1/R3/R4 training script.
- `scripts/eval_drone_race_checkpoint.py`: deterministic checkpoint eval report generator.
- `docs/gpu_setup_ubuntu_22_04.md`: GPU/Vast setup notes.

## Current Branch

The active branch is:

`native-v4-drone-port`

This branch is the GitHub default branch for the project.