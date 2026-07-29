# VQ2 Informed-Dreamer Puffer Goal Prompt — historical

> Supersession note, 2026-07-28: the user retired this document's Dreamer-only,
> emergent-course-phase, and no-classical-label training requirements. Do not
> execute its training or next-action instructions. Preserve it as the detailed
> record of the N523--N735 research lineage and its runtime-legality boundary.
> The canonical active objective is now
> `docs/vq2_paper_guided_puffer_goal_prompt.md`.

Work autonomously in `/root/pufferlib-drone` toward a competition-legal,
fully autonomous, repeatable valid lap in the official **VQ2 Training** event
of:

`C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe`

After obtaining a reliable valid finish, minimize the official elapsed race
time without sacrificing ordered gate completion, collision avoidance, stream
health, or command-rate compliance. Never select or control VQ2 Submission
without explicit user authorization.

## Required mechanism: emergent end-to-end behavior

The deployed controller must be a learned recurrent **PufferLib visual policy**
trained with an **Informed Dreamer / DreamerV3-style model-based RL** pipeline.
PufferLib supplies the high-throughput environment, rollout collection,
evaluation, and deployable policy integration; this requirement does not mean
falling back to Puffer PPO. Perception, latent state and parameter estimation,
course-phase memory, trajectory choice, and control behavior must emerge from
world-model and actor learning. Do not solve VQ2 with a classical controller
hidden behind a learned action head.

The deployed actor may consume only causal, competition-available signals:

- pixel-level camera input derived from the current and previous official
  `640x360` JPEG frames; raw RGB, grayscale, or a learned pixel-level
  segmentation/depth-like representation is allowed;
- measured body rates from the gyro;
- measured actuator or motor-speed feedback if the live interface publishes
  it reliably;
- the actor's previous action and recurrent hidden state;
- causal camera timing such as new-frame validity, frame age, and elapsed step
  time, needed because the official camera and command loops run at different
  rates.

Do not provide the deployed actor with hand-engineered gate corners, gate pose,
range, bearing, a detector confidence, a reconstructed attitude quaternion,
position or velocity estimates, calibrated gate vectors, mapped flight-plan
vectors, course coordinates, kinematic predictor state, classical-controller
actions, or action-level overrides. In particular, do not copy SkyDreamer's
runtime flight-plan feedback subsystem. Do not provide official
`active_gate_index` as an actor input; let the recurrent visual latent state
learn to remember course phase. Official race status remains the sole judge for
reward accounting, ordered-gate validation, stopping, and finish authority.

Every deployed action must be one complete neural-policy output. No classical
controller may emit, blend, clip individual channels, schedule a corrective
action, or take over during uncertainty. Fixed conversion from normalized
network outputs to the chosen legal wire message is allowed.

## Action layer

Prefer direct neural motor/actuator commands if and only if the installed VQ2
runtime is proven to accept `SET_ACTUATOR_CONTROL_TARGET` legally and with
measured, repeatable semantics. The bundled v3391 example mentions this message,
but the public TS-002 table lists only `SET_ATTITUDE_TARGET` and
`SET_POSITION_TARGET_LOCAL_NED`; do not infer actuation from stale code or from
an output-status stream.

If direct actuator input is not proven, emit the complete collective-thrust and
body-rate vector through the supported `SET_ATTITUDE_TARGET` interface. This is
still an end-to-end actor: no external perception, planner, state estimator, or
outer-loop controller may shape its four outputs. Stream commands at a measured
rate in `[50,100) Hz` and maintain heartbeat at `>=2 Hz`.

## Training formulation: Informed Dreamer is primary

Treat training as an informed POMDP. Privileged native state, gate geometry,
camera extrinsics, dynamics parameters, collision distances, and progress may
exist **only during training** for:

- reward and termination;
- world-model decoder targets and interpretability diagnostics;
- reward and continuation prediction;
- critic/value targets and offline evaluation.

They must never enter the deployed actor observation or a runtime preprocessing
state. The privileged decoder may estimate pose, velocity,
current-gate-relative state, course phase, camera extrinsics, and dynamics
parameters so the latent is interpretable, but those estimates are diagnostic
outputs—not runtime actor inputs or control signals. Add automated dependency
and intervention tests that fail if changing any privileged value can change
an actor action while the legal observation/history is held fixed.

Build a high-throughput visual training environment around the native PufferLib
drone dynamics. Start with a learned gate segmentation representation resized
to `64x64`, matching the SkyDreamer reference design, while retaining a path to
train directly from standardized RGB if segmentation becomes the bottleneck.
Render from the same camera convention as VQ2, then domain-randomize textures,
illumination, JPEG artifacts, false-positive/false-negative mask regions, mask
thickness/erosion, occlusion, rolling-shutter shear, gate appearance, camera
extrinsics, motor/dynamics parameters, image/action latency, camera cadence,
repeated frames, and dropped frames. Standardize camera intrinsics before the
actor where the official calibration permits. The renderer may use privileged
state to generate pixels; the deployed policy may see only the pixels and
causal onboard signals.

Implement the recurrent state-space model as a CNN observation encoder,
discrete stochastic latent, single-layer GRU sequence model, latent dynamics
predictor, privileged-information decoder, reward predictor, and continuation
predictor. Alternate off-policy environment collection, replayed world-model
learning, and actor-critic learning through latent imagination. Begin from the
SkyDreamer/DreamerV3 reference values—16 imagined steps and discount `0.997`—
but tune them only from held-out evidence for VQ2's lower camera cadence.

The actor is a Gaussian policy over all four low-level action channels and the
critic operates on the same latent state. Apply smoothness regularization to
the policy mean, beginning with SkyDreamer's `0.002` weight, while retaining
exploration variance during training; deploy the deterministic mean. No MPC or
iterative optimizer may run at inference. The deployed artifact consists only
of image/sensor encoders, the posterior recurrent sequence update, and actor,
and must run comfortably above the required command cadence on the actual
Windows Python environment.

Use objective-aligned training-only rewards: progress toward the next ordered
gate, accurately centered gate passage, valid completion, elapsed time/body
rate cost, and return truncation on collision or invalid state. Include no
explicit perception or look-at-gate reward: camera orientation and visual
attention must emerge because they improve return. Do not use action imitation,
a classical teacher, controller-shaped targets, a prescribed spline, or dense
terms that encode a particular flight path.

Model the asynchronous VQ2 interface explicitly. The official camera is
nominally about `30 Hz` while legal commands must remain in `[50,100) Hz`.
Collect gyro and reliable motor/actuator feedback at every policy step, reuse
the most recent image only with its causal age/freshness indicators, randomize
sensor timing during training, and make the recurrent latent predict through
inter-frame intervals. Never fabricate a new visual observation by consulting
privileged state at deployment.

## Current evidence and closed lineage

Preserve N294/N295 and later artifacts as historical evidence. N522
`vq2_n522_corrected_geometry_gate2_bounded_004` ran exactly once and is rejected
without unchanged retry: it passed official Gate 1, remained at official index
`1`, and collided with object `1002` before Gate 2. Its explicit corrected
Gate-2 vector, detector-derived observation, and kinematic predictor lineage is
now closed by user direction. Do not tune another vector, predictor, detector,
teacher blend, or phase-specific checkpoint from that lineage.

No further FlightSim actuation is authorized until the emergent visual policy
has passed source-locked offline evaluation, held-out visual/dynamics
randomization, deployment-rate benchmarks on actual Windows Python, and a
zero-control live shadow in visibly selected VQ2 Training.

## Promotion and evidence

Promote through increasingly strict, separately preregistered stages:

1. Native full-course discovery with Informed Dreamer and no classical action
   teacher; require success from the actor learned through latent imagination,
   not from a PPO or scripted bootstrap controller.
2. Held-out visual, camera-extrinsic, latency, and dynamics randomization with
   zero runtime privilege and zero action overrides.
3. Exact actor export and Linux/Windows action parity.
4. Active-VQ2-Training zero-control shadow proving camera/body-rate/actuator
   inputs, recurrent inference cadence, hashes, and zero flight commands.
5. One bounded official Gate-1 reproduction, then Gate-2 and later ordered-gate
   milestones. Never retry a failed candidate unchanged.
6. A full valid Training lap: all live-discovered ordered gates passed and
   nonnegative `race_finish_time_ns`, with no collision, invalid state,
   dropout, drain-limit failure, or command-rate violation.
7. Time optimization only after repeatable valid finishes, using learned-policy
   training and objective-aligned rewards rather than classical runtime tuning.

For every live attempt, record exact source/model hashes, reset detection,
arm/control/disarm counts, effective command rate, stream health, official gate
transitions, collision/invalid state, finish timestamp, elapsed time, and a
command-free passive post-run disarm proof. Abort immediately on collision or
invalid state. Preserve the responsive simulator process and treat runtime
packets—not comments or assumptions—as authoritative.

Keep `AGENTS.md`, `PRD.md`, the competitive run ledger, tests, and this prompt
current. Continue safe offline work autonomously until a genuinely emergent
Puffer policy earns the next bounded VQ2 Training interaction.

## Reference and adaptation boundary

Use Verraest et al., *SkyDreamer: Interpretable End-to-End Vision-Based Drone
Racing with Model-Based Reinforcement Learning* (arXiv:2510.14783v1, CC BY
4.0), together with the official Informed Dreamer implementation, as the
primary architectural references. Reproduce the informed RSSM and latent
imagination principles, not assumptions that conflict with the measured VQ2
wire contract. The mapped runtime flight-plan vector in SkyDreamer is
intentionally excluded here so that VQ2 course-phase behavior remains learned
and recurrent.

The full paper extraction, figure/table scan, training schedule, limitations,
and VQ2 adaptation matrix are documented in
`docs/skydreamer_paper_review_2026-07-27.md`. In particular, do not conflate
the paper's replay context `16`, world-model sequence length `64 -> 256`, and
imagination horizon `16`, and do not cite SkyDreamer's results as evidence that
course phase can emerge without its mapped flight-plan/runtime-threshold
subsystem.

- Geles et al.: <https://doi.org/10.15607/RSS.2024.XX.082>
- Paper: <https://arxiv.org/abs/2510.14783>
- Informed Dreamer: <https://github.com/glambrechts/informed-dreamer>
