# SkyDreamer paper review and VQ2 adaptation notes — 2026-07-27

## Source and review scope

Primary source:

- Aderik Verraest, Stavrow Bahnam, Robin Ferede, Guido de Croon, and
  Christophe De Wagter, *SkyDreamer: Interpretable End-to-End Vision-Based
  Drone Racing with Model-Based Reinforcement Learning*, arXiv:2510.14783v1,
  submitted 2025-10-16.
- [Versioned HTML](https://arxiv.org/html/2510.14783v1)
- [Versioned PDF](https://arxiv.org/pdf/2510.14783v1)
- [Abstract and license](https://arxiv.org/abs/2510.14783v1)
- [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

This note is based on an end-to-end reading of the v1 HTML and a visual scan of
all 17 PDF pages. The scan explicitly covered the architecture diagram,
coordinate figure, trajectory and state-estimation plots, randomization and
results tables, and both implementation appendices. Statements labeled
"paper" below report the authors' method or results. Statements labeled
"VQ2 decision" or "inference" are repository-specific interpretations and
must not be attributed to the authors.

The paper is a strong architectural reference, but it is not a drop-in VQ2
controller specification. In particular, the published policy receives a
mapped flight-plan vector at runtime and advances it using a decoded
gate-relative state threshold. The active VQ2 goal explicitly forbids both
inputs. SkyDreamer's real-world results therefore do not prove that a policy
can infer ordered course phase from vision and recurrent memory alone.

### PDF visual-scan index

The PDF scan added information that is easy to underweight when reading the
linearized HTML alone:

| PDF page | Item | Review takeaway |
|---:|---|---|
| 1 | Figure 1 | The actor sees a visibly imperfect gate mask and ultimately emits low-level motor commands onboard. |
| 2 | Table I | Comparative claims are author-reported and the visual-ambiguity claim is tied to progress tracking. |
| 4 | Figure 2 | World, gate, body, and camera frames all use NED conventions but are not generally aligned. |
| 5 | reward/dynamics equations | Centered gate reward, rate cost, virtual gate thickness, and shaped ground termination are separate mechanisms. |
| 6 | Table II and visual augmentation | The physical-drone coefficients sit beside the intrinsic standardization, GAN, erosion, and rolling-shutter details. |
| 7 | Figure 3 | Deployment visibly feeds decoded state back into the flight-plan update; actor imagination is a separate graph. |
| 9 | Table III | Camera, dynamics, disturbances, aperture, and initial-state ranges are all randomized. |
| 10 | Figure 4 | Camera-axis arrows point toward gates without a perception reward, but the trajectory also uses mapped flight-plan input. |
| 11 | Figure 5 | Available-thrust parameters are identifiable; drag/yaw parameters are weak and several estimates drift. |
| 12 | Figures 6-7 | Real trajectories remain coherent under incomplete, rounded, and false-positive masks. |
| 13 | Figure 8 and Table IV | Battery adaptation is supported by RPM-estimate traces; the 100% small-track table has stated exclusions/fixes. |
| 14 | Figure 9 | The large track combines high speed and complex maneuvers, but includes minor contacts and a distribution-bound failure. |
| 16-17 | Appendices A-B | GateNet and StochGAN are concrete deployed/training components, not unspecified image preprocessing. |

Four attributed 120-DPI page renders are retained under
[`docs/assets/skydreamer_v1`](assets/skydreamer_v1/README.md):

- [Figure 3 architecture and data flow](assets/skydreamer_v1/page07_architecture.png);
- [Table III randomization and schedule](assets/skydreamer_v1/page09_randomization.png);
- [Figure 5 estimator convergence/drift](assets/skydreamer_v1/page11_estimation.png); and
- [Appendix A GateNet details](assets/skydreamer_v1/page16_gatenet.png).

They are review evidence, not repository training data.

## What the paper claims

SkyDreamer is presented as an onboard vision-based racing policy that maps a
pixel-level gate representation, measured body rates, and measured motor RPMs
to four normalized motor commands. It uses an Informed-Dreamer world model to
decode training-only state and system parameters. The decoded quantities make
the learned latent state inspectable and allow the authors to treat the world
model as an implicit state and parameter estimator.

The paper's claimed differentiators are:

- direct neural motor control rather than collective-thrust/body-rate control
  followed by an inner PID or INDI loop;
- onboard inference without motion-capture or offboard computation;
- non-trivial visual sim-to-real transfer through a learned gate segmentation
  representation and visual corruptions;
- reconstruction of state, camera extrinsics, and selected dynamics parameters
  from recurrent history;
- high-speed real-world flight, including inverted-loop, split-S, ladder, and
  larger multi-gate maneuvers;
- deterministic deployment of a stochastic Gaussian actor trained in latent
  imagination; and
- online adaptation to reduced motor capability during battery depletion.

The comparison in paper Table I should be read as the authors' framing, not an
independent benchmark. Its "robust to visual ambiguity" entry depends on the
explicit flight-plan subsystem described below.

## Problem formulation

### Informed POMDP

The paper formulates training as an informed partially observable Markov
decision process. The physical state is never directly available to the
execution policy. Training additionally exposes an information variable that
contains the legal observation plus privileged observations. Execution has
only the causal history of legal observations and actions.

In compact form, the training problem includes:

\[
\widetilde{\mathcal P}=(\mathcal S,\mathcal U,\mathcal I,\mathcal O,
T,R,\widetilde{\mathcal I},\widetilde{\mathcal O},P,\gamma),
\]

while the execution POMDP omits the privileged information space. The policy
is optimized for expected return in the execution POMDP, not for access to the
training-only variables.

This distinction is central to the paper. Privileged information supervises
the world-model decoder; it is not concatenated into the deployed actor input.
The VQ2 privilege-intervention tests are therefore a faithful and necessary
adaptation of the paper's informed-POMDP boundary.

### State, legal observation, privileged target, and action

The paper's simulator state contains:

- world- and current-gate-frame position, `p_w` and `p_g`;
- world- and current-gate-frame velocity, `v_w` and `v_g`;
- roll, pitch, world yaw, and gate-frame yaw;
- true body rates and four true propeller speeds;
- three camera-extrinsic Euler angles;
- randomized dynamics parameters;
- acceleration, moment, and action disturbances; and
- a flight-plan vector describing mapped gate relationships.

The legal sensor observation is

\[
o_t=[X,\hat\Omega,\hat\omega]^T,
\]

where `X` is a binary gate segmentation mask, `hat(Omega)` is the measured
three-axis body rate, and `hat(omega)` is the measured four-motor RPM. The
accelerometer is deliberately omitted because the authors' physical platform
can saturate it under vibration and high acceleration.

The decoder information target includes the measured body-rate/RPM scalars plus
privileged position, gate-relative position, velocity, orientation, true body
rates, true motor speeds, camera extrinsics, and dynamics parameters. The
rendered mask itself is excluded because its information is already represented
by those targets and reconstructing the image would slow training. The current
VQ2 implementation's 34-value auxiliary head reconstructs only its privileged
tail, so it is a deliberate compact adaptation rather than an exact copy of
the paper's information decoder.

The paper's action is four normalized motor power fractions in `[0,1]`. The
electronic speed controller converts these fractions to PWM without an
intermediate attitude-rate controller.

Paper detail: the printed disturbance vector repeats `epsilon_u,1` where the
second entry appears intended to be `epsilon_u,2`. Treat this as a likely
typographical error; do not copy it into an implementation silently.

## Objective and termination

The paper uses three reward terms at a `90 Hz` control rate:

\[
r_t=5r_{prog}-r_{rate}+30r_{gate},
\]

\[
r_{prog}=\lVert p_{t-1,g}\rVert_2-\lVert p_{t,g}\rVert_2,
\]

\[
r_{rate}=\frac{\exp(\min(\lVert\Omega_t\rVert_1,17))-1}
{2f_c10^5},\quad f_c=90\,\mathrm{Hz},
\]

and, on a gate pass,

\[
r_{gate}=1-\frac{\max(|y_{t,g}|,|z_{t,g}|)}{d_g}.
\]

The gate term therefore follows the square gate geometry through an infinity
norm, with maximum reward at the center and zero at an effective edge. The
dense progress reward is the reduction in Euclidean range to the next
pre-gate. It is set to zero while the vehicle is between the pre- and
post-gate.

The authors add invisible pre- and post-gates to represent gate thickness and
vehicle extent. Each receives the same centered pass reward. There is no
explicit camera-alignment or look-at-gate reward. The authors report that
camera orientation toward the next gate emerges from the task return.

There is also no separate negative gate-collision reward. A collision instead
terminates the episode, truncating its discounted return. A ground-contact
termination is shaped to trigger near the ground under high downward velocity
or large roll/pitch. The authors report that this teaches the drone to climb
before accelerating from its raised podium.

VQ2 decision:

- retain objective-aligned ordered progress, centered pass reward, body-rate
  cost, and immediate termination;
- adapt the gate geometry and pass authority to the official VQ2 contract;
- retain the ban on an explicit perception reward;
- do not import invisible maneuver gates used later in the paper to force a
  split-S or other prescribed path; and
- abort immediately on any official collision. The paper's later ability to
  continue after physical gate contact is not legal acceptance evidence here.

## Dynamics model

The simulator uses a quaternion attitude, gyroscopic coupling, first-order
motor response, rotor-specific roll/pitch/yaw effectiveness, aerodynamic drag,
and injected acceleration, angular-acceleration, and action disturbances. It
is integrated in JAX with fourth-order Runge-Kutta at `2.2 ms`.

The motor model has the form

\[
\dot\omega_i=(\omega_{ci}-\omega_i)/\tau,
\]

with a nonlinear normalized-command-to-steady-RPM curve between
`omega_min` and `omega_max`. Action disturbance is added before clipping the
normalized command to `[0,1]`.

Selected nominal paper parameters from Table II are:

| Parameter | Paper value | Parameter | Paper value |
|---|---:|---|---:|
| thrust effectiveness `k_w` | `1.55e-6` | quadratic x drag `k_x2` | `4.10e-3` |
| rotor-speed x/y drag `k_x`, `k_y` | `5.37e-5` | quadratic y drag `k_y2` | `1.51e-2` |
| angle-of-attack factor | `3.145` | horizontal inflow factor | `7.245` |
| quadratic vertical drag | `0.0` | x gyroscopic coefficient | `-0.89` |
| y gyroscopic coefficient | `0.96` | z gyroscopic coefficient | `-0.34` |
| `omega_min` | `341.75 rad/s` | `omega_max` | `3100 rad/s` |
| command-curve coefficient | `0.50` | motor time constant | `0.03 s` |

The full table also gives rotor-specific moment coefficients. These values
describe the paper's physical drone and are not VQ2 plant constants. The useful
transfer is the *parameterization and randomization strategy*, especially
available thrust, motor lag, rotor asymmetry, drag, and actuator disturbance.

## Pixel representation and visual transfer

### Intrinsic standardization

SkyDreamer calibrates camera intrinsics but randomizes extrinsics. RGB frames
are undistorted, cropped, and resampled to a shared pinhole model. At `64x64`,
the nominal focal length is `25 px` and the principal point is `(32,32)`:

\[
K=\begin{bmatrix}25&0&32\\0&25&32\\0&0&1\end{bmatrix}.
\]

The paper treats intrinsic calibration as stable and easy to repeat, while
extrinsic calibration can change after handling or contact. Randomizing
extrinsics lets the recurrent latent infer them online.

VQ2 adaptation: standardize the official `640x360`, `fx=fy=320`,
`cx=320`, `cy=180` stream explicitly, but center extrinsic randomization on
the measured VQ2 camera convention rather than the paper's approximately
`50 deg` camera pitch. Keep the measured camera optical uptilt distinct from
the IMU mount pitch.

### GateNet

The deployed paper system does not feed raw RGB directly to SkyDreamer. A
separate U-Net-style segmenter, GateNet, produces a binary gate mask that is
resized to `64x64`.

Appendix A specifies:

- an encoder-decoder with additive skip connections;
- double-convolution blocks using batch normalization and ReLU;
- five supervised output resolutions;
- Dice plus twice-weighted binary cross-entropy at each output;
- total output weights `4, 2, 1, 1, 1` from highest to lower priority;
- Xavier-uniform convolution initialization;
- stronger shot noise because the physical camera uses `1 ms` exposure;
- a `196x196`, channel-scale-2 model for MAVLab gates; and
- a `384x384`, channel-scale-4 model for orange gates.

The orange-gate segmenter used 200 manually labeled real images. The harder
MAVLab-gate segmenter used 700 manually labeled real images plus 8,500 synthetic
examples. GateNet is therefore a learned pixel-level front end, but it is still
a separate deployed component with its own data, failure modes, runtime, and
hashing requirements.

### Mask randomization and StochGAN

Ideal masks are rendered in batches with PyTorch3D. The paper then applies a
stochastic CycleGAN trained on about 4,000 unpaired rendered and real masks.
No paired labels are needed for this translation model.

Appendix B describes a `64x64` StochGAN with:

- generator encoder `c7s1-32, d64, d128`;
- six `R128` residual blocks;
- decoder `u64, u32, c7s1-1-tanh`;
- one additional input noise channel sampled uniformly from `[-1,1]`;
- PatchGAN discriminator `C32, C64, C128, C256, conv4-1`; and
- Adam with `beta1=0.5`, `beta2=0.999`, learning rate `1.5e-4`.

Additional online training corruptions are important:

- erode the `64x64` mask by one pixel with `50%` probability;
- hold the erosion mode for an average of 100 environment steps so corruption
  has realistic temporal persistence; and
- approximate rolling shutter with a randomized parameter `s in [0,0.02]`,
  horizontal shear proportional to camera-frame yaw rate, and vertical scale
  proportional to camera-frame pitch rate.

The temporal persistence point is easy to miss and matters for recurrent
training: independent per-frame corruption is not equivalent to the slowly
varying perception bias used by the paper.

## World model and policy

### Data flow

Figure 3 is the most important architecture figure in the paper. Its four
panels distinguish:

1. standard DreamerV3, whose decoder reconstructs ordinary observations;
2. SkyDreamer deployment, where each real observation updates the recurrent
   posterior and the dynamics prior is not used for the action step;
3. world-model learning, where encoder posterior and dynamics prior are
   matched and the privileged, reward, and continue heads are trained; and
4. actor-critic learning, where the dynamics prior generates imagined latent
   trajectories without environment observations.

The recurrent state-space model is summarized as:

\[
z_t\sim q^e_\theta(\cdot\mid h_t,o_t),
\]

\[
h_t=f^e_\theta(h_{t-1},z_{t-1},u_{t-1}),
\]

\[
\hat z_t\sim p^d_\theta(\cdot\mid h_t).
\]

The posterior latent `z_t` is discrete and stochastic; image input is encoded
with a CNN. The deterministic sequence state `h_t` is produced by a
single-layer GRU. The dynamics predictor produces the latent prior used for
imagination.

Instead of reconstructing the mask, the decoder predicts the privileged
information from `(h_t,z_t)`. Separate heads predict reward and continuation.
The authors argue that state/parameter reconstruction is both a learning
scaffold and a debugging interface because recorded sensor/action histories
can be replayed through a revised world model without another physical flight.

### Actor-critic details

The actor is Gaussian over all four motor channels. The critic estimates the
discounted return from the same latent. The paper uses:

- discount `gamma=0.997`;
- REINFORCE actor estimation;
- 16 imagined steps, corresponding to about `0.18 s` at `90 Hz`;
- a slow target critic in the reported experimental configuration;
- entropy during training; and
- deterministic mean action at evaluation and deployment.

To avoid near bang-bang motor policies, the paper adds

\[
L_{smooth}=0.002\,E_t[\lVert\mu_t-\mu_{t-1}\rVert_2^2]
\]

to the actor loss during imagined rollouts. Only the policy mean is
regularized. Penalizing samples from the full stochastic distribution would
conflict with the entropy objective.

### Three sequence lengths that must not be conflated

The paper uses three different temporal quantities:

- `replay_context=16`: replay context/burn-in;
- world-model batch length `64`, increased to `256` after 8 million steps; and
- imagination horizon `16`.

They serve different purposes. A 16-step imagination horizon does **not** imply
that 16-step world-model training sequences are faithful to the paper. The
authors explicitly lengthen world-model sequences to improve long-term
dependency and parameter identification.

### Pinned DreamerV3 implementation audit

The paper pins DreamerV3 commit
`cdf570902b1eaba193cc8ef69426cd4edde1b0bc`. A direct source audit adds several
details that the prose does not enumerate:

- the reward head is `symexp_twohot` with 255 bins and zero output scale;
- the value head uses the same 255-bin `symexp_twohot` distribution;
- default world-model loss scales are dynamics `1.0`, representation `0.1`,
  and reward `1.0`;
- default categorical KL free nats are `1.0`;
- `reward_grad=true` enables actor learning from imagined reward;
- `size12m` uses deterministic state 2,048, hidden width 256, and 16
  categorical classes in the pinned configuration; and
- actor imagination starts are not limited to the final replay state.

The last point is operationally important. In the pinned `agent.py`,
`K=min(imag_last or T,T)` and `starts=self.dyn.starts(...)` are reshaped to
`B*K`. The default configuration has `imag_last=0`, batch size 16, and batch
length 64, so actor learning begins from all `16*64=1,024` posterior time
states. A hardware-bounded VQ2 adaptation may sample a fixed subset of those
states, but using only the 16 sequence endpoints is not faithful to the pinned
algorithm.

This source audit does **not** imply that every replay state is useful. The
paper continually collects with the current policy, so replay becomes
task-relevant. Applying all-time starts to an old random-policy replay changes
the training distribution and needs its own admission evidence.

## Flight-plan subsystem: paper fact and VQ2 exclusion

SkyDreamer does not solve course ambiguity from pixels and unconstrained
recurrent memory alone. At runtime the actor also receives a mapped flight-plan
vector containing absolute positions/yaws and relative position/yaw changes
for the current and next gates. The paper describes three upcoming gates in
this vector.

The active flight-plan index is advanced using the world model's decoded
gate-frame forward position. The published threshold is
`estimated_x_g > -0.15 m`; it was relaxed from `0 m` after an aborted
real-world flight. During early training, when decoded state is inaccurate, the
index is randomized between the pre- and post-gate instead.

Figure 3(b) makes the runtime feedback explicit: a cyan arrow runs from decoded
state into the flight-plan update, and the resulting plan returns as an encoder
input. This is a deterministic runtime phase mechanism coupled to a mapped
course representation.

VQ2 decision: exclude the entire subsystem. The actor must not receive mapped
gate vectors, absolute or relative gate coordinates, official gate index, or a
decoded-state-driven phase update. Official progress is reward/judge/stop
evidence only. Consequently, VQ2 must separately prove that its recurrent
latent distinguishes observation histories that look locally similar but
require different actions.

## Training protocol

### Small-track training

The paper starts from the
[pinned JAX DreamerV3 implementation](https://github.com/danijar/dreamerv3/commit/cdf570902b1eaba193cc8ef69426cd4edde1b0bc)
and the standard `size12m` model. Unless overridden, it uses DreamerV3 defaults.
The paper does not enumerate every layer width, stochastic-latent size,
optimizer setting, or loss coefficient; those must not be invented from the
high-level description.

Reported settings are:

- 17 million environment steps;
- replay buffer capacity `10,000,000`, described as 31 hours of flight;
- `replay_context=16`;
- `train_ratio=128`;
- slow target critic enabled;
- 70% of training initializations in front of any gate;
- 30% of initializations at the evaluation/start gate;
- approximately 50 hours on a 40 GB, 56-SM partition of an A100 80 GB GPU.

Training has three phases:

1. DreamerV3 defaults during warm-up;
2. after 8 million steps, world-model batch length increases from 64 to 256;
3. after 13 million steps, entropy coefficient falls from `3e-4` to `1e-5`
   and learning rate falls from `4e-5` to `2e-6` for smoother fine control.

The paper also adds invisible gates during training to enforce maneuvers such
as the split-S. That technique is not part of the VQ2 adaptation because it
encodes a selected trajectory rather than only the ordered visible-gate
objective.

### Domain randomization from Table III

| Quantity | Training | Evaluation |
|---|---:|---:|
| camera roll/yaw extrinsics | `[-5,5] deg` | `[-5,5] deg` |
| camera pitch extrinsic | `[45,55] deg` | `[45,55] deg` |
| min/max motor RPM | `+/-20%` | `+/-20%` |
| other dynamics parameters | `+/-30%` | `+/-20%` |
| acceleration disturbance, slow | `[-3,3] m/s^2` | `[-2,2] m/s^2` |
| moment disturbance, slow | `[-3,3] rad/s^2` | `[-2,2] rad/s^2` |
| moment disturbance, every step | `[-125,125] rad/s^2` | `[-100,100] rad/s^2` |
| action disturbance, every step | `[-0.2,0.2]` | `[-0.2,0.2]` |
| effective gate half-size `d_g` | `0.8 m` | `1.0 m` |
| virtual gate thickness `t_g` | `0.8 m` | `0.8 m` |

For a nominal `90 Hz` loop, the paper calls a disturbance "1 Hz" when it has a
`1/100` resampling probability per policy step; it calls a disturbance "90 Hz"
when it changes every step.

Initial states use gate-forward range `[-4,-2] m`, lateral range `[-1,1] m`,
vertical range `[0,1.3] m` in training and `[0.7,1.3] m` in evaluation, zero
linear velocity, body rates in `[-0.1,0.1] rad/s`, rotor speeds at
`[0.25,0.5] omega_max`, and roll/pitch/gate-yaw error in
`[-pi/9,pi/9]`.

### Big-track changes

The large-track policy is not the same training recipe:

- 35 million environment steps;
- `train_ratio=64`;
- symlog disabled in encoder and decoder;
- phase transitions at 23 million and 31 million steps;
- virtual gate thickness `0.5 m`; and
- an additional `+/-300 rad/s` disturbance on `omega_max`, resampled every
  10 steps.

This is evidence that the paper's recipe required track-scale-specific tuning;
it is not evidence for a universal unchanged configuration.

## Deployment and timing

The physical deployment path is JAX checkpoint to custom PyTorch conversion,
then ONNX, TensorRT, and execution on a Jetson Orin NX 16 GB. Reported average
runtimes are `1.3 ms` for encoder + sequence model + actor and `3 ms` for
GateNet.

The physical camera is a rolling-shutter Arducam IMX219 with a `175 deg` field
of view at `90 Hz`. Camera timestamps anchor the policy loop. Training models
`33 ms` image delay and `11 ms` action delay. Motor commands are emitted with a
desired execution timestamp, and the runtime buffers work to absorb occasional
segmentation/policy timing variability.

VQ2 adaptation:

- VQ2 has approximately `30 Hz` unique camera frames and a `50-100 Hz` command
  requirement, so it cannot copy the paper's one-new-frame-per-action loop;
- held frames need explicit causal freshness/age and the latent must propagate
  across inter-frame action steps;
- timing randomization must include duplicate frames, dropped frames, JPEG
  assembly delay, action delay, and control-loop jitter;
- actual Windows inference cadence and cross-platform parity remain promotion
  gates; and
- direct motors remain unauthorized unless the official interface is proven
  legal and actuated without unsafe experimentation. Otherwise the actor must
  emit the complete supported body-rate/thrust vector.

## Results and evidence caveats

### Simulation

On the simulated ladder inverted loop, the paper reports approximately
`13 m/s`, `6 g`, a roughly `1.5 m` loop radius, and a trajectory mostly within
`1 m` of the ladder gate. Figure 4 shows camera-axis arrows that generally face
the gates despite no perception reward.

Figure 5 aggregates 20 successful, differently initialized, 2,000-step
simulated flights. The useful result is selective identifiability:

- maximum motor RPM and thrust effectiveness converge quickly and remain
  comparatively stable;
- motor response is estimated well;
- individual rotor roll/pitch effects are moderate;
- drag and yaw-effectiveness estimates are weak;
- camera extrinsics converge to roughly `1 deg` standard deviation;
- position converges within about 10 steps to a `10-15 cm` spread;
- velocity error is about `0.5 m/s`; and
- some estimates drift when 256-step training sequences are used over
  2,000-step flights.

The authors report parameter convergence in 50-100 steps (`0.6-1.1 s`), but
also report material variation between training runs. Decoder accuracy is
therefore a diagnostic metric, not a guaranteed calibrated estimator.

### Small real-world tracks

The paper reports 75 completed laps: five flights of five laps on each of three
track/gate combinations. Table IV reports 100% success for the included runs,
with mean later-lap times of `3.25 s`, `3.62 s`, and `2.97 s` respectively.
Peak acceleration is measured at about `6 g`; peak speed is an estimate of
about `13 m/s`.

Two qualifications matter:

- one ladder flight was excluded after its `x_g > 0` phase threshold failed;
  the threshold was then changed to `x_g > -0.15 m`; and
- a background feature consistently misclassified as a MAVLab gate was
  physically covered.

Figures 6 and 7 are nevertheless useful evidence that the recurrent policy can
tolerate rounded, incomplete, and false-positive segmentation regions when its
training corruption distribution covers them.

During a battery-depletion run, measured peak RPM fell from approximately
`3200` to `2200 rad/s`, outside the nominal training interval. Figure 8 shows
the decoded maximum-RPM estimate following that change while the vehicle
continued completing inverted loops. This motivates decoder targets for
available thrust and actuator health, but it does not justify feeding those
decoded estimates into a separate VQ2 controller.

### Big real-world track

Five of six two-lap flights succeeded. The failed drone had camera extrinsics
outside the training range; the five successful flights used one drone inside
the range. Three successful flights had audible minor gate contact. The paper
reports estimated speed up to `21 m/s` and acceleration near `6 g`.

For VQ2, a contact is an immediate invalidation/abort, so these runs demonstrate
recovery capability rather than collision-free acceptance reliability.

### Limitations acknowledged by the paper

- decoded parameters can drift over long horizons;
- estimate quality varies across training runs;
- decoded states can jump at individual timesteps, possibly because of the
  discrete latent or world-model error;
- the policy remains vulnerable to segmentation false positives;
- extrinsics outside the training distribution caused a real crash; and
- training cost was roughly 50 A100 hours for the 17-million-step recipe.

Additional VQ2 limitation: the paper does not ablate the mapped flight-plan
vector, decoded-state phase threshold, or invisible maneuver gates. It cannot
be cited as evidence that emergent course memory without those aids will work.

## Repository adaptation matrix

| Paper mechanism | VQ2 disposition | Reason |
|---|---|---|
| `64x64` pixel-level gate representation | Adapt | Current continuous edge mask is legal, but deployment must prove a causal official-frame front end and corruption coverage. |
| measured gyro and motor feedback | Adopt | Both are public VQ2 signals when runtime validity is established. |
| no accelerometer in actor | Adopt initially | Matches the paper and avoids dependence on uncertain high-acceleration specific force. |
| discrete stochastic RSSM + GRU | Adopt | Needed for partial observability and asynchronous images. |
| privileged state/parameter decoder | Adopt with strict tests | Training scaffold and diagnostic only; privilege intervention must leave actions exact. |
| reward and continuation heads | Adopt | Required for latent imagination. |
| 16-step imagination, `gamma=0.997` | Adopt as starting values | Tune only from held-out VQ2 evidence. |
| mean-only smoothness at `0.002` | Adopt as starting value | Preserves training entropy while discouraging bang-bang mean actions. |
| deterministic deployment mean | Adopt | Required for replay and promotion parity. |
| direct motor output | Exclude for current lineage | VQ2 public direct-motor legality/semantics are unproved; use one complete neural CTBR vector through the supported fixed decoder. |
| mapped three-gate flight-plan vector | Exclude | Violates the emergent visual actor contract. |
| decoded-state flight-plan threshold | Exclude | It is a runtime state estimator and deterministic phase controller. |
| official gate index as actor input | Exclude | Judge/reward/stop evidence only. |
| invisible maneuver gates | Exclude | They encode a chosen flight path rather than only the official objective. |
| intrinsic standardization | Adopt | Official intrinsics are known and stable. |
| camera-extrinsic randomization | Adapt | Center on measured VQ2 camera geometry, not the paper's physical camera. |
| persistent mask erosion and false regions | Adopt/adapt | Recurrent errors must be temporally correlated and fitted to passive VQ2 imagery. |
| rolling-shutter affine corruption | Adapt | VQ2 camera transport must determine the relevant magnitude; retain a bounded randomized model. |
| 90 Hz image/action synchronization | Exclude | VQ2 is approximately 30 Hz camera and 64 Hz control. |
| long world-model sequences | Adopt | Distinct from the 16-step imagination horizon and necessary for phase/parameter memory. |
| 70% gate-local / 30% full-start resets | Adapt | Useful discovery curriculum, but promotion must use uninterrupted full-course starts without actor phase input. |
| post-contact continuation | Exclude | Official collision is immediate-abort authority. |

## Historical implications for the N523-N528 line

These are repository inferences, not paper claims:

1. N528's `524,288` collected agent steps are only about 3% of the paper's
   17-million-step small-track run and about 1.5% of its 35-million-step
   big-track run. Its sparse training-time Gate-1 discovery, and the absence
   of a completed deterministic N528 evaluation, do not yet falsify the
   architecture.
2. N528 uses 16-step replay sequences. The paper uses replay context 16 but
   world-model sequence length 64, later 256. Matching only the 16-step
   imagination horizon leaves a major temporal-training mismatch.
3. N528's registered `train-ratio=1`, `train-every=4`, batch size 4, and
   sequence length 16 replay roughly 64 records per 256 newly collected agent
   transitions, or about `0.25` replay records per transition before framework
   accounting. The paper reports `train_ratio=128`. The definitions are not
   mechanically interchangeable, but the current update density is plainly a
   smoke/discovery budget rather than a paper-scale run.
4. The paper's local-gate reset mixture makes early maneuver discovery easier,
   while its flight-plan vector removes much of the ordered-phase memory
   problem. VQ2 needs local discovery curricula plus separate uninterrupted
   screens that prove latent phase memory without leaking gate index.
5. N528's normal-versus-blank/flip counterfactual is important. The paper shows
   successful visually robust flight but does not publish an action-dependence
   ablation that holds gyro/RPM/flight-plan history fixed while removing the
   mask.
6. Before scaling training, the VQ2 trainer should report separate replay
   context, world-model sequence length, imagination horizon, collected
   transitions, replayed sequence records, and actor/world update counts. A
   single generic "steps" number is too ambiguous for comparison.

## Paper-derived evidence checklist

Before treating a future VQ2 checkpoint as a serious SkyDreamer-style
candidate, require offline evidence for:

- action sensitivity to causal visual content under fixed sensor/action
  history;
- exact invariance to every privileged-target intervention;
- posterior/prior, reward, and continuation calibration over held-out
  sequences;
- decoder error versus time for pose, velocity, body rates, available thrust,
  motor lag, camera extrinsics, and ordered phase;
- drift screens substantially longer than the world-model training sequence;
- persistent false-positive, false-negative, erosion, dropout, and frame-hold
  corruptions;
- camera-extrinsic and intrinsic-preprocessing bounds covering the measured
  official stream;
- actuator/dynamics randomization, especially available thrust and motor lag;
- local-gate discovery followed by uninterrupted full-course recurrent
  evaluation with actor gate index removed;
- deterministic mean-action replay and Linux/Windows export parity; and
- actual-Windows throughput above command cadence before any zero-control
  shadow.

None of this changes the current live restriction: FlightSim actuation remains
frozen, and VQ2 Submission remains forbidden.

## Recurring audit result after N528-N531

The arXiv record was checked again on 2026-07-27 and still exposes v1 only.
Sections III-A and III-B, Figure 5, Table III, and the training appendix were
revisited against the N528 failure. The paper facts retained for the next
iteration are replay context `16`, initial world sequence `64`, imagination
`16`, 70% gate-local initializations, 17M small-track environment steps, and
roughly 50 hours on a 40 GB A100 partition. This repository does not claim
equivalent compute.

Observed VQ2 adaptation results:

- N528 is `0/64` with `64/64` low crashes, despite measurable mask dependence.
  That rejects the checkpoint, not the architecture: 524,288 collected agent
  transitions are about 3.1% of the paper's 17M schedule, and its old temporal
  contract conflated context, world sequence, and imagination.
- N529 implements and verifies distinct `16/64/16` temporal roles with a
  deterministic no-gradient burn-in. This adapts the paper because recurrent
  carry tensors are not stored in this compact replay.
- N530 changes the actor to a complete CTBR vector and persists replay as
  bounded WSL memory maps. This preserves low-level neural control while
  respecting the only proven VQ2 actuation path and the local 7.7 GiB RAM
  limit.
- N531 implements Table III's 70/30 discovery idea without the paper's mapped
  flight-plan vector. The measured completed-episode local fraction is
  `0.736111` over 72 episodes. Gate phase remains decoder-only.

The next paper-derived hypothesis is resource scheduling, not a larger model:
AMP plus microbatch accumulation should raise replay density and permit world
sequences 64 then 128 under 8 GiB VRAM. Failure to improve held-out visual
control at a serious `0.5M -> 2M` transition budget would trigger diagnosis of
reward/world-model calibration before any architecture expansion.

## VQ2 application audit through N572

The later N532-N572 experiments directly tested the pinned mechanisms instead
of assuming that a larger run would repair every mismatch. All results below
are native/offline; no post-N522 FlightSim command was sent in this branch.

| Evidence | Result | Paper-guided conclusion |
|---|---|---|
| N540 paper-correct density-4 run | `0/64` Gate 1; policy moved backward/climbed | More replay alone did not fix causal action semantics. |
| N545 distributional reward/value | finite and cheap, but imagined pitch still had the wrong sign | Copying the pinned scalar heads is insufficient when transition causality is wrong. |
| N551 prior-only distillation | prior KL fell to `0.000327`; prior/posterior decoder pitch both preferred native `-0.5` | Frozen-posterior prior fitting is an effective, low-cost dynamics repair. |
| N552 decoder-progress audit | native/prior/posterior pitch all preferred `-0.5` | The paper's known progress equation is useful as a training-only diagnostic. |
| N554 endpoint-only actor | `0/64`, backward flight | Using only final replay states violated the pinned implementation. |
| N556 all-time actor (64 sampled states/update) | pitch sign corrected, but collective `+0.01523` caused climb to `8.56 m`; `0/64` | All-time starts fixed one mismatch but exposed unverified control axes. |
| N561 anchored paired replay | all four CTBR axes, eight values, initial pre-action record, no terminal/crash | Exact action-response alignment is mandatory for causal audits. |
| N563 anchored world fit | exact pitch fixed; roll/collective/yaw still dominated by decoder drift | A state decoder is a diagnostic, not a calibrated reward oracle. |
| N564 learned reward audit | nearly flat; wrong pitch/collective/yaw preferences | The current reward head was not causally admissible. |
| N567 reward-only fit | two-hot loss `0.646 -> 0.0126`, but action ordering remained wrong | Low distributional loss can hide lost sub-bin control information. |
| N569 reward x64 fit | better numerical resolution, but pitch reversed to `+0.5` | Scaling alone cannot repair a center-dominated, imbalanced causal dataset. |
| N570 progress plus effort | exact-start pitch `-0.5`; roll/collective/yaw centered at zero | Paper-style progress/control decomposition can make a local objective causal. |
| N572 all-time actor on frozen N549 replay | `0/64`, backward flight, `z=60.21 m`, timeout | A locally correct objective does not validate stale random replay as an actor-start distribution. |

The fixed 255-bin support is especially important for interpreting N567. The
bins adjacent to zero are `+-0.1705577`, while early 64 Hz VQ2 progress rewards
are commonly `1e-5` to `1e-3`. Two-hot interpolation is mathematically valid,
but a head can minimize average cross-entropy by concentrating on the center
while failing to preserve tiny action-order differences. The solution is not
necessarily a larger head; use task-scale targets, balanced causal data, and a
held-out action-order audit.

The stronger conclusion from N572 concerns replay. SkyDreamer alternates
collection, model learning, and actor learning, so its all-time posterior starts
come from replay continually refreshed by the current task policy. N549 is a
fixed Gaussian exploration buffer. Optimizing all its time states faithfully
copies a line of code but not the paper's data distribution. The next VQ2
application must therefore:

1. retain the post-N522 recurrent 4,118-value visual Puffer actor rather than
   return to the closed detector/32-value lineage;
2. alternate native collection, world-model learning, and actor/critic learning
   so the current policy continually refreshes its own replay distribution;
3. sample all-time posterior starts only from valid, reset-free recent/current-
   policy sequences and report replay age and policy-source provenance;
4. carry N570's centered effort objective into online decoder-progress
   imagination while continuing to audit every action channel; and
5. preserve the actor boundary: no detector geometry, phase input, action
   teacher, imitation data, checkpoint selector, or classical override.

This is an adaptation, not a rejection of the paper. The useful mechanisms are
temporal posterior/prior learning, privileged auxiliary reconstruction,
task-aligned progress, persistent visual corruption, and all-time starts over
continually refreshed replay. The excluded mechanisms remain mapped flight
plans, detector/decoded-state runtime phase updates, action imitation, and
direct copying of A100-scale budgets.

## Recurring paper audit through N591

The arXiv API was checked again on 2026-07-27 and still identifies only
`2510.14783v1`; no later source version was silently substituted.

The pinned DreamerV3 source was re-read at its imagination and `imag_loss`
paths after the N572/N581 failures. Two details materially changed the local
implementation:

1. `imag_last=0` makes `K=T`, so the actor starts from all `B*T` posterior
   states. With local `B=16,T=64`, that is 1,024 starts, not the earlier
   hardware-conservative subset of 64.
2. `imag_loss` uses the slow value target for its baseline, separately
   normalizes advantage, and multiplies policy/value losses by cumulative
   predicted continuation.

These are source facts from the pinned code, not claims inferred from the PDF.
The VQ2 adaptations and falsification results are:

| Evidence | Paper mechanism tested | VQ2 result |
|---|---|---|
| N573-N574 | continually refreshed task-policy replay | fail-closed resume works; N574 reaches complete current-policy replay turnover |
| N579-N581 | slow-target baseline plus normalized advantage | fixes forward pitch behavior; N581 reaches the Gate-1 plane with zero crashes but misses vertically |
| N585 | all-time replay causal surface | world objective prefers `-0.5/0/-0.1/0`; collective signal is real but very low SNR |
| N586-N588 | all `B*T=1024` starts | fits at about 1.02 GB and yields the first emergent-policy Gate-1 passes, `11/64` |
| N589-N590 | cumulative-continuation loss weighting | corrects the first roll step but does not satisfy the yaw-direction gate |
| N591 | four independent samples per posterior start | fits at 3.79 GB, but does not repair coupled whole-actor direction |

The hardware conclusion is now measured rather than assumed: the paper's
functional all-time-start estimator fits an RTX 3070, and even a fourfold
Monte Carlo expansion stays below the 7.2 GiB admission cap. A larger model or
A100 is not the current need. The remaining problem is statistical/channel
coupling in the actor update. The next audit should compare the pinned
stateful return/value/advantage normalizers and policy-head update structure,
then test the smallest training-only constraint that preserves correct
roll/yaw rows without creating a runtime channel override.

## Recurring paper audit through N612

The 2026-07-28 arXiv check still exposes only SkyDreamer v1. The pinned
DreamerV3 `imag_loss` and defaults were revisited after N612. Source facts
remain: returns use a stateful 5/95 percentile normalizer with rate `0.01` and
minimum scale `1.0`; value and advantage normalizers are disabled; SkyDreamer
sets the slow target critic on; action/value losses use cumulative predicted
continuation; and the policy uses REINFORCE over stochastic imagination.

N592 previously conflated percentile normalization, whole-actor training, and
a non-semantics-preserving migration from the legacy transformed Normal to the
pinned bounded Normal. N593-N612 isolate output rows but retain the local
center/std advantage adaptation. The learned reward surface on N604 has correct
collective ordering with only about `1e-4` total 64-step return spread. One and
two updates move collective negative, while four barely remains negative and
eight/sixteen reverse. The selected four-update child improves paired native
vertical error only `0.0211 m` and remains `0/64`.

The next paper-derived question is estimator evidence, not another checkpoint:
hold the actor distribution and posterior batch fixed, compare the local and
paper normalizers, and measure collective virtual-step direction across
independent prior/action samples at one and four samples per posterior start.
This separates reward signal strength, Monte Carlo variance, and normalizer
effects without repeating N592's distribution migration. If no scheme has a
reliably negative gradient, more sequential actor updates are not justified;
the reward head needs finer action discrimination or better current-policy
gate-event support.

## Event-support falsification through N623

N613 confirms the estimator failure is not repaired by the pinned percentile
normalizer or four Monte Carlo samples. N615 then separates sign from scale:
the local one-step collective derivative is usually correct, but the reward
difference is on the order of `1e-8`, and N604's whole fit sees only about nine
expected gate-event targets.

N617 adds 128 legal current-policy gate events without teacher actions. Jointly
updating the complete world/reward model on those events does learn phase, but
N622/N623 show the paper-style monolithic objective is not sufficient in this
small-data adaptation: event reward moves backward, dense representation
drifts, local action credit reverses, and actor gradients remain coin flips.

The next adaptation preserves the paper's world-model principle while staging
its supervision: first learn the observable-to-latent ordered transition using
the privileged phase only as a training target, with reward and actor frozen;
then fit reward on frozen latents. Neither target enters deployment. This is a
measured local decomposition, not a claim that SkyDreamer's upstream method or
paper result is wrong.

## Transition-target falsification through N632

The first staged phase target also fails under stronger isolation. N624-N627
show that aggregate phase regression and a binary crossing term reduce event
error only by creating a global positive delta on dense non-events; no branch
predicts a crossing. This rejects that local target, not the paper's recurrent
world-model principle.

N628-N632 identify the more precise issue. A native pass is a sharp threshold
over a smooth, sub-frame approach. A global binary classifier is overstrict:
even true current forward distance has held-out AUC about `0.87`, while it
orders the final event correctly in every window. N604's legal encoder/RSSM,
however, does not preserve that continuous ordering—no frozen linear feature
passes, the nonlinear decoder has `216.6` true steps of error, and refitting
its frozen hidden features reaches only `0.1005` correlation at best.

The paper-guided adaptation is therefore to improve recurrent state
representation before reward calibration: use privileged continuous
gate-relative distance only as an auxiliary training label, hold event groups
disjoint, preserve dense current-policy features/actions, and keep reward and
actor tensors frozen. The label remains absent from the deployed 4,118-value
actor input. This is consistent with Informed Dreamer's training-only state
reconstruction and avoids inventing a runtime phase estimator.

## Recurring paper audit through N710

The 2026-07-28 primary-source check still exposes only SkyDreamer v1. The
relevant paper boundary has not changed: legal execution history contains the
visual representation, measured body rates/RPM, and prior actions; privileged
state and parameters supervise the recurrent world model; actor learning is
separate latent imagination; and the paper's visual-ambiguity result depends
on its mapped flight-plan update, which remains forbidden in VQ2.

N704-N710 are a measured local adaptation of the informed-state objective.
The decoder-free covariance loss learns a rotation/sign-invariant progress
geometry from training-only plane labels, and N710 proves that one frozen
train-only linear readout generalizes to physical validation. This does not
alter the actor input or claim that SkyDreamer uses covariance supervision.
It addresses the paper's published decoder-drift limitation with a falsifiable
VQ2-specific representation gate while preserving the informed-POMDP boundary.

Paper coverage is actionable-complete for the next experiment. The local
review covers the full methodology, Figure 3 data flow, Tables II--IV,
GateNet/StochGAN appendices, randomization, staged training schedule,
deployment, flight-plan dependence, and limitations. Exact SkyDreamer code,
checkpoint, replay contents/order, and full experiment configuration remain
unpublished, so replication completeness and exact-reproduction claims remain
unavailable.

## Recurring paper audit through N722

N711-N722 preserve the paper's informed-POMDP boundary. Privileged distance
supervises only a frozen auxiliary/readout and prior training; the deployed
posterior actor consumes only legal visual/IMU/action history. N713 confirms the
representation is causally useful, while N714/N717 show why Dreamer-style actor
imagination cannot be assumed valid after changing encoder/sequence/posterior
without recalibrating prior and reward.

The measured repair follows the paper's model-before-actor dependency. N716
repairs categorical/hard-state dynamics, but low KL alone does not guarantee
the continuous progress direction needed by this VQ2 adaptation. N719-N721 add
a frozen-readout diagnostic without privileged runtime state. N722 rejects
naive sequential minibatch continuation and points to a train-aggregate
gradient audit.

Coverage remains actionable-complete, not replication-complete. Published
method details and the pinned DreamerV3 dependency suffice for the next audit;
unpublished exact code, checkpoints, replay ordering/content, and full
experiment configuration remain unavailable and must not be inferred.

## Solve-first adaptation update — 2026-07-28

The paper remains useful for its causal recurrent-state framing, history-aware
training, separate privileged training signals, model-before-policy validation,
and deployment discipline. It does not require this repository to preserve the
failed N523--N735 world-model optimizer as the only solution path. SkyDreamer's
mapped runtime flight-plan vector is also outside the current legal actor ABI,
so its course-conditioning result is not evidence that phase will emerge here.

The active VQ2 adaptation therefore uses the smallest falsifiable perception
and control stack: deterministic intrinsic normalization, a causal soft mask
that preserves every red pixel, a compact CNN, one recurrent unit, and a
Gaussian actor whose deterministic mean supplies the complete CTBR command.
Native geometry may generate offline oracle labels, rewards, curricula, and a
privileged critic, but it may not enter actor observations or deployment
actions. Full-history BC/DAgger establishes ordered multi-gate behavior before
teacher-free recurrent PPO improves robustness and time.

This is a solve-first change in optimizer and representation, not a relaxation
of runtime legality. A learned GateNet becomes eligible only if source-locked
official frames demonstrate that the all-red-pixel mask is insufficient. The
N523--N735 Dreamer work remains frozen evidence, and N712's sealed test remains
permanently consumed. FlightSim commands and Submission remain unauthorized.
