# VQ2 Informed-Dreamer Puffer Implementation — 2026-07-27

## Status

This is offline training infrastructure, not an authorized FlightSim policy.
No simulator lifecycle, arm, reset, setpoint, or actuator command was sent while
building or testing it. The authoritative live failure is N522 and live
actuation remains frozen. N572 rejects its checkpoint and the frozen N549
Gaussian-replay actor continuation; it does not reject the post-N522 4,118-
value emergent visual Puffer/Informed-Dreamer architecture. This infrastructure
remains the active basis of that legal successor.

The implementation combines two primary visual-racing references:

- Geles et al., *Demonstrating Agile Flight from Pixels without State
  Estimation*, RSS 2024: continuous inner-gate-edge masks, three prior actions,
  efficient projective mask rendering, and randomized corrupted edge segments.
- Verraest et al., *SkyDreamer*, arXiv:2510.14783v1: a recurrent discrete RSSM,
  privileged decoder targets, reward/continue prediction, latent imagination,
  deterministic deployment, and mean-action smoothness.

Geles et al. used a feed-forward PPO actor, CTBR, an asymmetric critic, and an
explicit perception reward. Those parts are not copied: VQ2 uses recurrent
Informed Dreamer and has no camera-alignment/look-at reward. SkyDreamer's mapped
flight-plan vector is also excluded. Ordered course phase must remain in the
actor's recurrent latent state.

Primary sources:

- <https://doi.org/10.15607/RSS.2024.XX.082>
- <https://arxiv.org/abs/2510.14783v1>
- <https://github.com/glambrechts/informed-dreamer>

Detailed paper extraction and the repository-specific adopt/adapt/exclude
matrix are in
[`skydreamer_paper_review_2026-07-27.md`](skydreamer_paper_review_2026-07-27.md).
That review is based on the full v1 text and a visual scan of all 17 PDF pages.
It is authoritative for what SkyDreamer actually reports, including the
important distinction between replay context `16`, world-model sequence length
`64 -> 256`, and imagination horizon `16`.

The authoritative execution loop that turns those findings into hardware-
bounded experiments and eventual VQ2 promotion is
[`vq2_paper_guided_puffer_goal_prompt.md`](vq2_paper_guided_puffer_goal_prompt.md).

## Native Puffer ABI

The new `drone_race_vision` binding is separate from `drone_race`, preserving
the historical 32-float checkpoint ABI.

The vector environment emits 4,152 floats per agent. This is a transport ABI,
not the actor ABI:

| Slice | Size | Use |
|---|---:|---|
| continuous `64x64` gate-edge mask | 4,096 | deployed actor |
| measured body rates | 3 | deployed actor |
| measured motor/actuator feedback | 4 | deployed actor |
| three executed action vectors | 12 | deployed actor |
| new-frame, normalized frame age, normalized step `dt` | 3 | deployed actor |
| privileged state/parameter targets | 34 | training decoder only |

The legal actor boundary is therefore exactly 4,118 values. The 34-value
training-only tail contains normalized pose, current-gate-relative position,
world/body velocity, attitude, true body rates and RPM, camera extrinsics,
selected dynamics parameters, ordered phase, and aperture. It is never passed
to the observation encoder, RSSM posterior, recurrent update, or actor.

The camera cadence uses a fractional accumulator: the initial frame is fresh,
then a nominal 30 Hz mask is held across the 64 Hz policy loop with explicit
freshness and age. Training can randomize frame dropout, edge dropout, false
segments, camera extrinsics, rolling-shutter deformation, course geometry, and
dynamics without changing the legal schema.

## Model and trainer

`pufferlib/vq2_dreamer.py` implements:

- CNN legal-observation encoder;
- discrete stochastic latent and single-layer GRU sequence model;
- prior dynamics and legal-observation posterior;
- privileged decoder plus reward and continuation predictors;
- Gaussian four-channel actor and slow target critic;
- symlog reward/value scaling;
- 16-step default latent imagination with REINFORCE actor loss;
- deterministic posterior/action inference;
- policy-mean smoothness regularization at the SkyDreamer starting weight
  `0.002`.

`scripts/train_vq2_informed_dreamer.py` collects only native Puffer rollouts.
Its time-major replay stores masks as `uint8` and remaining values as `float16`,
then reconstructs float batches for world-model and imagination updates. The
replay can be a bounded six-segment NumPy memory map on the WSL filesystem and
resumes only when its versioned metadata exactly matches the checkpoint. The
actor is never bootstrapped from a classical controller or action teacher.

Replay context, optimized world-model sequence, and imagination horizon are
separate controls and metrics. A deterministic, gradient-free 16-step burn-in
reconstructs the recurrent state; loss is applied only to the following
64-step world sequence. A regression test proves that full deterministic
unrolling and context-plus-warm-start unrolling are exact.

The current action schema is `normalized_attitude_ctbr_v1`, paired with visual
observation schema `vq2_visual_ctbr_v2`. All four policy values form one neural
pitch/roll/collective/yaw target vector. A fixed gyro-closed decoder converts
it to the public `SET_ATTITUDE_TARGET` body-rate/thrust contract. No external
controller chooses, blends, clips, schedules, or replaces any action channel.
Zero action is hover- and level-centered in the native plant.

The native discovery curriculum samples a uniformly selected active gate for
70% of resets and the true course start for 30%. Local episodes begin at a
nominal 2-4 m offset before the active plane, followed by the separately
configured spawn-position noise, with neutral velocity, attitude, and rates.
Reset provenance and gate phase are training diagnostics/decoder labels beyond
the 4,118-value actor boundary; neither is an actor input. Every admission
screen must disable the curriculum and start from the uninterrupted course
start.

The current trainer is a compact Puffer-native adaptation, not a reproduction
of the paper's compute schedule. SkyDreamer reports 17 million environment
steps for its small tracks, `train_ratio=128`, replay context `16`, initial
world-model sequence length `64`, and sequence length `256` after 8 million
steps. Its big-track variant uses 35 million steps and a different recipe. The
N523-N528 runs are integration and early-discovery experiments and must not be
described as paper-scale training.

## Verification completed through N531

- Historical native regression suite: passed.
- New visual C regression suite: passed.
  - continuous centered mask;
  - exact 4,118/34 schema;
  - 30/64 Hz frame hold and freshness;
  - three-action history;
  - training-only mass intervention changes the privileged target but leaves
    the legal slice bitwise unchanged.
- Focused PyTorch contract/model/replay/counterfactual suite: `13 passed`.
  - encoder rejects 4,152-value legal-plus-privileged tensors;
  - privileged intervention leaves posterior and actor action exact;
  - finite world-model loss and gradients;
  - finite imagination actor/critic losses and gradients.
- The visual native suite now passes `8/8`, including CTBR decoder targets,
  hover-centered zero action, terminal replay ordering, centered gate reward,
  and gate-local reset coverage without actor phase exposure.
- N528 disjoint deterministic evaluation is `0/64`, all low crashes. Its
  initial-mask blank counterfactual changes `64/64` actions with mean absolute
  delta `0.0084648455`; the checkpoint sees the image but is not competent.
- N529 executes exactly two world/critic updates and zero actor updates under
  context/world/imagination `16/64/16`. Peak CUDA allocation/reservation is
  `65,055,232/90,177,536` bytes at batch one.
- N530 repeats that temporal contract with versioned CTBR schemas and a
  25,927,680-byte NumPy-memmap replay. Checkpoint SHA-256 is
  `6e877f67078c2c29c04bc427f2f1bd591de0acbec8fec0d87f9e3a72b434924d`.
- N531 runs 164,480 native agent transitions with zero optimizer steps,
  completes 72 episodes, and measures gate-local reset rate `0.7361111045`.
  Its bounded replay is 21,606,400 bytes and peak CUDA allocation is only
  24,141,312 bytes. Metrics SHA-256 is
  `088b13822d29f429ecf7ad2a67690b38624f9df1377ef54170e8e45d2ecfd1da`.

These smokes prove integration only. They do not prove course discovery,
held-out robustness, valid gate completion, Windows cadence, or deployment
fitness.

## Implementation added after N531

The hardware and temporal prerequisites were completed without approaching
the 8 GiB limit:

- BF16 autocast plus weighted logical microbatches;
- bounded WSL memory-mapped replay with exact before/after hashes;
- independent context, world-sequence, and imagination lengths;
- paper-matched 255-bin symmetric `symexp_twohot` reward support;
- action-conditioned reward as an optional VQ2 diagnostic adaptation;
- prior-only distillation with every non-prior tensor frozen;
- all-time posterior imagination starts with a fixed hardware-bounded subset;
- training-only decoded gate-range progress;
- exact paired CTBR causal replay with a pre-action initial record; and
- decoder-progress plus channel effort as an auditable training objective.

Relevant implementations are:

- `pufferlib/vq2_dreamer.py`;
- `scripts/continue_vq2_informed_dreamer_prior_offline.py`;
- `scripts/continue_vq2_informed_dreamer_actor_offline.py`;
- `scripts/collect_vq2_causal_channel_replay.py`;
- `scripts/finetune_vq2_informed_dreamer_causal_world.py`;
- `scripts/continue_vq2_informed_dreamer_reward_offline.py`;
- `scripts/analyze_vq2_informed_reward_channels.py`; and
- `scripts/analyze_vq2_decoder_causal_surface.py`.

The focused Dreamer/actor/reward tests pass, including deterministic prior
feature construction, reward-target scaling, all-time start selection, effort
validation, privilege isolation, and actor/critic gradients.

## Verification through N572

- N551 distills the deterministic prior to raw KL `0.000327` in 100 updates,
  with eight prior tensors changed and about 112 MiB peak allocation.
- N552 establishes correct native/prior/posterior pitch ordering from the
  training-only decoded progress equation.
- N556 applies the pinned all-time-start correction. It fixes pitch sign but
  produces positive collective and climbs, proving a one-axis audit is
  insufficient.
- N561 records 16,640 valid exact-start causal transitions over all four CTBR
  axes, eight values, and 64 action steps, including 256 pre-action anchors.
- N563 corrects exact-start pitch dynamics, but weak roll/yaw effects and
  collective are dominated by residual decoder drift.
- N567 fits only the reward predictor for 128 updates; loss reaches `0.01258`
  while the learned reward ordering remains wrong. The nearest nonzero
  DreamerV3 support bins are `+-0.1705577`, much larger than early VQ2 rewards.
- N569 scales reward by 64, but both learned pitch surfaces reverse to `+0.5`.
  The scaled distributional-reward branch is rejected.
- N570 locks decoder progress plus effort weights `[0,0.025,0.06,0.5]` and
  passes its exact-start four-channel causal audit.
- N572 performs 128 actor/critic updates from 64 all-time starts/update at only
  119,347,200 bytes peak CUDA allocation. It is nevertheless `0/64` natively:
  positive recurrent pitch/collective drives backward flight and a climb to
  mean `z=60.21 m` before timeout.

This falsifies the use of N549's stale Gaussian replay as the actor-start
distribution. It does not falsify compact recurrent world-model auxiliaries.
The pinned paper continually refreshes replay with the current task policy;
matching its all-time-start code without matching that data loop is not a
faithful application.

## Next evidence gate

Do not resume N572, launch another fresh random-replay actor, or send FlightSim
traffic. Continue the emergent visual architecture through the alternating
collection/model/actor loop that the paper actually uses:

1. make the online trainer sample actor starts from all valid posterior time
   states rather than only sequence endpoints;
2. carry N570's centered effort weights `[0,0.025,0.06,0.5]` into the online
   decoder-progress imagination objective;
3. require fresh transitions from the current visual actor before resumed
   actor/critic updates, with fail-closed replay/checkpoint restoration;
4. report collector provenance, records written since resume, replay refresh
   fraction, and replay-age statistics so stale-data failure is visible; and
5. screen unassisted full-start behavior under held-out visual, timing, camera,
   actuator, and dynamics randomization before scaling the transition budget.

The deployed artifact remains one recurrent Puffer action source over the
legal 4,118-value visual observation. No detector geometry, official phase,
classical action labels, teacher imitation, checkpoint selector, or runtime
override is permitted.

## Verification through N591

The online/resume contract and actor estimator were subsequently tightened
against the pinned DreamerV3 implementation:

- replay samples expose session age/current-policy provenance;
- resume inserts a reset boundary and withholds all optimizer work until 128
  fresh current-policy vector steps have been collected;
- all-time imagination uses a single centralized selector;
- the actor advantage uses the slow target critic and centered/scaled
  advantages; and
- actor and critic losses now use cumulative predicted-continuation weights.

N573 validates partial refresh and N574 turns the entire bounded replay over to
current-policy data before optimization. N575/N578 still fail because the
local actor estimator omitted pinned target-baseline/advantage normalization.
After that correction, N581 is `0/64` but becomes the first forward policy: all
64 episodes reach Gate 1's plane, zero crash, with mean radial error `3.03748 m`
dominated by `3.02240 m` vertical error.

Horizon-64 causal audits N582-N585 show the unchanged N581 world objective
prefers pitch `-0.5`, roll `0`, collective `-0.1`, and yaw `0`. The collective
advantage over zero is only about `0.0023` against a stochastic return spread
near `2.6`, exposing estimator variance rather than a wrong action ordering.

N586 proves the paper's complete `16*64=1024` start set fits easily:
`1,007,411,200` peak allocated CUDA bytes and `4.45 s` for one update. N587
runs 16 such updates at `1,020,971,520` bytes and produces the first unassisted
emergent-policy Gate-1 passes. N588 scores `11/64`, zero crash/timeout; vertical
error improves to `2.71434 m`, but lateral error regresses to `1.18906 m` as
roll drifts negative.

N589 confirms zero roll remains the deterministic and stochastic replay-wide
optimum. Adding pinned cumulative-continuation weighting makes the first N590
roll step correct but fails its yaw-direction gate. N591 repeats each posterior
four times (`4,096` rollouts) at `3,788,349,440` bytes peak, yet coupled whole-
actor gradients still move roll/yaw the wrong way and collective toward zero.
It is rejected.

The RTX 3070 is not the limiting resource for this compact model: full paper-
matched starts and four Monte Carlo samples both fit. The active blocker is
actor-gradient/channel coupling under a low-signal learned objective. Retain
N587 only as the offline `11/64` frontier. Before another child, audit the
pinned stateful normalizers and test a training-only actor-head constraint that
keeps already-correct roll/yaw neural rows stable while improving pitch and
collective. Runtime remains one complete recurrent Puffer action vector.

## Verification through N612

Restricted mean-head hooks keep the actor trunk, standard-deviation rows, and
unselected action rows bit-identical while allowing a complete neural action at
deployment. N593-N594 validate that mechanism. N595 is `0/64`, zero crash/
timeout, with substantially improved lateral but still `2.17037 m` vertical
under-correction.

N604 refreshes the complete bounded replay under the mixed paper curriculum
with current-policy data and eight centered native gate events while holding
the actor fixed. Its distributional learned reward has weak correct collective
ordering. N608 is correct for one update, N609 reverses by 16, and N610/N611
show the direction has already reversed by eight. N612's selected four-update
child remains `0/64`; its paired vertical improvement is only `0.02110 m`.

Do not continue the fixed learned-reward update-count branch. The next
implementation gate is a read-only gradient-SNR audit that holds one posterior
batch fixed, mutates no checkpoint, and compares one/four imagination samples
under local center/std and paper percentile return normalization. Only stable,
measured negative collective direction can admit another restricted actor
training design. FlightSim remains frozen.

## Verification through N623

The read-only gradient and local-reward audits now have source-locked tools and
tests. N613 rejects normalizer/sample-count changes. N615 inventories replay
support and proves local reward direction is far below useful numerical scale.

`scripts/collect_vq2_gate_event_windows.py` records compact 17-step windows
ending at true native gate events while the unchanged recurrent policy supplies
all actions. N617 produces 128 windows across all agents and phases with exact
privilege isolation. `scripts/continue_vq2_event_balanced_world_offline.py`
supports dense/event mixing, fresh optimizer control, fixed dense validation,
and full-corpus prior event metrics.

N618-N623 falsify joint world/reward continuation: phase prediction improves,
but reward/crossing and local action credit do not, and dense state eventually
regresses. Do not continue those children. Implement the next integration with
parameter and loss isolation: RSSM transition plus privileged phase decoder
only, reward/continuation/actor/critic exact. A later reward-only stage may use
frozen latent features only after phase and dense gates pass.

## Verification through N632

`scripts/continue_vq2_phase_transition_offline.py` isolates RSSM sequence/prior
and privileged phase row updates. N624-N627 establish that neither smooth phase
regression nor crossing BCE learns a crossing before dense non-event delta
drifts beyond tolerance. Reward, continuation, actor, critic, encoder,
posterior, and non-phase rows stay exact, so the failure is target-specific.

Three read-only tools then localize the representation failure:

- `scripts/audit_vq2_event_latent_separability.py` uses equal legal histories,
  group-disjoint splits, and a privilege counterfactual;
- `scripts/audit_vq2_event_representation_localization.py` compares frozen
  encoder/posterior/prior feature families; and
- `scripts/audit_vq2_plane_progress_support.py` plus
  `scripts/audit_vq2_plane_representation_capacity.py` replace the binary
  target with continuous active-gate forward distance.

The native label is smoothly ordered in all held-out windows. N604's decoded
forward estimate has correlation `0.0547` and `216.6` step-scaled MAE, and no
frozen feature regression exceeds `0.1005` held-out correlation. The next
implementation must update the legal encoder/recurrent posterior with this
training-only continuous target, while anchoring N604 dense features and actor
actions. Prior, reward, continuation, actor, and critic weights remain frozen
until representation admission. No N624-N632 artifact is screenable.

## Verification through N669

The representation branch now separates three distinct questions that the
short N617 windows conflated:

- exact recurrent-boundary replay (`collect_vq2_gate_event_windows.py` v4 and
  `filter_vq2_event_windows_replay.py`);
- broad current-policy range coverage (`audit_vq2_dense_replay_coverage.py`
  and `audit_vq2_broad_visual_association.py`); and
- long successful visual-history capacity
  (`audit_vq2_success_prefix_sequence_capacity.py`).

N662 proves the frozen N587 carried state does not expose near-plane progress.
N664-N666 prove N652's failure-heavy replay exposes coarse mask range but lacks
crossing-scale direction. N667/N668 then provide 128 replay-exact successful
321-step N587 prefixes with zero selected latent mismatch. N669 removes fixed
event-clock leakage using seven crop offsets and passes its untouched visual
sequence test at `0.960515` correlation, `0.240912 m` near-plane MAE, and
`80.23%` direction. Its tail-only control stays near zero.

This does not promote a model: all diagnostic probe weights are discarded and
N587 remains `11/64`. The next implementation must train the existing legal
encoder/RSSM from exact crop boundary states while the actor and all task heads
are frozen, with deterministic action anchors on successful and broad replay.
Use float32 semantic drift as authority. No native screen or FlightSim action
is admitted by N669.

## Verification through N672

Two source-locked integration executables now implement exact successful-
prefix boundaries, a differentiable soft-posterior training auxiliary,
float32 actor/latent drift audits, and continuous-child validation from stored
initial states:

- `scripts/continue_vq2_success_prefix_representation_offline.py` performs the
  one-step N670 trust-region integration; and
- `scripts/continue_vq2_success_prefix_representation_joint_offline.py`
  performs N671/N672 multi-step joint probe and legal encoder/RSSM/posterior
  training with fixed hard/soft actor distillation.

N670 proves N587's frozen posterior does not already decode continuous plane
progress. N671 shows the signal can grow to `0.7392` correlation, but numerical
latent preservation conflicts with learning even when actor outputs remain
stable. N672 removes that false proxy and reaches `0.942461` continuous-child
validation correlation, `0.302188 m` overall MAE, and `0.803922` direction at
step 400. The remaining failures are localized: `0.542766 m` near-plane MAE
and hard-action RMSE `0.001016/0.001715` on validation/broad histories.

The N672 executable selected and saved step 150 because later checkpoints
exceed the fixed action RMSE gate. Its `model.pt` must never be confused with
the unsaved high-signal step-400 state. The next implementation is an
export-only deterministic reconstruction of that exact trace, followed by a
frozen-representation stage that calibrates the training probe and distills
the actor back to fixed N587 actions. No reward/prior/decoder/critic update,
N668 test evaluation, native environment, or FlightSim path is admitted yet.

## Verification through N676

N673 establishes that the original CUDA optimization was not bitwise/`1e-7`
reproducible, so its recovered endpoint is rejected. N674 instead creates a
fresh self-contained donor and saves its own selected step 350. N675 adds the
frozen-feature calibration path:

- `calibrate_vq2_frozen_representation_actor.py` precomputes legal donor
  features, holds every representation/task tensor exact, fits the auxiliary
  probe, and distills only actor tensors to N587 actions;
- N652 calibration train/validation sequences are sampled without replacement
  and have zero overlap; and
- `eval_vq2_frozen_representation_test.py` performs the sole mutation-free
  N668 test read and rechecks the exact N652 holdout.

N675 passes validation with `0.944367` correlation, `0.316195/0.467785 m`
overall/near-plane MAE, `0.827731` direction, and action RMSE below `0.000173`.
N676 then generalizes at `0.931859` correlation, `0.328685 m` overall MAE,
`0.807359` direction, and test action RMSE `0.000104851`, but near-plane MAE is
`0.501634717 m`. The fixed `0.50 m` gate fails; the branch is closed.

The N668 test partition is consumed. Future representation work must collect
new episode-disjoint successful prefixes, freeze new validation/test groups in
advance, and treat all N668 records as development-only. No native or live
policy authority follows from the near miss.

## Verification through N680

N677 runs the exact-history collector at episode offset 4000 and freezes 416
deterministic N587 Gate-1 histories from 64 agents and 412 vector steps. N678
then replays every prefix independently and writes the first 400 of 411 records
within the fixed `0.002` deterministic/logit bounds; selected stochastic
mismatch is zero.

`freeze_vq2_event_window_partitions.py` adds a model-free partition contract.
It loads only the integer `vector_step`, shuffles sorted unique groups under a
fixed seed, assigns every group wholly to one split, locks source hashes, and
records the arrays deliberately not loaded for partitioning. N679 exposed and
closed a direct-CLI import defect before any input read. The corrected direct
CLI and focused collector/filter/partition suite pass `10/10`.

N680 freezes 272 train, 64 validation, and 64 test events in 268, 64, and 64
disjoint vector-step groups. The manifest SHA is `91b9acf1...`; the test split
has not been evaluated. The next implementation should mechanically
materialize or index the three partitions so development code cannot open test
rows, then operate only on train/validation support until a fixed-gate donor
and frozen-calibration sequence earns one preregistered test read.

## Verification through N687

`materialize_vq2_event_window_partitions.py` mechanically emits exact physical
train/validation/test files from N680; N681 verifies every copied array while
computing no content metric. `audit_vq2_event_partition_support.py` has no test
argument and shows that uniform overlapping crops halve the relative influence
of late near-plane timesteps despite every event containing 43--49 of them.

The partitioned joint-continuation path now supports fresh train-only probe
normalization, inverse crop-exposure plane loss, fixed near-plane weighting,
and a source-locked fixed reference batch. N683 exposes recurrent CUDA
batch-shape sensitivity; after N684's report-only shape bug, N685 proves batch
416 reproduces the filter boundary while 4/64/336 do not. N686 then passes the
new path end to end with no test access.

N687 trains 400 updates from N587. It passes replay, isolation, resource,
action, correlation, overall-MAE, and donor near-plane gates, but direction
accuracy reaches only `0.713914`; the donor is rejected. The implementation
corrected timestep-level exposure only. The next smallest change is to compute
crop exposure for temporal pairs and apply inverse pair weights to delta loss,
with the same fixed near-plane rule, before any further donor run.

## Verification through N695

`audit_vq2_event_pair_support.py` extends the physical-partition audit from
samples to adjacent post-burn temporal pairs. N688 shows 78,064 unique train
pairs but 180,880 crop-pair exposures. The partitioned joint trainer now
supports inverse pair exposure and an either-endpoint near-plane multiplier in
`_weighted_delta_loss`; N689 proves that path in one update.

N690 reaches a fully passing validation state at step 350, but exposes a
selection bug: the prior lexicographic key preferred step 400's higher
correlation even though overall MAE exceeded its fixed gate. The selector now
orders candidates by all preservation gates, then complete progress admission,
then the existing scalar tie-breaks. `audit_vq2_donor_selector.py` proves that
rule on immutable history without model or dataset access in N691. N690 state
is still rejected; the audit cannot reconstruct an unsaved checkpoint.

Fresh-seed N692 does not reproduce the pass. The report-only/train-only
`audit_vq2_donor_sampling_variance.py` recreates the exact interleaved crop and
dense sampling stream. N693 finds N692 in the lower 11th percentile for
near-plane loss share and 8.5th for qualified-pair exposure; the two schedule
measures correlate `0.989558` across 200 seeds. A 357-update counterfactual
exposes every crop exactly three times and every event exactly 21 times.

The joint trainer therefore has a default-off `_EpochBatchSampler`, source-
locked sampling summaries, and a count-spread process gate. N694 passes the
one-update path. N695 completes the exact three epochs and passes correlation,
overall MAE, near-plane MAE, replay, action, and resource gates, but direction
is only `0.646205`. Balanced delivery is necessary hygiene, not a sufficient
representation solution.

The remaining uncontrolled coupling is the random `SoftPlaneProbe`
initialization: its randomly oriented decoder participates in representation
gradients from the first update. Before adding a warmup, implement a read-only
model audit that computes representation-gradient norms, pairwise cosine, and
top-coordinate sign agreement for at least eight independent probe seeds on a
fixed N681-training batch. In the same source-locked audit, fit only ephemeral
probe tensors for one exact balanced epoch with N587 frozen, then recompute the
same gradients. Do not use validation or test labels for this decision.

Only if probe-only warmup materially improves alignment under preregistered
finite/norm/agreement gates should the joint trainer gain a default-off warmup
stage. Smoke it once, then run one fresh balanced donor from N587. If alignment
does not improve, abandon decoder-mediated plane regression and test a direct
ordinal/contrastive recurrent-feature target instead. The N681 sealed test,
actor/reward/prior training, native environment, and FlightSim remain frozen.

## Verification through N703

`audit_vq2_probe_gradient_alignment.py` proves that one balanced epoch of
probe-only fitting does not align representation gradients across eight probe
seeds. The overall median cosine falls from `0.028238` to `0.010819`; N696
therefore closes warmup. `audit_vq2_actor_low_sensitivity_progress_channel.py`
then separates two mechanisms. A deterministic low-sensitivity direction is
actor-invisible but cannot backpropagate through categorical argmax state.
The actor-null/simplex-tangent direction has exact residuals near `1e-16`,
negligible action drift, and nonzero gradients in every RSSM component.

N699's immutable report is formally rejected because its source checker
expects a field absent from the older N698 schema. The report-only N700 audit
verifies this is the sole mismatch without loading a model or data.
`continue_vq2_success_prefix_representation_joint_offline.py` now contains a
default-off tangent residual path that requires N700, inverse sample/pair
weighting, and reports tangent-only component gradients. N701 passes exactly
one update; its checkpoint is quarantined.

N702 repeats paired N695 for 357 updates with tangent weight/scale `0.01/0.05`.
It preserves every process and action gate but direction reaches only
`0.656994`. `audit_vq2_tangent_validation_readout.py` then evaluates the
fixed feature projection directly on consumed validation without fitting.
N703 fails direction and near-plane gates at `0.639137` and `0.661581 m`.
This rules out a hidden passing fixed-channel representation and closes weight
sweeps of that mechanism.

The next implementation should begin as a no-checkpoint train-only audit of a
decoder-free geometry. Prefer weighted squared covariance magnitude between
soft posterior features and plane level plus adjacent delta because it is
invariant to orthogonal feature rotations and target sign. Verify invariance,
non-collapse, and finite nonzero encoder/sequence/posterior gradients with
N587/actor bit-exact. If it fails, move to supervised pairwise-distance/triplet
geometry. A random `SoftPlaneProbe` must not influence representation updates;
fit any readout afterward on frozen train features. Test/native/FlightSim stay
frozen until the unchanged donor and strict calibration gates pass.

## Verification through N705

`audit_vq2_rotation_invariant_progress_geometry.py` implements weighted
squared cross-covariance magnitude for plane level and adjacent delta. N704
uses N681 train only and verifies a dense Householder rotation, target-sign
flip, non-collapse, component gradients, fixed replay, source contracts, and
N587 immutability. All gates pass; level/delta scores are
`0.00585197/0.000005260` and gradient norms are
`0.004173/0.042069/0.001793` for encoder/sequence/posterior.

The partitioned joint trainer now has a default-off invariant objective,
mandatory N704 source report, and `--detach-probe-representation`. It separately
measures invariant-only component gradients. N705 passes one update: invariant
and complete RSSM gradient maxima match, proving the random probe does not
shape the representation. The smoke child is quarantined.

Do not launch a full donor yet. Implement a deterministic affine ridge readout
with one preregistered lambda and float64 solve. Fit on frozen N681-train rows
using train-only normalization/inverse exposure, then evaluate validation
without selection. Smoke model immutability, deterministic coefficients,
finite metrics, fixed source hashes, and absence of sealed test/native/live
paths on N705. Only then may a fresh three-epoch invariant representation start
from N587 and be evaluated by that same frozen readout.

## Verification through N710

`audit_vq2_frozen_ridge_readout.py` closes the readout contract in N706. It
uses continuous legal replay, covered history indices 32--319, unit base
weight with the fixed near-plane multiplier two, train-only weighted
standardization, lambda `0.001`, and a float64 solve. The duplicate fits are
byte-identical; the model is frozen and validation does not choose the fit.

The joint trainer now also supports fixed-final selection. N707 starts from
N587 and performs 357 balanced invariant updates. Sampling is exact: 5,712
train draws give every crop three exposures and every event 21; 2,856 dense
draws give each anchor 22--23 exposures. The random probe is detached,
invariant gradients reach encoder/sequence/posterior, only the allowed RSSM
tensors change, and all action/parameter/process/resource gates pass.

`audit_vq2_invariant_donor_ridge.py` source-locks the N707 training contract,
N704/N706, N681 physical partitions, and the unchanged N706 numerical helper.
N708 exposes a reportless CUDA/DXG execution failure. N709 moves to CPU and
localizes a NumPy-boolean JSON defect. The repaired helper converts the exact
covered-history predicate to a native boolean and its regression suite passes
7/7. N710 then completes with coefficient SHA `bd2b1e32...`, 512 active
features, and validation correlation/overall/near/direction
`0.957912/0.261334 m/0.443587 m/0.861979`. All four fixed donor gates and all
process gates pass; no model tensor changes and no checkpoint/test/native/live
effects occur.

Do not invoke the existing N675 calibrator unchanged. It assumes the combined
N668 split, N674 step 350, and a learned probe. The next implementation should
accept physical N681 train/validation inputs plus N710, refit the exact ridge,
store its complete fit as auxiliary metadata, and verify strict validation
without any model optimizer. N707's recorded N681-validation and N652 hard-
action RMSE/max are already below `0.001/0.01`; independently reproduce them
with actor step zero. Only a bit-exact-model validation package may admit an
adapted one-time evaluator for the still-unopened physical N681 sealed test.

## Verification through N722

`package_vq2_invariant_donor_readout.py` produces N711 with a bit-exact N707
model and frozen ridge. `eval_vq2_invariant_ridge_sealed_test.py` reads the
physical test exactly once in N712 and passes; the test is permanently
consumed. `screen_vq2_n711_native_gate1.py` records N713's paired `13/64`
Gate-1 result.

`audit_vq2_imagination_readiness.py` identifies stale prior/reward heads after
the representation-only change. `continue_vq2_invariant_prior_offline.py`
yields N716 with validation KL about `0.0002`, while
`audit_vq2_repaired_imagination_readiness.py` exposes the remaining
fixed-readout soft-progress mismatch.

The progress path adds a differentiable frozen-ridge loss, restored virtual
Adam surface, virtual-to-real parity, and held-out validation. N719 selects
level-only at `3e-7`; N720 reproduces it and N721 proves held-out improvement
at all horizons. N722's resumed second minibatch fails, so sequential minibatch
continuation is closed. Next implement a read-only full-train aggregate-gradient
surface from N720. Actor, reward, native, test, and FlightSim paths stay absent.

## Frozen continuation through N735 and solve-first supersession

N723--N726 closed the one-step aggregate-gradient branch: the exact full-train
surface produced only a small open-loop improvement and did not justify a real
update. N727--N728 then exposed a train-versus-held-out conflict in direct
multi-horizon optimization. N729--N732 established N731 as the retained prior
diagnostic; N732 measured direct held-out open-loop MAE at horizons
`1/2/4/8/16` as `0.135247/0.134998/0.131514/0.132079/0.157686 m`.

N733--N735 tested the final small direct continuation. N734 improved horizons
2--16 but regressed held-out horizon 1 by `0.0001622075 m` relative to N731, so
N735 rejects it. Its report SHA-256 is
`46bfe16694965f514b6030f685cea37cb2c7bcd99268b7fb4ab9e327bbaab7f1`.
Do not resume this line as N736. N712's physical sealed test was consumed once
and must never be opened, hashed, scored, or used for tuning again.

On 2026-07-28 the user explicitly superseded the prior requirement to solve VQ2
only through emergent Informed Dreamer training without classical labels. This
document and N523--N735 are now frozen research evidence. The active direction
is the solve-first recurrent Puffer plan in
`docs/vq2_paper_guided_puffer_goal_prompt.md`: causal all-red-pixel mask
preprocessing, one compact CNN/GRU full-output actor, privileged native
oracle/critic only during training, full-history BC/DAgger, and recurrent PPO
fine-tuning. Runtime actions remain exclusively the deterministic output of one
recurrent Puffer policy. No live FlightSim authority is created by this change.
