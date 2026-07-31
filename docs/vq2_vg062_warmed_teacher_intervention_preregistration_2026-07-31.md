# VQ2 VG062 warmed teacher-intervention feature collection — 2026-07-31

## Evidence and causal decision

Run exactly one resumable offline collection tagged
`vq2_vg062_warmed_teacher_intervention_features_001`. VG061 terminally rejects
ordinary behavior cloning on VG039's failed phase-4 trajectories: direct
indexed-head scales `0.1` through `1.5` leave the parent's ordered gate reach
exactly unchanged at `256/237/44/3/0/0`, and larger scales only add crashes.
The rejection SHA-256 is `dbd93998...`; an unchanged retry is forbidden.

VG062 is the causally distinct repair. Let admitted VG033 emit every plant
action through held public phase 0. Beginning only when the legal held public
phase becomes 1, execute the admitted SF016 alignment-oracle query as a
**training-only** plant action. Continue advancing VG033's recurrent state on
the legal observation stream, including the causal history of the action that
actually drove the training plant. Store a compact feature/label row for each
teacher-intervention step. This creates successful on-distribution recovery
histories instead of relabeling failure histories.

## Fixed collection contract

- Seed `429197`, 512 vector agents, one terminal episode per agent, exact-uniform
  native course counts 5 through 12, and at most 4,096 steps at 64 Hz.
- VG033 deterministic mean is the complete plant action while the held public
  phase is 0. SF016 is the complete training-only plant action at held public
  phases 1 and later. There is no blend, sampling, clipping, fallback, native
  teacher path, checkpoint switch, or student update.
- The actor input remains the legal 4,119-value ABI. Native privileged state is
  read only by SF016 to create the training plant/label and is never stored.
- Store only the warmed 256-value recurrent feature as float16, the four-value
  base pre-tanh output, the four-value oracle action, public phase index, agent,
  native step, and terminal marker. Bind a stable 550-byte record dtype.
- Require exact action-history parity against the selected plant action and
  public phase updates only on their 4 Hz ticks.

This dataset is deliberately non-admissible for deployment. Teacher actions
may only create offline training histories. Every later admission screen and
runtime action must be the deterministic output of one recurrent Puffer actor.

## Corpus admission

Admit the training corpus only with exactly 512 episodes and uniform 5--12
course-count mass, at least 90% full-course completion, zero crash, zero
out-of-order/action/wire/thrust hard fault, at least 95% Gate-1 warm-up reach,
at least 100,000 feature records, and exact equality between feature records
and teacher plant actions. Require nonzero records for public indices 1 through
11, finite features/actions, oracle actions inside the legal envelope, exact
plant action history, and no public-phase decrease, skip, off-tick update, or
encoding error.

A completed rejection cannot be rerun unchanged. Admission authorizes one
separately source-locked teacher-free distillation into learned indexed Puffer
heads; it grants no deterministic screen, shadow, FlightSim, or live authority.

## Source identity and compute gate

Before the first step bind the pushed commit, float32 native extension,
runtime, VG033 checkpoint/report/admission, SF016 report/source, VG061
rejection, 48-hour goal, collector, runner, tests, legal actor/oracle/public
phase sources, evaluator/config sources, and every zero-authority field.
Require the retained Vast RTX 4090 worker, at least 32 visible CPUs, 20 GB free
disk, CUDA, Clang/OpenMP, ccache, a fresh SM89 build, both native suites, and
the focused tests. `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=1` are fixed.

Frozen new source hashes:

- collector: `6a7d738ac0bdea5875b4ecd791efa7123fc4d57e38748a810d2a7c6f20a44672`
- runner: `5fb790bc4b4db8f8acf5ba1d02f592140f0fb587ee6d65c92870630dbb9ca24c`
- focused test: `aca71c05de42bcecc8e5dcaad704b71bd8c1b9f03395615d5a7a49ed1c219ac2`

## Safety

FlightSim packets, VQ2 Training, shadow, Submission, sealed-test access, and
student updates are zero. The training-only teacher action count is measured
and must exactly match stored records. Runtime teacher authority is false.
VQ2 Submission remains forbidden without explicit user authorization.
