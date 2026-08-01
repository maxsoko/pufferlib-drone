# VQ2 LC175 phase-15 alignment-return PPO preregistration

Three phase-15 DAgger fits and their source-locked interpolation direction fail
teacher-free despite a 256/256 training-only oracle rescue. Run three closed-loop
rollouts from LC173 over 512 exact seed-15 trajectories. Explore only at phase
15 with source-measured anisotropic noise and update only the phase-15 Puffer
residual MLP. Native gate alignment, crossing, and invalid-state geometry may
shape offline returns; no teacher action may enter the plant or PPO target.

Admit training only if transport is exact, all two PPO updates are finite,
nonzero, within the `4.0` delta cap, phase returns have nonzero variance, and
all non-phase-15 parameters are bit-exact. A candidate exists only if a later
rollout improves the baseline closed-loop rank. It then earns one deterministic
raw-index-16 screen, not live authority. No FlightSim packet is sent and VQ2
Submission remains forbidden.
