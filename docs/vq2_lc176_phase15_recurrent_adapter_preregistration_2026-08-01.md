# VQ2 LC176 phase-15 recurrent-adapter preregistration

LC175's sampled policy reaches raw index 16 on 6/512 trajectories, but LC177
rejects its saved deterministic mean at 0/128. Fit a 64-state recurrent adapter
on the admitted LC172 phase-15 sequence corpus. The adapter consumes
only the frozen Puffer's legal-observation recurrent feature, starts at exact
zero, updates only while public progress is phase 15, and emits a four-action
residual only at that phase. The frozen encoder, base GRU, base action head,
all existing phase residuals, public ABI, and all non-phase-15 actions must be
bit-exact.

Split agents deterministically by `agent % 4`: 384 training and 128 validation,
balanced across the 256 failure and 256 oracle-rescue trajectories. Select the
lowest validation action-MSE epoch from 20 preregistered epochs. Numerical
admission requires finite state, exact frozen base parameters, and at least
`5x` validation improvement. The oracle actions are offline labels only; no
teacher action enters a plant. Admission grants one teacher-free raw-index-16
screen, sends no FlightSim packet, and grants no live or Submission authority.
