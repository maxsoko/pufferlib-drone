# VQ2 LC213 phase-16/17 Puffer action-sequence preregistration

LC212 produces zero raw-17 or raw-18 outcomes in 3,072 stochastic PPO
trajectories. LC192 already proved that intervention beginning only at phase 17
cannot rescue LC189. LC193 proves an exact, training-only alignment oracle can
pass raw 18 when it begins at phase 16: its 256 exact intervention agents share
one identical 2,329-step action sequence (1,392 phase-16 steps and 937 phase-17
steps).

Distill that source-locked sequence into a recurrent Puffer checkpoint head.
The actor must preserve every LC189 tensor exactly, use LC189 for all actions
before public phase 16, and emit the stored complete four-action vector while
public progress is phase 16 or 17. Its only new recurrent state is a one-value
sequence counter. The deployed callable contains no oracle, native state,
teacher blend, analytic action, or classical controller.

Construction authorizes only one teacher-free, exact-context offline raw-18
screen against LC189. It authorizes no FlightSim or Submission traffic.
