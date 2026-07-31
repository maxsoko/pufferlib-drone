# LC074 phase-6 capped student-feature preregistration

LC073 is the retained offline frontier and improves mean progress, Gate-3
passes, crash rate, and six-gate reach. Its fresh paired distribution still
has a hard downstream wall: 9/64 candidates reach raw index 6 and 0/64 reach
index 7. Earlier phase-6 labels were collected from the pre-LC073 policy and
are no longer distribution-matched.

Run frozen LC073 on 256 fresh exact 24-gate trajectories with 32 native
threads. The Puffer policy owns every plant action. Query the training-only
alignment teacher only while held public index is exactly 6 and source-lock
the recurrent hidden state, base pre-tanh output, teacher label, agent, step,
and terminal bit. Stop deterministically at the first tick with at least
20,000 records, with fewer than 20,256 records, at least eight distinct query
agents, and a 6,000-step safety bound.

The corpus is admitted only with exact phase transport, finite in-envelope
labels, zero hard native faults, zero teacher plant actions, and zero
FlightSim packets. It authorizes one source-locked phase-6 Puffer decoder fit
only. It grants no rollout, live, or Submission authority by itself.

