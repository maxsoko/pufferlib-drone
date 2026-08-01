# LC118 exact phase-6--9 rescue feature preregistration

LC117 establishes the latest recoverable LC105 horizon: phase-7 intervention
is 0/128 at raw progress 10, while phase-6 intervention creates 4/128 paired
passes with no loss or transport fault. The successful candidate cohort uses
60,082 teacher plant actions across 19 trajectories.

Collect one exact full-batch corpus with saved LC105, 256 agents, duplicated
128-seed cohorts, seed `432050`, 24 proxy gates, 32 CPU threads, CUDA inference,
and 12,000 steps. Both identical cohorts use the training-only alignment oracle
only at held phases 6--9; Puffer owns all earlier and later plant actions and
its recurrent state advances continuously. Record Puffer hidden state, Puffer
pre-tanh mean, teacher action, phase, agent, step, and terminal outcome for each
teacher plant action.

The deterministic duplicated contract must yield 120,164 records/actions from
38 queried trajectories, with 8 raw-10 successes and 30 failures, no censored
trajectory, exact action/progress transport, only phases 6--9, finite labels,
and zero live packets. If any count differs, reject the corpus. Admission
authorizes one whole-Puffer phase-6--9 distillation only; FlightSim and
Submission remain forbidden.
