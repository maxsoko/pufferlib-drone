# VQ2 LC183 adapter-owned phase-15 DAgger preregistration

LC182 rejected both the initial and converged phase-local recurrent adapters at
0/128 raw-index-16 passes. Run two independent complete LC181 recurrent Puffer
actors over paired 256-row seed-15 groups. The control group executes LC181
unchanged. The intervention group uses the alignment oracle only as an offline
plant rescue at public phase 15. Capture oracle targets on both LC181-owned
failure states and oracle-rescued states.

Persist only the frozen base Puffer's 256-value recurrent state, its pre-adapter
pre-tanh action, the oracle target, public phase, agent, step, and terminal flag.
The adapter's private 64-value state may determine the collected trajectory but
must not enter the saved training input. Admit the corpus only if control is
0/256, rescue is 256/256, all 512 rows are represented exclusively at phase 15,
the paired transport is exact, and all labels are finite and in envelope.

This is offline training evidence. It sends no FlightSim packet and grants no
live Training or Submission authority.
