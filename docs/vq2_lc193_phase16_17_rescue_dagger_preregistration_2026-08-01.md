# VQ2 LC193 phase-16/17 rescue DAgger preregistration

LC192 established that the source-locked LC189 whole-Puffer reaches raw index
17 in all 512 paired trajectories, but a training-only alignment oracle first
enabled at phase 17 cannot rescue any trajectory to raw index 18. LC193 moves
that oracle boundary exactly one phase earlier. It runs 256 unchanged controls
and 256 paired training-only interventions from phase 16 through 17, using two
independent complete recurrent LC189 actors so recurrent batching matches the
admitted screens.

The experiment is diagnostic unless the intervention group reaches raw index
18 in all 256 trajectories with no paired losses, exact transport, finite
in-envelope actions, and source-locked features from both phases. The dataset
then contains teacher targets on both unchanged failure states and oracle-owned
rescue states. Any failed predicate rejects the dataset and authorizes only an
earlier offline rescue-boundary diagnostic. Runtime remains recurrent
PufferLib-only; oracle actions are native-training-only. FlightSim and VQ2
Submission are forbidden.
