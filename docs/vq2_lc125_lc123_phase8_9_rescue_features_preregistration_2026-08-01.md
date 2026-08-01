# VQ2 LC125 LC123 phase-8/9 rescue features preregistration

LC124 rejects LC123 as a deployment candidate because it creates no raw-10
pass and loses LC105's sole raw-9 trajectory. It nevertheless changes the
paired tail from one raw-8 trajectory to three, providing a causally new
teacher-free phase-8 state distribution. LC123 remains nondeployable and
LC105 remains the frozen frontier.

In one combined diagnostic and collection run, execute LC123 as a complete
saved-form Puffer over two identical 128-seed groups at seed `432050`. The
control group remains Puffer-only. In the intervention group only, replace the
plant action with the training-only alignment oracle while the held public
phase is 8 or 9. Use 24 proxy gates, 12,000 steps, and raw index 10 as the
milestone. The Puffer recurrent state must still advance on every legal public
observation; the oracle is never a deployment action source.

Capture the intervention group's Puffer hidden state, frozen base pre-tanh
action, oracle action, held phase, agent, step, and post-step terminal flag for
every oracle plant action. This avoids a second native rollout if the new
distribution is recoverable. Admit the corpus only if the intervention creates
at least one paired raw-10 gain with zero loss, every episode resolves, feature
and oracle-action counts match exactly, all records are finite and in-envelope,
only phases 8--9 and the intervention group are stored, and at least two
success and two failure trajectories are present for a whole-trajectory split.

This is training-only offline evidence. The official VQ2 course remains
approximately 20 gates or more by direct simulator inspection; only a
nonnegative official finish time proves a lap. Send zero FlightSim packets and
keep VQ2 Submission forbidden.
