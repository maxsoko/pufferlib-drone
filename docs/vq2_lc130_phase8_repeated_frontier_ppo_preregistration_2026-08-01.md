# VQ2 LC130 repeated-frontier phase-8 PPO preregistration

LC129 proves that all LC123 phase-8 entrants lie in relative seeds 0--127;
seeds 128--255 add zero records. Do not broaden the seed cohort again. LC125
also proves that oracle control over phases 8--9 can rescue one of the three
known frontier states to raw 10. The next efficient step is teacher-free
closed-loop training on repeated copies of exactly those states.

Starting from nondeployable LC123, run three 512-environment stochastic
rollouts with native seed-group size 128, creating four exact copies of seeds
0--127. Preserve actor-batch numerics by executing rows 0--255 and 256--511
through two independent complete 256-row LC123 Puffer actors. Outside held
phase 8, execute the deterministic Puffer mean. At phase 8 only, sample
pre-tanh actions with standard deviation `0.01`. Stop each trajectory at raw
index 9 or a native terminal; do not query or execute a teacher.

After the first two rollouts, assign whole-trajectory advantages from maximum
raw progress with a four-point raw-9 bonus and perform four PPO epochs. Train
only the four phase-8 indexed residual tensors at learning rate `5e-5`, clip
coefficient `0.10`, maximum gradient norm `0.5`, and `1e-4` pre-update anchor.
Freeze LC123's phase-6 update, every other phase, the camera encoder, recurrent
base, action head, log standard deviation, and ABI byte-exact.

Select at most one post-update state for a deterministic raw-9 screen only if
its stochastic rank strictly beats rollout 1 by raw-9 passes and then mean
maximum raw index. Training does not promote LC123 or the new state. This job
uses a 24-gate proxy; the official course remains approximately 20 gates or
more and finish status is authoritative. Send zero FlightSim packets and keep
Submission forbidden.
