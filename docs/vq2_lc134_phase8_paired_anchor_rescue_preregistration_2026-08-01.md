# VQ2 LC134 paired phase-8 anchor/rescue preregistration

LC133 raises stochastic phase-8 progress return by 55.5% but still produces
zero raw-9 passes. Reject its checkpoints. The bottleneck is now causal
credit, not throughput. Collect a new phase-8-specific supervised dataset;
do not fit or relabel the quarantined LC125 raw-10 dataset.

Run one 256-environment paired native vector from LC123 at seed `432050`, with
two identical 128-seed groups in one complete saved-form Puffer actor. Stop at
raw index 9 or native terminal. In group 0, execute the complete LC123 Puffer
action and record phase-8 hidden state plus that same Puffer action as the
failure anchor. In group 1, execute the training-only alignment oracle only at
held phase 8 and record its action as the rescue label. Outside phase 8, both
groups execute the complete LC123 Puffer action.

Admit the dataset only if the seed groups begin byte-identically, control has
zero raw-9 passes, oracle intervention has exactly three paired raw-9 gains
and zero losses, recorded outcomes contain exactly three control failures and
three intervention successes, all records are phase 8, actions are finite and
in envelope, and transport is exact. The actor observes only the unchanged
legal ABI; native state is used solely to generate offline labels. No student
update occurs in LC134.

This is a randomized 24-gate proxy for an approximately 20-plus-gate official
VQ2 course. LC134 sends zero FlightSim packets, authorizes no live attempt,
and keeps Submission forbidden.
