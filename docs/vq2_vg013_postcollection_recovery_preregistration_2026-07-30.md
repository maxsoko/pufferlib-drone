# VQ2 VG013 deterministic VG012 postcollection recovery — 2026-07-30

## Failure boundary

VG012 ran once from source commit `107963410014d8bb8009e1fb4ded2dd79adeab11`.
Remote build/ABI and 36/36 tests passed, then all 512 episodes completed and
the writer finalized five arrays plus metadata. Adding the accepted predicate
map exposed a postcollection-only bug: `post_gate_1_records_present` was a
NumPy boolean, so strict JSON report serialization raised `TypeError`.

The failed state remains `collecting` with `resume_count=0`, zero recorded
state counters, and no report. Never invoke its collector resume path: that
would delete the finalized output and rerun the policy unchanged. The exact
state/metadata/log/archive SHA-256 values are `75a89042...`/`e7bfdba1...`/
`c860ba57...`/`c2f868dc...`. The archive was created only after the process
exited and does not modify corpus content.

## One recovery replay

Run exactly one resumable deterministic recovery tagged
`vq2_vg013_vg012_postcollection_recovery_001`. This is not a fresh course
sample or a new policy screen. It recreates seed `429064` and the exact
count-5--12 native vector, then replays the finalized action sequence.

For every nonterminal record, the plant action is read from the next stored
legal observation's newest action-history slot. VG010 is advanced on the
source-locked current observation only to prove action reproduction. The one
unavailable action per episode is the terminal action; it is deterministically
derived from the same VG010 checkpoint/recurrent state on the same CUDA
runtime. Teacher actions remain zero.

Before the first replayed action, the recovery hashes all five arrays and
binds the failed state, metadata, runner log, preservation archive, VG010,
VG011, current source commit, compiled extension, and runtime. It audits the
171,903-record layout, single final terminal per episode, causal public phase,
and finite enveloped query labels.

## Admission

Admission requires every original VG012 hard predicate reconstructed from the
native replay plus all of the following:

- stored active rows and native terminal events match exactly;
- current native image-mask bytes match stored bytes exactly;
- current legal tail and public phase match within `1e-7`;
- VG010 reproduces every recorded nonterminal action within `1e-7`;
- replayed executed-action history matches within `1e-7`;
- all 512 terminal actions are derived and all 512 episodes terminate;
- actor outputs remain finite and inside `[-1,1]`; and
- all stored phase records equal replayed phase records.

Crash and Gate-2 reach retain VG012's diagnostic-only corpus status. A
recovered corpus is not policy admission: later teacher-free screens still
require reliable full-course completion and zero crash. Any replay mismatch
rejects recovery; it does not authorize a second VG012 rollout.

## Safety and next authority

VG013 performs zero new course sampling, teacher plant action, student update,
FlightSim packet, sealed N712 access, shadow, Training run, or Submission
action. Admission can authorize only a separately preregistered source-balanced
refit preserving VG003 clean anchors, VG009 first-round states, and the
recovered VG012 failure states.
