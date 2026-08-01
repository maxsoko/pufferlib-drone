# VQ2 LC140 repeated-seed phase-9 CEM preregistration

LC139 confirms that frozen LC105 reaches raw index 9 on native seed index 15,
but the memoryless alignment oracle cannot rescue that state.  Do not fit the
failed oracle labels.  Add a default-zero native vector seed-index offset and
use it only to replicate seed 15 across one 512-environment vector.  This makes
every row execute the exact LC105 prefix through raw index 9 instead of paying
for 127 uninformative courses per useful phase-9 trajectory.

Run at most four teacher-free cross-entropy generations.  Each row is a
complete recurrent LC105 Puffer actor.  Before phase 9 its action is bit-exact
LC105.  During held public phase 9, add one bounded, trajectory-constant
four-channel pre-tanh residual bias; this is exactly the policy family obtained
by changing only LC105's indexed phase-9 residual output bias.  Candidate zero
is included in every generation.  Rank candidate policies first by raw-10 pass
and then by the native phase-9 return.  Stop immediately after the first
generation with a raw-10 pass.

Admit a checkpoint only if at least one candidate passes raw index 10, all 512
rows query phase 9 from byte-identical initial states, transport is exact, and
checkpoint surgery changes only the phase-9 residual output-bias row.  The
selected checkpoint receives only teacher-free deterministic screen authority;
it receives no FlightSim authority.

The native proxy has 24 gates.  Direct simulator inspection indicates that the
official course has approximately 20 gates or more, and only official finish
status can prove a completed official lap.  LC140 sends no FlightSim packet and
does not authorize Submission.
