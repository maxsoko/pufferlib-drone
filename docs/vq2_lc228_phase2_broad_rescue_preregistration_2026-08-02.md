# LC228R phase-2 broad-rescue preregistration

LC216 completes the 24-gate proxy only on the repeated source context. LC227's
fresh default control fails Gate 3 `0/8`, and LC216's phase-2 tensors remain the
known-brittle LC105 tensors. Before fitting, LC228 tests whether a training-only
alignment oracle can causally rescue the broad phase-2 failure distribution.

Run paired `256+256` complete recurrent LC216 actors in one native vector with
seed offset `287`, 32 threads, target raw index `3`, and at most `12,000` steps.
The control group executes LC216 throughout. The intervention group executes
LC216 except while public held phase is exactly `2`, where the offline oracle
owns the plant action. Store legal recurrent features and oracle labels for
both groups; privileged state and oracle actions remain absent from any
deployed actor.

The first LC228 source stopped during feature packaging because the inherited
writer expected a 256-value recurrent state while LC216 carries a 321-value
base/adapter/counter state. It produced no terminal report. LC228R changes only
the corpus projection: retain the base 256 recurrent values and exclude the
64-value phase adapter plus one sequence counter. Use the new tag
`vq2_lc228r_phase2_broad_rescue_001` and do not reuse the empty LC228 directory.

Admit the training corpus only if intervention reaches Gate 3 in at least
`250/256`, gains at least `128` paired completions over control with zero paired
loss, all 512 trajectories contribute phase-2 labels, at least 128 control
failures and 250 intervention successes are represented, and transport,
pairing, finiteness, phase ordering, and action-envelope checks pass. This is
command-free offline training evidence. It grants no FlightSim or Submission
authority and does not prove the official approximately-20+-gate finish.
