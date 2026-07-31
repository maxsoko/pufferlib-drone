# VQ2 VG071 nonlinear residual count-5 multi-offset — 2026-07-31

Run one resumable paired offline confirmation tagged
`vq2_vg071_nonlinear_residual_count5_multi_offset_001`. VG070 selects residual
alpha `0.05` with `17/64` finishes and `14` crashes versus alpha zero's `10/64`
and `28` crashes. Confirm that gain away from the selection episodes before
testing six-gate transfer.

Compare exactly two complete recurrent Puffer actors: behavior-exact alpha zero
and the exact VG070 selected checkpoint. Use four fresh offsets
`216,224,232,240`, seeds `429205..429208`, 32 episodes per offset, four native
threads, five gates, and 23,040-step horizon. Both actors see the same paired
episodes. No teacher, blend, action override, clipping, or analytic controller
is present.

Admit only if all transport checks pass, aggregate Gate 1 reach does not
regress, aggregate crashes do not exceed the parent, and aggregate completed
five-gate episodes strictly increase. A pass authorizes one separate paired
six-gate transfer screen; it grants no replay, shadow, FlightSim, or live
authority.

Bind the pushed commit, VG033, VG069/VG070 source evidence, selected checkpoint,
VG070 admission, goal, evaluator, runner, tests, native source/binding/config,
runtime, offsets, and safety fields. Frozen new hashes are filled before launch:

- evaluator: `e2b2a7c57c3016feb9eade1ea2859da6c07e7de168df380b9dc30ec3fd6d7a83`
- runner: `29ab3077366466a4071eb8aa30a2604f75d1e48868d3f7d6eb73a9c432357d1e`
- tests: `1116bb57211dd241a289115c356cef851045d13f975ab750449a7b722212ae93`

FlightSim, Windows shadow, VQ2 Training, VQ2 Submission, teacher actions, and
sealed test access are zero. VQ2 Submission remains forbidden.
