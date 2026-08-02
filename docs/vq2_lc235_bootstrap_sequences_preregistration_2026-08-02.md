# VQ2 LC235 remaining-phase sequence bootstrap preregistration

LC233/LC234 proved that a checkpointed complete-action Puffer sequence transfers
through exact deployment arithmetic: seed 287 passed Gate 3 in 8/8 teacher-free
runs. Its first uninterrupted 24-gate run then failed deterministically at Gate
4. With the deadline governing cycle time, LC235 automates the same bounded
offline construction for every remaining proxy phase.

For each raw phase 3--23, LC235 starts two byte-identical seed-287 environments.
Both run independent complete recurrent Puffer policies with all previously
checkpointed sequence heads. Row 0 is teacher-free control. Row 1 may receive
the training-only native alignment-oracle plant action only while held public
progress equals the phase under test. If control passes, no sequence is added.
If control fails and intervention passes, the complete contiguous oracle action
stream is checkpointed for that phase. Any intervention failure, transport
fault, nonfinite/out-of-envelope action, progress decrease/skip, or unresolved
row aborts construction. The oracle never enters policy observations or files.

After phases 3--23, LC235 freezes all sequence arrays in one recurrent Puffer
checkpoint and pure-NumPy callable archive. Admission requires a final separate
teacher-free native screen: eight independent batch-1 callable instances must
complete all 24 proxy gates with zero pre-finish terminal, exact transport,
finite actions, and ordered 4 Hz public progress. Construction results alone do
not admit the checkpoint.

LC235 is offline/native only and sends zero FlightSim packets. Even an admitted
result authorizes only N399 official-prefix replay and a zero-command shadow;
live control remains frozen. VQ2 Submission remains forbidden.
