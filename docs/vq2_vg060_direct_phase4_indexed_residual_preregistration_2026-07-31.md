# VQ2 VG060 direct warmed phase-4 indexed residual fit — 2026-07-31

Close VG057's shared-head direction. Load VG033 into the indexed recurrent
Puffer actor and keep every base parameter and indexed head exactly frozen at
zero except head 4. Use only VG039's full-start, VG033-visited phase-4 rows for
loss, so the learned correction is fit on warmed deployment-shaped state and
cannot change any action before official index 4.

Run 12 epochs, seed `429192`, 256-step causal chunks, eight agents per batch,
64 validation agents, and AdamW `1e-3`. Only phase 4 has row weight `1`; all
other phases have zero. The optimizer tensor contains the complete indexed
table, but the loss has zero gradients outside head 4 and all other heads must
remain bit-zero.

Numerical admission requires at least 1,000 phase-4 validation rows, strict
phase-4 MSE improvement, exact phase-0--3 validation outputs, exact base
parameters and non-target heads, and head-4 L2 at most `16`. A pass authorizes
only a teacher-free residual-scale bracket. FlightSim, shadow, Training, and
Submission are forbidden; Submission requires explicit user authorization.

Source lock before the run:

- VG039 report/VG059 rejection: `a0bdcd0d...`/`a1810662...`
- generalized core/VG060 wrapper/runner/test: `46052481...`/`576a07e4...`/`c1f97021...`/`070b2e24...`
