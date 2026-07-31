# VQ2 VG050 offset-24 confirmation — 2026-07-31

Run one paired offline confirmation of VG049 alpha `0.0025` against VG033:
64 count-5 episodes each, episode offset `24`, seed label `429164`, four
threads, and 2,560 steps. Both behavior projections must differ from offset
16. Require both hard predicates, no Gate-1/2 or crash regression, and strict
Gate-3-or-finish improvement.

Parent/candidate manifest hashes are `843bea12...`/`d2e08008...`; VG049
admission is `f6f81e89...`; comparator/runner/test hashes are
`d393bd4f...`/`cec1c76c...`/`7db6a0ee...`. Bind the exact pushed commit and
all component/native source hashes.

This is deterministic recurrent PufferLib-only offline evaluation. Teacher
action/blend, optimizer updates, FlightSim packets, shadow, Training, and
Submission authority are zero.
