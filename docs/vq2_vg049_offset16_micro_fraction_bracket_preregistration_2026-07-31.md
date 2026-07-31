# VQ2 VG049 offset-16 micro-fraction bracket — 2026-07-31

Run one resumable offline bracket tagged
`vq2_vg049_vg033_to_vg042_offset16_micro_fraction_bracket_001`.
VG048 rejects alpha `0.025` because crashes regress `8 -> 11` and Gate-3
reach `10 -> 9`, but it is the first screened actor to reach Gate 4. Search
alphas `0/0.0025/0.005/0.0075/0.01/0.0125/0.015/0.02/0.025` on the identical
offset-16, 64-episode, four-thread, 2,560-step fixture.

Qualify a nonzero alpha only when both reports pass every hard predicate,
Gate-1/2 reach and crashes do not regress, and finishes, Gate-4 reach, or
Gate-3 reach strictly improves, in that priority order. Rank by finishes,
Gate-4, Gate-3, Gate-2, mean gates, fewer crashes, then smaller alpha.

Every action is the deterministic mean of one recurrent full-output PufferLib
policy over the legal 4,119-value ABI and held public phase. Teacher action,
blend, sampling, clipping, analytic fallback, optimizer updates, FlightSim
packets, and sealed-test access are zero. A selection authorizes only a
different-offset confirmation.

Source SHA-256 values:

- VG048 rejection:
  `a8f5b91f24d8713348e860149520300438175dfe5b7bb1d4f0877cb5e60bf440`
- micro-bracket wrapper:
  `5064814b773c88997df2335701e1ee54c48796cfe6604e13fd913499544e8ca7`
- runner:
  `8ca229f36f2480cc1b5b38753a43b7cd00348ed511873789f6de865e259bc34c`
- dedicated test:
  `74f45ad16d33516a5c1f7c277b65b17e17360c9e2d6621a53bfba1d76e162227`

Require the exact pushed commit, retained Vast environment, both native
regression suites, fresh SM89 float32 build, CUDA, focused tests, and exact
source hashes. This is offline-only; shadow, Training, and Submission remain
unauthorized.
