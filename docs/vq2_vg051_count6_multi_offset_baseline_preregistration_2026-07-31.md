# VQ2 VG051 six-gate multi-offset baseline — 2026-07-31

Run one offline VG033 baseline on the actual six-gate proxy: 64 episodes at
each native episode offset `0/8/16/24`, four threads, 3,072 steps, and true
`0.75 m` apertures. Record ordered Gate-1--6 reach, finishes, crashes,
misses, and hard transport for every offset and the 256-episode aggregate.

This replaces count-5 interpolation as the active optimization target after
VG050 rejection. Every action is the deterministic recurrent PufferLib mean
over the legal 4,119-value ABI. Teacher action/blend, updates, FlightSim
packets, shadow, Training, and Submission are zero.

VG050 rejection SHA is `0cd7e823...`; evaluator/runner/test hashes are
`cd5e7a88...`/`ee7b7e83...`/`e2fe240f...`. Bind the exact commit, VG033
checkpoint/report/admission, native sources, compiled extension, offsets, and
goal prompt. Results authorize only a separately preregistered rollout-aware
offline repair.
