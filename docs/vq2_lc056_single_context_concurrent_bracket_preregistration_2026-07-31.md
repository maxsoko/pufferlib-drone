# VQ2 LC056 single-context concurrent bracket benchmark — 2026-07-31

LC050's four-candidate sequential bracket took 111.757 seconds. LC047 proved
that separate CUDA processes time-slice badly, while LC055 proved that one
wider native vector does not accelerate rare late-phase trajectories.

LC056 replays LC050's exact four alphas, seed, 32 episodes per candidate,
24-gate course, 12,000-step bound, and deterministic mean Puffer policies.
Build all actors in one process and one CUDA context. Give each candidate its
own seed-identical 32-agent native vector, then run the four vector loops in
four Python threads; each native step releases the GIL and may use 32 OpenMP
threads. Candidate comparisons remain paired rather than assigning different
environment rows to different alphas.

Require exact equality with LC050 for candidate state hash, mean progress,
maximum-index distribution, crash/miss/timeout rates, and transport validity.
Infrastructure promotion additionally requires at least `2.0x` aggregate wall
speedup. This benchmark emits no deployable checkpoint, sends zero FlightSim
packets, and grants no live or Submission authority.
