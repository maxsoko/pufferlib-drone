# VQ2 LC057 grouped single-vector bracket benchmark — 2026-07-31

LC056 showed that four separate native vectors do not overlap efficiently: it
exceeded 235 seconds without completing, versus LC050's 111.757-second
sequential baseline. LC057 instead uses one 128-agent native vector and one
128-thread OpenMP region, so the native scheduler can occupy one core per live
environment without competing parallel regions.

A default-off vector option repeats environment RNG indices every 32 envs.
Thus groups `[0,32)`, `[32,64)`, `[64,96)`, and `[96,128)` begin from exact
seed-identical observations. Each group keeps its own LC050 candidate actor and
recurrent state, evaluated at the original 32-row numerical batch width. The
binding exposes read-only, non-resetting group log aggregation so each
candidate retains its own outcome metrics.

Require initial observation equality across groups, exact equality with LC050
for every decision field and candidate state hash, clean transport, 128 total
terminal episodes, and at least `2.0x` wall speedup. All new vector and log
features default off or read only. LC057 sends zero FlightSim packets, writes
no deployable policy, and grants no live or Submission authority.
