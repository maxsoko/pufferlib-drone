# VQ2-SF004 slow direct-oracle preregistration — 2026-07-28

Tag: `vq2_sf004_native_oracle_direct_slow_64`

VQ2-SF003 passed Gate 1 in `64/64` episodes but missed Gate 2 laterally in all
episodes at a mean Gate-1 exit speed of `3.171 m/s`. Its mean terminal lateral
error was `1.534 m`, versus only `0.061 m` absolute vertical error. Retain the
direct active-gate-relative controller unchanged.

## Single feasibility change

Reduce only the privileged pitch teacher's target speed from `4.0` to
`2.5 m/s`. Extend the native episode horizon from `40` to `60 s` so the
approximately `115 m` synthetic six-gate course remains feasible at that solve-
first speed. The time extension is coupled to the lower speed and is not a
controller aid.

Every other SF003 value remains exact, including direct roll/thrust gains,
`0.75 m` apertures, full-course start, configured start-position jitter, fixed
course and plant, seed `42002`, yaw target, zero external actions, and full
teacher blend.

## Decision rule

Run exactly `64` episodes. Retain only `64/64` ordered six-gate finishes with
zero collision, timeout, missed-gate, invalid, and out-of-order results. A
failure rejects SF004 and must identify whether reduced speed improves the
earliest failing gate and lateral crossing error before another controller
change.

A pass is still diagnostic. No BC/DAgger labels may be collected until the
separate `4096`-episode, `>=99.9%`, zero-collision/out-of-order randomized
oracle admission succeeds.

## Safety boundary

Native only. No FlightSim traffic, sealed-test access, checkpoint admission, or
Submission action is authorized.
