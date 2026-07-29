# VQ2-SF010 randomized-course oracle preregistration — 2026-07-28

Tag: `vq2_sf010_native_oracle_randomized_course_512`

SF009 passed `64/64` fixed-course episodes with zero safety event. Before the
4096-episode oracle admission, run one disjoint course-randomization screen and
add diagnostics that do not alter dynamics.

## Diagnostic-only instrumentation

For each ordered crossing, record sampled count and mean radial/right/vertical
error separately for all six gates. Mark an episode as a crossing-margin
violation if any accepted crossing has radial error above `0.50 m`. Also mark
separate episode-level violations for a nonfinite/out-of-`[-1,1]` executed
action, a post-clamp body-rate wire command outside the configured limit, or a
collective-thrust command outside `[sitl_min_thrust, sitl_max_thrust]`. These
fields are logs only and must not feed observations, rewards, actions, or
termination. Existing native regressions must remain green.

## Fixed screen

Reuse the exact SF009 alignment-governed controller and fixed plant. Keep every
gate radius at `0.75 m`, standard full-start noise, zero external actions, and
a `180 s` horizon. For every episode independently:

- sample course geometry scale uniformly from `0.35..1.0`;
- jitter every gate by up to `+-3 m` forward, `+-5 m` lateral, and `+-0.5 m`
  vertical; and
- retain monotonic nominal `22 m` spacing and all other SF009 settings.

Run exactly `512` episodes with seed `42010`. Admission requires `512/512`
ordered finishes, zero collision/miss/out-of-order/timeout, zero crossing-
margin and envelope violations, all six crossing sampled rates equal to one,
and each gate's mean radial error at most `0.10 m`.

A pass admits one disjoint 4096-episode screen of the same fixed distribution.
A failure requires localization before any controller change. No label,
replay, or checkpoint may be written.

## Safety boundary

Native only. No FlightSim traffic, consumed-test access, student update,
checkpoint admission, or Submission action is authorized.
