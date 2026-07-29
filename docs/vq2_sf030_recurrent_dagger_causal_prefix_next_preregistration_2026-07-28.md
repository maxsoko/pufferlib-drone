# VQ2-SF030 SF028-driven causal DAgger prefix — 2026-07-28

Tag: `vq2_sf030_recurrent_dagger_causal_prefix_next_512`

SF029 is rejected but improves the closed-loop distribution: total crash falls
from SF023's `0.921875` to `0.765625`, low crash from `0.916016` to `0.152344`,
mean gates rise from `0.220703` to `0.308594`, and two courses pass Gate 2.
Its report SHA-256 is
`b95b671c24ddb64ee688d6686f0d28421709bb282dd62bc03135986484d2adc7`.

SF030 collects the next causal state distribution without retaining SF029's
rare `8161`-step lost tails.

## Fixed collection

- Use SF028 checkpoint/report SHA-256
  `b74e2f1c792fbd0e250a06a9c65b4fdaa3cc4f132e5e6bbab0a4c07a240b0403` /
  `c1ebd267ca0d35ed873287da1378720c316eb8957115465c5813bf284a4c7ccd`.
- Run `512` new randomized courses, seed `42030`, but collect at most `512`
  control steps per agent. Stop an agent at its first native terminal; retain a
  clean partial prefix with no fabricated terminal for every horizon survivor.
- SF028 emits every plant action. Teacher blend/switches remain zero. Query the
  admitted oracle only before each current step and persist only legal
  observation plus full four-channel label.
- Require exact public action-history replay within `1e-7`, finite values,
  contiguous valid prefixes, at most one final terminal, and zero stored
  privilege.

Run once after source-locking implementation and tests.

Frozen SF030 wrapper/test SHA-256 values are
`3ee3a92043f42b6671322377d86ebfb0a125a52edbdd7f9835cacfbd7c202aca` /
`fd10813f3d04dbffa0157470952a647e7faafc0ee9e14ae85670f239405460f4`.
The causal-prefix/next/broad collectors, SF029 loader, and oracle suite passes
`9/9` before the one collection.

## Admission

Require native completed count equal the stored terminal count; every survivor
length exactly `512`; zero timeout, out-of-order, action, wire-rate, thrust,
non-finite, or layout fault; exact label count; and zero stored privilege.
Native crash/miss terminals remain training evidence, not actor admission.

## Safety boundary

Offline native collection only. Send zero FlightSim packets, do not access
N712, and do not shadow, reset, arm, setpoint, run a bounded attempt, or select
Submission.
