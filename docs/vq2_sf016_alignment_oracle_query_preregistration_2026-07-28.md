# VQ2-SF016 alignment-oracle query parity preregistration — 2026-07-28

Tag: `vq2_sf016_alignment_oracle_query_parity_64`

SF015 rejects direct promotion of SF014: `21/64` courses pass Gate 1 and none
pass Gate 2. All actions are finite and there are no collisions or timeouts;
the failure is closed-loop BC distribution shift. The next minimal bridge is
DAgger, with SF014 driving the plant and the admitted SF009 oracle labeling the
same visited states.

Before collecting those labels, SF016 must prove that the standalone
training-only Python query reproduces the native oracle.

## Query contract

The query reads only the native `34`-value privileged training tail. It decodes
active-gate-relative position, world velocity, and attitude, then evaluates the
exact memoryless SF009 equations and fixed SF011 plant constants. It returns
pitch, roll, thrust, and yaw. This module is never imported by the actor and
none of its privileged input may be persisted into a student dataset.

Run `64` new randomized full-course oracle episodes, one per agent, seed
`42016`, fixed `0.75 m` aperture and fixed plant. Before every active native
step, compute the Python query from `O_t`. Execute the unchanged native oracle
with zero policy input and blend one. Recover the actual executed native action
from the newest action-history slot of `O_(t+1)` and compare every channel.

Run once:

```bash
.venv/bin/python scripts/validate_vq2_alignment_oracle_query.py
```

Frozen SHA-256 values:

- Python query: `877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694`;
- parity evaluator:
  `a4be8488a80a715858a70d637601ed7109338c5e5b73ed3cb74a58eff71a5c25`;
- query tests: `a9578d5987ad9a4e18b6cbef572819d6abc065c761632013b775b767990150d1`;
- parity tests: `025e25b431910892834b7fd8ff60b85555d6c9edcd547dcf65921e66027ba490`;
- admitted native controller:
  `47f5c40d0cd04e59caf0e3aad604847a8273deb7c4e64f6f683c2a813b2ab5b4`;
- legal/privileged ABI:
  `35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a`;
- shared oracle evaluator:
  `98e099a3286173322b56054a98b095faa4366cedd72359516ead606cdb37863f`;
- SF015 report:
  `b32be4f4c1c590ad085ed03dfb525f270e05f3e745cae50c1bcd5ab01ec3aa78`.

The query/parity/oracle/ABI suite passes `24/24` before the native parity run.

## Admission

Require `64/64` native oracle six-gate finishes with every SF011 safety and
action-envelope diagnostic zero. Require finite query results and maximum
absolute Python-versus-native action error at most `5e-5` over every active
transition and every channel.

A pass admits this query only to label a separately tagged SF014-driven DAgger
dataset. It does not admit SF014, a retrained child, a native teacher blend in
any policy screen, or any FlightSim action.

## Safety boundary

This parity run persists only aggregate errors and metrics: zero labels, zero
student actions, zero updates, and zero checkpoints. Send zero FlightSim
packets, do not touch N712, and do not shadow, reset, arm, setpoint, run a
bounded attempt, or select Submission.
