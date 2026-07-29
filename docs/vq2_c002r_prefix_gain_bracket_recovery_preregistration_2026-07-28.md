# VQ2 C002R prefix-gain bracket recovery preregistration — 2026-07-28

C002A/B/C are execution-invalid because report serialization failed after the
offline plant run. The failure and partial directories are preserved in
`docs/vq2_c002_prefix_gain_bracket_execution_failure_2026-07-28.md`, SHA-256
`13905b3353a67bbc75d626847fd2f686eac051f99014c4c723889b195c1fa313`.

The recovery evaluator differs only by resolving its output path before the
existing non-overwrite check. Its SHA-256 is
`1dbf88ef5da36df32648834b28cb86be25be597cd536efaa96906b55e525a668`.
Focused tests pass `9/9` after the correction.

Run the original fixed values once under new tags:

| Recovery tag | Coupled longitudinal/braking gain |
|---|---:|
| `vq2_c002r_gain10p0_exact1` | `10.0` |
| `vq2_c002r_gain12p5_exact1` | `12.5` |
| `vq2_c002r_gain15p0_exact1` | `15.0` |

Every other input, seed (`43002`), one-agent bound, target, tolerance,
selection rule, and safety condition remains exactly as preregistered in
`docs/vq2_c002_n294_prefix_gain_bracket_preregistration_2026-07-28.md`, whose
pre-run SHA-256 is
`2c15e8e47d134fbabf544060e7cb1aee7e28cfeda253d9e6bc36bf1c6deb5e33`.

No further recovery is authorized for an evidence-writing failure. If C002R
cannot produce complete source-locked reports, stop this calibration branch.
