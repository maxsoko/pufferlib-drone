# VQ2 C002 prefix-gain bracket execution failure — 2026-07-28

The three preregistered C002 processes executed their offline plants but failed
while serializing each report. The CLI passed a repository-relative output
path, while the evaluator called `trace_path.relative_to(ROOT)` without first
normalizing that path to absolute form. Each process raised `ValueError` after
writing `trace_agent0.npz` and before writing `report.json`.

Affected tags:

- `vq2_c002a_prefix_gain10p0_exact1`
- `vq2_c002b_prefix_gain12p5_exact1`
- `vq2_c002c_prefix_gain15p0_exact1`

These are execution-invalid. Do not reconstruct missing native metrics from
their agent-0 traces, do not call them calibration results, and do not rerun
the tags unchanged. Preserve their partial directories as failure evidence.

The only permitted correction is `output = output.resolve()` before the
existing non-overwrite check. The recovery retains the same three fixed gains,
seed, one-agent bound, policies, plant contract, and admission thresholds under
new `vq2_c002r_*` tags. No FlightSim packet, teacher action, or policy update
occurred.
