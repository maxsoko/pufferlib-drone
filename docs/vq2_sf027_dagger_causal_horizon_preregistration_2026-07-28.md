# VQ2-SF027 DAgger causal-horizon diagnosis — 2026-07-28

Tag: `vq2_sf027_dagger_causal_horizon`

SF026 is rejected. Its selected checkpoint/report SHA-256 values are
`166c9b28bc1d4848e0e5e6d7028f9637835540cbe2f8e0ae883a8962d8e0e2b7` /
`a1df63ceb0bf7fe85cc50a0010ed32d08395d76bc4f162b38aa900a67c643c03`.
It preserves SF012/SF021 validation (`0.004217/0.004586`) but the complete
SF024 failure trajectories remain at `0.039751`, dominated by late roll and
thrust labels.

SF027 measures whether the residual is distributed across the causal approach
or concentrated in rare late failure suffixes.

## Fixed analysis

- Use only frozen SF026 and held-out SF024 agents `448..511`.
- Replay the complete legal recurrent prefixes without acting on a plant.
- Accumulate pitch/roll/thrust/yaw MSE, fixed `1/1/4/1` weighted MSE, record
  count, and mean visible-mask mass in bins `0..128`, `128..256`, `256..384`,
  `384..512`, `512..768`, `768..1024`, and `1024..1400`.
- Report cumulative complete-bin horizons at `128/256/384/512/1400`.
- Do not use privileged state, write labels, update a model, or access N712.

Run once after source-locking the implementation. This analysis cannot admit a
policy. It may justify a separately preregistered causal-prefix curriculum only
if late-suffix error materially exceeds the first `384` steps while the clean
six-gate dataset remains complete in every later fit.

Frozen analyzer/test SHA-256 values are
`ee59f422a3272b9f463ecacad84123bb86217e5c267db7df3c6f1838aeaa8e26` /
`a927bb4c0585f7b912f6e2ac738a58757b64202e0f4ccbb2145181f8769606fb`.
The horizon/aggregate/legal-loader/recurrent suite passes `14/14` before the
one analysis.

## Safety boundary

Offline read-only replay. Send zero FlightSim packets and do not screen,
shadow, reset, arm, setpoint, run a bounded attempt, or select Submission.
