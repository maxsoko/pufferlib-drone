# VQ2-SF036 Gate-2 causal-horizon diagnosis — 2026-07-28

Tag: `vq2_sf036_gate2_causal_horizon`

SF035 source-locks `359,933` legal oracle labels on the SF033-driven SF034
failure distribution. Its report/metadata SHA-256 values are
`d07db36e11d321962defa217c80e06ff7678c8682f19ea6212d5fc8e3bb8313f` /
`8622f6c9be5cc15fbd46d9e5fd65d86b223406e820d658013f3478ebb8ec9851`.
Episodes range from `300` to `2048` steps with mean `702.994`.

Run a read-only recurrent replay of the frozen SF033 checkpoint over SF035
validation agents `448..511`. Report fixed `1/1/4/1` action error in bins
`[0,256)`, `[256,512)`, `[512,768)`, `[768,1024)`, `[1024,1280)`,
`[1280,1536)`, `[1536,1792)`, `[1792,2048)` and at every cumulative boundary.
Advance recurrent state causally from the true episode start and use valid
records only for metrics.

This diagnosis makes no student update, changes no checkpoint, and cannot
authorize training or a screen by itself. Any causal cap or sampling contract
must be separately preregistered from the measured result. Send zero FlightSim
packets, do not access N712, and do not select Submission.

