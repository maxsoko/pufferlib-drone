# VQ2-SF042 public-phase recurrent fit — 2026-07-28

Tag: `vq2_sf042_public_phase_recurrent_fit_001`

SF040 and SF041 show that phase-only input/readout adapters cannot fit the
Gate-2 correction. Retain the same legal 4,119-value input, one visual encoder,
one 256-wide GRU, and one joint four-action output. Start from SF041 epoch 11.
Freeze the visual encoder and log standard deviation; train the existing GRU,
existing joint head, phase embedding, and phase-gated joint residual. There is
still one recurrent Puffer actor and no selector, planner, teacher action,
blend, clip, fallback, or second policy at runtime.

Use SF039 agents 0--447 for training and 448--511 for validation, with the
fixed 768-step causal horizon. Optimize phase-zero oracle loss at weight `2`
and phase-`1/6` oracle loss at weight `1` so the shorter Gate-1 prefix remains
anchored. Use seed `42042`, 8 epochs, batch 8, chunks 64, AdamW learning rate
`0.00005`, weight decay `0.00001`, gradient clip `1.0`, action weights
`[4,4,4,1]`, and smoothness `0.0001`.

Select an admitted epoch first, otherwise lowest sum of phase-zero and Gate-2
validation weighted MSE. Numerical admission requires phase-zero weighted MSE
`<=0.03`, Gate-2 weighted MSE `<=0.02`, and every action-channel MSE `<=0.05`
in both partitions. Numerical admission permits only a fresh offline
teacher-free native screen; it does not authorize FlightSim. Send zero
FlightSim packets, do not access N712, and do not shadow or select Submission.

