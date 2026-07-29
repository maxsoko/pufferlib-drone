# VQ2-SF045 full public-phase recurrent fit — 2026-07-28

Tag: `vq2_sf045_public_phase_full_fit_001`

SF042 preserves Gate 1 but remains high-error on Gate 2, and SF044 now supplies
its fresh on-policy prefixes. Continue the same single legal 4,119-input
Puffer actor from SF042. Train its visual encoder, one 256-wide GRU, joint
four-action head, phase embedding, and phase-gated joint residual; freeze only
the unused action log standard deviation. This adds no runtime component,
selector, teacher action, analytic controller, blend, clip, or fallback.

Use SF044 agents 0--447 for training and 448--511 for validation over all 768
stored prefix steps. Balance phase-zero oracle loss at weight `2` and phase
`1/6` loss at weight `1`. Use seed `42045`, 8 epochs, batch 8, chunks 64,
AdamW learning rate `0.00002`, weight decay `0.00001`, gradient clip `1.0`,
action weights `[4,4,4,1]`, and smoothness `0.0001`.

Select an admitted epoch first, otherwise minimum sum of phase-zero and Gate-2
weighted validation MSE. Admission remains phase-zero `<=0.03`, Gate-2
`<=0.02`, and every channel `<=0.05`. A numerical pass permits only a fresh
teacher-free native screen. Send zero FlightSim packets, do not access N712,
and do not shadow, run a bounded attempt, or select Submission.

