# VQ2 LC023 phase-2 student-state head fit — 2026-07-31

LC022 provides 649,205 finite in-envelope phase-2 oracle labels collected on
states reached exclusively by the LC021 selected recurrent Puffer. Fit only
residual head 2 of that exact Puffer for 10 epochs at learning rate `2e-4` and
seed `431230`. Freeze the encoder, MinGRU, shared decoder, distribution scale,
and all 32 other phase heads bit-exact.

Use agent-group-disjoint validation and retain the lowest validation-MSE
epoch. Require finite weights, exact frozen parameters, phase-2 validation MSE
improvement of at least 1.02x, and trainable L2 at most 512. Numerical
admission is only a label-fit result; it does not imply closed-loop promotion.

Do not deploy the full fitted head directly. If admitted, the sole next
authority is a separately source-locked paired bracket of small interpolations
from the LC021 phase-2 head toward this fit. LC023 sends zero FlightSim packets
and grants no live or Submission authority.
