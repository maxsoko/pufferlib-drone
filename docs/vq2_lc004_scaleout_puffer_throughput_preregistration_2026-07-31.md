# VQ2 LC004 scale-out PuffeRL throughput benchmark — 2026-07-31

LC003 rejects at `87,337` end-to-end rollout-plus-train SPS (`2.627739x` the
LC001 one-agent baseline), with 19% peak GPU utilization. Its measured profile
spends `10.926120 s` in rollout but only `0.375917 s` in training. The GPU is
starved by environment scale, not optimizer cost.

LC004 changes only the throughput scale and seed:

- 4,096 agents instead of 1,024;
- 64 persistent buffers instead of 16;
- 256 configured CPU threads instead of 128;
- minibatch 65,536 instead of 16,384;
- seed 431040 instead of 431030.

Hidden size 256, one MinGRU layer, horizon 16, two warm-up cycles, 64 measured
rollout-plus-optimizer cycles, float32, 24 gates, teacher blend 1.0, and every
LC003 safety/rejection rule remain fixed. The larger batch should feed the GPU
while the 64 buffers occupy the 255-core host.

Admission remains at least `10x` end-to-end SPS over the exact LC001 baseline,
at least 50% sampled peak GPU utilization, finite logs/losses, exact step
accounting, zero FlightSim packets, and zero checkpoint writes. The native
benchmark actor consumes privileged training values and can never be deployed,
regardless of throughput.
