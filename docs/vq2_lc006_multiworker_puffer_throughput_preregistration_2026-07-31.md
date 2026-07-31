# VQ2 LC006 multiworker PuffeRL throughput benchmark — 2026-07-31

LC005 proves that scaling one PuffeRL engine from 1,024/16/128 to
4,096/64/256 agents/buffers/threads regresses end-to-end SPS from `87,337` to
`46,763`. Its rollout time grows to `84.270638 s`; the single-engine buffer
synchronization ceiling dominates.

LC006 therefore runs six independent LC003-sized PuffeRL workers concurrently
on the same retained Vast host and RTX 4090. Each worker retains 1,024 agents,
16 buffers, 128 configured threads, horizon 16, minibatch 16,384, two warm-up
and 64 measured cycles. Worker seeds are 431060--431065. Only worker 0 polls
NVML, avoiding six concurrent `nvidia-smi` samplers; its samples observe the
whole device. Expected aggregate resource demand is about 14 GB VRAM and 168
empirically active CPU cores.

Aggregate throughput is total measured agent steps divided by the union of the
six recorded measured intervals. Require all six integrity predicates, exact
source identity, at least 50% overlap of measured intervals, at least `10x` the
LC001 one-agent baseline, and at least 50% device peak GPU utilization. Every
worker remains privileged, throughput-only, unsaved, and forbidden from
deployment, replay, shadow, FlightSim, or Submission use.
