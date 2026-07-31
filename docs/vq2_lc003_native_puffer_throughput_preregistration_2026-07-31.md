# VQ2 LC003 native PuffeRL throughput benchmark — 2026-07-31

LC001 and LC002 prove the long-course teacher but raw Python-driven `cpu_step`
scales only `1.177568x` and `6.702488x` over the one-agent baseline. LC003
measures PufferLib's actual persistent-buffer CUDA rollout-and-train path on the
retained Vast RTX 4090.

Run one source-locked throughput-only benchmark on a 24-gate native course:

- 1,024 agents, 16 asynchronous buffers, 128 CPU threads;
- horizon 16, minibatch 16,384, hidden size 256, one MinGRU layer;
- two warm-up and 64 measured rollout-plus-optimizer cycles;
- float32 CUDA backend and CUDAGraph warm-up/capture setting 1;
- source-locked LC001 one-agent report as the baseline;
- poll GPU utilization during the measured interval.

The native teacher remains at blend 1.0 so this benchmark cannot create a
deployment candidate. The native CUDA policy also consumes the complete
training observation, including privileged suffix values; therefore no weight
may be saved, promoted, replayed, shadowed, or used in FlightSim. LC003 measures
framework throughput only.

Admission requires finite losses, exact step accounting, at least `10x`
end-to-end rollout-plus-train SPS relative to the LC001 baseline, maximum GPU
utilization at least 50%, no FlightSim packet, and no checkpoint write. A miss
must be retained and followed only by a distinct source-locked configuration.
