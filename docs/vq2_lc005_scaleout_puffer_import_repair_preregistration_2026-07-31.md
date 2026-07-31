# VQ2 LC005 scale-out PuffeRL import repair — 2026-07-31

LC004 stopped during Python import before environment creation, CUDA context,
rollout, report, or optimizer update because its executable wrapper did not add
the repository root to `sys.path`. Its output contains only the traceback and
must remain a failed preflight; it grants no throughput evidence.

LC005 adds only the standard repository-root bootstrap, a fresh tag, and seed
431050. The LC004 scale remains exact: 4,096 agents, 64 buffers, 256 threads,
65,536 minibatch, horizon 16, hidden size 256, one MinGRU layer, two warm-up and
64 measured cycles. Every LC003/LC004 throughput, GPU, privilege, checkpoint,
and FlightSim restriction remains unchanged.
