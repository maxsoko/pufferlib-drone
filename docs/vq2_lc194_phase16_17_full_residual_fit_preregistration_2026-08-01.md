# VQ2 LC194 phase-16/17 full-residual fit preregistration

LC193 admitted 715,520 source-locked phase-16/17 records after its
training-only oracle rescued raw index 18 in 256/256 paired trajectories while
the unchanged LC189 Puffer scored 0/256. LC194 independently fits the complete
four-parameter indexed residual head for phases 16 and 17. Each phase uses an
agent-disjoint train/validation split, equal trajectory weighting, 512 Adam
updates, learning rate `0.003`, batch size 4096, gradient norm cap 1, and an
anchor coefficient of `0.001`.

Each fitted row must improve held-out teacher-action MSE by at least 2x, remain
finite, and stay within an L2 delta of 64. All non-target model state—including
the phase-15 recurrent adapter—is inherited unchanged. Numerical admission
authorizes only a teacher-free proxy screen requiring retention of raw index 17
and progression to raw index 18. Runtime authority remains recurrent
PufferLib-only. FlightSim and VQ2 Submission are forbidden.
