# LC071 phase-2 constrained endpoint screen preregistration

LC070 admits a held-agent constrained decoder endpoint at source-direction
alpha 0.075. It improves failure teacher-action MSE by 1.1567x while limiting
success action drift to 0.001429. LC071 tests whether that offline correction
causes more actual Gate-3 passes.

Use one CUDA context and one 256-agent native vector. Assign eight exact
32-seed groups to interpolation fractions `0, 0.125, 0.25, 0.375, 0.50,
0.625, 0.75, 1.0` from LC062 to the complete LC070 endpoint. Use a fresh
source-locked seed and terminate at the Gate-3 milestone. Every plant action
is the deterministic mean of the corresponding whole Puffer checkpoint;
teacher actions, outcomes, classifiers, and analytic overrides are absent.

Select only a nonzero fraction with at least one additional Gate-3 pass over
the paired LC062 group, no additional crashes, valid transport, and zero
action-envelope violations. Break ties by passes, fewer crashes, then smaller
parameter displacement. A selection authorizes one larger different-seed
Gate-3 confirmation only. It grants no live or Submission authority.

