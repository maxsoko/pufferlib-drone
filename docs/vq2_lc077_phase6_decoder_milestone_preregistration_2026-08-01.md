# LC077 phase-6 decoder milestone preregistration

LC076 improves held phase-6 teacher-action MSE by 1.4865x on LC073-owned
states. LC077 tests whether that numerical direction causes actual progress
from raw index 6 to raw index 7.

Use one CUDA context and one 256-agent native vector. Assign eight exact
32-seed groups to LC073-to-LC076 decoder interpolation fractions `0, 0.001,
0.0025, 0.005, 0.01, 0.025, 0.05, 0.10`. Use fresh seed 431770, target raw
index 7, and the ordinary 12,000-step bound. Every plant action is the
deterministic mean of the corresponding whole recurrent Puffer checkpoint;
teacher actions and native state are absent.

Select only a nonzero fraction with at least one additional paired index-7
pass, no increase in pre-target terminals, exact phase transport, and zero
action-envelope violations. Break ties by target passes, fewer terminals,
then smaller parameter displacement. Selection authorizes one larger fresh-
seed index-7 confirmation only. No live or Submission authority is granted.

