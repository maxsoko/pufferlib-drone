# LC111 phase-8/9 full-residual endpoint preregistration

LC107--LC110 close both low-capacity phase-8 branches: the output-only teacher
fit is flat before losing the sole success, and measured constant biases lose
that success immediately. LC096 also showed that frozen residual features were
the weak point at phases 8 and 9: output-only validation improvement was only
`1.211x` and `1.157x` there.

Use LC095's admitted dense teacher-plant corpus, which contains 62,896 phase-8
and 58,288 phase-9 legal hidden-state records. Starting from frozen LC105, fit
the complete two-layer residual MLP at phases 8 and 9 only: input weight/bias
and output weight/bias. Freeze the legal ABI, encoder, recurrent policy, base
action head, and every other phase. Split whole native agents by index modulo
5. Use 512 Adam steps per phase, batch 4,096, learning rate `.003`, gradient
clip `1`, and an anchor coefficient of `.001` to the LC105 parameters. Evaluate
held agents every 16 steps and retain the best validation checkpoint.

Numerically admit only if both phases improve held teacher-action MSE by at
least `1.5x`, each phase delta L2 is at most 64, and all values are finite.
Admission authorizes one teacher-free pairwise-256 scale bracket only. The
endpoint itself is not promoted and grants no FlightSim or Submission
authority.
