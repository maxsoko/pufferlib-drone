# LC102 phase-7 endpoint scale-screen preregistration

LC101 numerically admitted a small phase-7 decoder endpoint in 0.647 seconds.
Its held-failure teacher-action error improved `1.100639x` while held-success
drift was `0.000074779` MSE. This is fit evidence only.

Screen endpoint scales `0,.25,.5,.75,1,1.5,2` using 64 paired full-start seeds
per group, seed 432020, and a 12,000-step bound on the fixed 24-gate proxy.
Every scale is instantiated as a complete saved-form recurrent Puffer policy.
Each of the seven actors executes over the full 448-agent CUDA batch; a group
selects one actor's complete action vector. No post-forward surgery, teacher
action, outcome label, native-state action, or analytic override is allowed.

Resolve each trajectory when it reaches raw index 8 or terminates. Select a
nonzero scale only if it adds at least one pass over LC094, adds no pre-target
terminal, and passes exact action/progress transport. Selection authorizes one
larger full-batch confirmation only—not promotion, FlightSim, or Submission.
