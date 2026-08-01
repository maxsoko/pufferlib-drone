# LC103 phase-7 pairwise-batch scale-screen preregistration

LC102 is a valid rejection of its 448-row execution context, not of LC101. Its
LC094 baseline went 0/64 to raw index 8, whereas the established 256-row
context reaches that milestone. This repeats the batch-size numerical
bifurcation already isolated by LC093.

Correct only the actor execution context. Test LC101 endpoint scales
`0,.5,1,2` on 128 duplicated native seeds per group, seed 432030, target raw
index 8, and at most 12,000 steps. For each candidate, execute a complete
saved-form recurrent actor over exactly 256 rows: the 128 baseline observations
plus that candidate's 128 observations. Maintain a separate 256-row recurrent
state per actor and select the candidate half's complete action vector. The
baseline likewise executes 256 rows. No post-forward surgery, teacher action,
outcome label, native-state action, or analytic override is allowed.

Select a nonzero scale only if it adds at least one target pass, adds no
pre-target terminal, and passes exact action/progress transport. This is not an
unchanged LC102 retry: it restores the source-locked numerical batch contract
and reduces the scale family. Selection authorizes only a different-seed
confirmation in the same context, never FlightSim or Submission.
