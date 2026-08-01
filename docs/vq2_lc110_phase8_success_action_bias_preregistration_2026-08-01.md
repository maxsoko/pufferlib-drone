# LC110 phase-8 successful-action bias preregistration

LC109 rejects the LC107 teacher-fit direction: scale 1.5 is outcome-flat and
scales 2--3 lose the only raw-index-9 success. The admitted LC106 corpus gives
a causally different signal. During the first 20% of phase 8, successful
LC105 trajectories average Puffer actions `[-.2334,+.2235,+.0136,+.0021]`;
failures average `[-.0777,-.9843,+.3086,+.0007]` in pitch/roll/thrust/yaw
order. Success therefore correlates with more positive roll, lower thrust, and
slightly more negative pitch—the opposite of the rejected teacher family.

Screen four complete saved-form Puffer policies: LC105; roll `+.05`; weak
combo `[-.005,+.025,-.01,0]`; strong combo `[-.01,+.05,-.025,0]`. Use 128
paired seeds per group, seed 432100, pairwise 256-row actor execution, target
raw index 9, and at most 12,000 steps. No teacher, outcome label, native-state
action, post-forward surgery, or analytic override executes.

Select only a nonbaseline candidate with one added pass, no added pre-target
terminal, and exact transport. Selection authorizes one independent
pairwise-256 confirmation only. FlightSim and Submission remain forbidden.
