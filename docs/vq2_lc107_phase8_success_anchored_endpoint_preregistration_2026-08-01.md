# LC107 phase-8 success-anchored endpoint preregistration

LC106 admitted 4,736 phase-8 records from four complete LC105 trajectories:
two successes and two failures. Puffer owned every plant action and the teacher
provided training-only labels.

Split whole agents by outcome with seed 432070, retaining one success and one
failure in each partition. Give both outcome classes equal total weight. Freeze
the legal ABI, encoder, recurrent policy, base action head, and all other phase
rows. Fit only phase 8's residual output weight and bias. Successful rows
retain LC105's action; failed rows target the training-only teacher in pre-tanh
residual space.

Use the established ridge/interpolation grid. Select the finite nonzero row
with minimum held-failure error subject to at most `0.00025` held-success drift
MSE and at least `1.05x` failure improvement. Admission authorizes one
teacher-free, pairwise-256 phase-8 scale screen only—never FlightSim or
Submission.
