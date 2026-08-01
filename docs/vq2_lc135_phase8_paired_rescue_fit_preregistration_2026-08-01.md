# VQ2 LC135 paired phase-8 rescue fit preregistration

LC134 admits 7,500 phase-8 records from six paired trajectories: three LC123
control failures and three training-only oracle raw-9 successes. Fit this new
dataset only; do not use the quarantined LC125 raw-10 dataset.

Start from the complete LC123 Puffer checkpoint. Split agents independently
within each outcome class, retaining two failures and two successes for
training and one of each for validation. At phase 8, train all four indexed
residual tensors toward oracle actions on successful trajectories and toward
the unchanged parent action on failed control trajectories. Weight trajectories
and outcome classes equally. Freeze LC123 phase 6, every other phase, camera
encoder, recurrent base, action head, log standard deviation, and ABI.

Run 512 Adam updates at learning rate `1e-3`, batch size 4,096, gradient norm
`1.0`, and parameter anchor `1e-3`. Every 16 steps, evaluate interpolation
scales `[0.10, 0.30, 0.50, 1.0]`. Select the lowest validation success-label
MSE subject to failure-parent action drift at most `0.00025` and phase delta
L2 at most `64.0`. Require at least `1.20x` held-out success-label improvement.

Numerical admission only authorizes one teacher-free deterministic
parent-versus-candidate raw-9 screen. It does not promote the checkpoint or
authorize FlightSim. The proxy has 24 gates; direct simulator inspection shows
approximately 20 official gates or more, and official finish time remains the
only lap proof. Submission stays forbidden.
