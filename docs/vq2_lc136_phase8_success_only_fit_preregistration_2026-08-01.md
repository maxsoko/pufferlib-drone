# VQ2 LC136 phase-8 success-only fit preregistration

LC135 confirms that treating matched LC123 failures as immutable anchors is
contradictory: the same three underlying seed trajectories pass raw 9 under
the phase-8 oracle. Under the `0.00025` failure-drift cap, held-out rescue-label
improvement stops at `1.0366x`; unrestricted scales exceed `3x`. Reject LC135.

Start again from complete LC123. Use only LC134's three oracle-rescued phase-8
trajectories as supervised targets; retain the paired LC123 control records
only for a reported action-drift diagnostic. Source-lock a deterministic
two-success-agent training split and one-success-agent validation split.
Weight each trajectory equally.

Train all four phase-8 indexed residual tensors for 512 Adam updates at
learning rate `1e-3`, batch size 4,096, parameter anchor `1e-3`, and gradient
norm `1.0`. Every 16 updates, evaluate interpolation scales
`[0.10, 0.30, 0.50, 1.0]`. Select the finite state with lowest held-out oracle
action MSE, subject to phase delta L2 at most `64.0` and paired-control action
drift MSE at most `0.02`. Require held-out rescue improvement of at least
`2.0x`. Freeze every non-phase-8 Puffer tensor byte-exact.

The control-drift cap is a numerical guard, not a deployment safety claim.
Only one teacher-free deterministic LC123-versus-LC136 raw-9 screen can admit
the checkpoint causally. No teacher, native state, blend, or action override
may appear in that screen. This 24-gate randomized proxy represents an
approximately 20-plus-gate official course; no FlightSim or Submission action
is authorized.
