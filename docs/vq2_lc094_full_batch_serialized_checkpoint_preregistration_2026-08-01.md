# LC094 full-batch serialization-exact checkpoint preregistration

LC093 rejected the saved LC087 state when each policy was evaluated in a
separate batch of 128: both averaged `3.203125` gates and neither reached raw
index 7. This is a batch-size numerical bifurcation, not evidence that LC087's
serialized state was ignored. LC092 independently executed the same saved
LC087 checkpoint over the established 256-agent batch and retained mean
progress `3.421875` with unchanged crash and miss rates.

Run one final 128-pair 24-gate audit in which the saved LC073 actor and saved
LC087 actor are independently loaded, but each forward call receives the full
256-agent observation batch and carries a full 256-agent recurrent state.
Select group 0 actions only from LC073 and group 1 actions only from LC087.
This preserves the established numerical batch context while ensuring every
selected action is the exact output of a serialized checkpoint; post-forward
bias surgery is forbidden. Use seed 431940, 32 threads, and 12,000 steps.

Apply LC093's unchanged promotion thresholds: at least `1/128` mean-gate gain,
one added raw-index-7 pass, no crash increase, no maximum-index regression, and
exact transport. Any promotion remains offline and explicitly batch-context
dependent. Batch-size robustness, FlightSim, and Submission authority are not
granted.
