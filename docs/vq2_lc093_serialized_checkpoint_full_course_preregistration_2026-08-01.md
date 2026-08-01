# LC093 serialization-exact full-course preregistration

LC092 proved a serialization boundary that prior bias screens did not audit.
LC087 evaluated its candidate by adding the phase-6 bias after the parent actor
forward pass, then serialized that bias inside the indexed Puffer head. The two
forms differ by up to `5.96e-8` in action due to floating-point addition order,
enough to redistribute rare late-gate outcomes. LC092's exact saved-checkpoint
run retained mean progress `3.421875` but observed eight phase-7 entrants and
two successes across duplicated groups rather than the preregistered six/four.

Run a fresh 128-pair 24-gate screen in one native vector with exact paired
initial seeds. Group 0 independently loads and executes the saved LC073
checkpoint. Group 1 independently loads and executes the saved LC087
checkpoint. No post-forward action surgery is allowed. Both recurrent states
advance only for their own active group. Use seed 431930, 32 native threads,
12,000 steps, deterministic means, and target raw index 7.

Promote only if the saved LC087 checkpoint improves mean gates by at least
`1/128`, gains at least one raw-index-7 pass, does not increase crash rate, does
not reduce maximum raw index, and passes all transport checks. A promoted
artifact must reserialize the exact LC087 model state. This grants only an
offline frontier; FlightSim and Submission remain unauthorized.
