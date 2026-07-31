# LC064 phase-2 residual-direction preregistration

LC063 rejects further constant four-action phase-2 biases. LC064 changes the
causal search family: it applies state-dependent whole-Puffer MLP parameter
directions derived from the admitted LC029 oracle-label fit relative to its
LC027 parent, on top of the retained LC062 checkpoint.

Use eight paired groups of 32, new seed `431640`, the 24-gate proxy, and the
3,500-step Gate-3 milestone. Test baseline; full fit direction at alpha
`-0.0025` and `-0.005`; decoder-only direction at `-0.0025` and `+0.0025`;
output-weight-only at `-0.0025`; and encoder-only at `-0.0025` and `+0.0025`.
Negative directions are intentionally opposite the exhausted imitation fit.

The evaluator computes each candidate's exact phase-2 residual from the
recurrent hidden state and its candidate MLP parameters, then adds only the
difference from LC062 before tanh. Thus every plant action is exactly the
output of a complete recurrent Puffer checkpoint; no analytic controller or
teacher emits an action. Advance only a paired Gate-3 gain with no terminal or
transport regression. No FlightSim or Submission authority exists.
