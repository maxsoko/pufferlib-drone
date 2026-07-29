# VQ2 C004 measured handoff result — 2026-07-28

C004 validates the recurrent replay and whole-action handoff, but rejects the
native course guard configuration before any policy conclusion.

Both true-range and live-alias fixtures:

- replay N294 and SF066 prefixes with maximum error `0.0`;
- begin at phase `1/6` and select SF066's complete action;
- deliver that action with maximum error `0.0` and no envelope/nonfinite fault;
- execute exactly one plant step; then
- terminate all `128/128` agents with `out_of_order=1`, zero collision, zero
  miss, and zero timeout.

The direct cause is that `num_gates=6` retained the checkpoint's phase scale,
but only gates 0 and 1 were assigned. Native gates 2–5 therefore defaulted to
the origin. The first positive-X motion crossed those future zero planes and
correctly triggered the fail-closed ordered-course guard. This is not SF066
behavior and the reports' `next_gate_passed=false` is not a policy screen.

Frozen artifacts:

- true-range report SHA-256
  `32e3b89917f275b7901030c1d20b8a29ed262129b6afd3eb581070df8335def3`;
- live-alias report SHA-256
  `79d8a543cb14e8386175a6998a83012a95d2b2e3a6a8824f40018cb99b1f8441`;
- true-range trace SHA-256
  `73ecb46331645746c77b40e272df4ebe74058a4c606e0e8181dff17c418f8d7b`;
- live-alias trace SHA-256
  `5409005008bb844a8194726a7f30a005c344c3a358bde9f1e009650d100ebe49`.

For C005, assign gates 2–5 explicit inert positions far beyond the measured
Gate-2 plane. They exist only to keep the historical `/6` feature scale and
cannot affect the plant before the evaluator stops at the `2/6` transition.
Add a focused native assertion that one zero-action step remains nonterminal.
No other fixture input changes. C005 is the final fixture execution; another
configuration recovery is forbidden.

C004 sent zero FlightSim packets and executed no teacher, student-update, or
Submission action.
