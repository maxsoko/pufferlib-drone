# VQ2-C001 N294-prefix visual-suffix exact diagnostic

Tag: `vq2_c001_n294_visual_suffix_exact_128`

Run exactly once, offline, with `128` agents, seed `43001`, the exact measured
first two gates, native `64 Hz` control, a `14 s` continuous clock, zero reset/
course/plant/camera randomization, zero teacher blend, and zero FlightSim
traffic.

N294 SHA-256
`a57ca5f4af1bea5d7236b09d6efd9db3fdacbf87114e09a4f3195aeff4169dc6`
owns every complete action while held public index is zero. SF066 SHA-256
`f4ee6782de66736110c79a892efaa70635a2ad6c14f0bfaeeaba9812da0ae7a8`
advances from reset on the dense legal visual observation and owns every
complete action after the held index advances. The selected action feeds both
policies' next observation. No action is blended or analytically overridden.

Source-lock the helper/evaluator at
`3136d518441ce54404f1fce0884fc4346e49455be5d27e6af8e57e9d2854474f` /
`24497a2ddb5185dc64dc79badfdf48e53fa5ee8cef0c30b0c30c21b7dee4d3f7`.
Focused tests must pass before execution.

The diagnostic contract passes only with finite whole-Puffer actions, zero
envelope violation, executed-action error at most `5e-5`, an exercised N294
prefix for every agent, and an exercised suffix for every agent whose held
status reaches index one.

The harness is admitted for training only if all `128/128` agents pass Gate 1
in order with no earlier terminal. Gate 2 is diagnostic: record success,
collision/miss/timeout, crossing/closest error, transition steps, selector
counts, recurrent trace, and hashes without imposing a success claim.

If Gate 1 is not `128/128`, reject the harness and fix only its legacy-view,
timing, status-hold, or selected-previous-action contract. Do not train. Do not
rerun unchanged, send a FlightSim packet, or authorize Training/Submission.
