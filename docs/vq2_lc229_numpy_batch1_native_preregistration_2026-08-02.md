# VQ2 LC229 exact NumPy batch-1 native preregistration

LC227 failed Gate 3 with a small Torch actor batch, while LC228R passed Gate 3
for all 256 control environments with the same immutable LC216 checkpoint and
course context. This preregisters a deployment-arithmetic audit rather than a
policy update.

LC229 must execute eight independent `NumpyLC216Policy` objects. Each recurrent
policy receives exactly one legal 4118-value native observation plus the held
public progress value and emits the complete four-action vector. It must not
use batched policy arithmetic, Torch inference, a teacher, privileged state in
the actor input, action blending, action overrides, or FlightSim packets.

Stage 1 uses the LC227 control seed `432224`, native seed offset `287`, eight
fresh full starts, the native 24-gate proxy, and a raw-index-3 target bounded at
10,000 steps. Admission requires 8/8 target passes, zero pre-target terminal,
exact command transport within the existing tolerance, ordered 4 Hz progress,
finite in-envelope actions, and no unresolved environment. Only if Stage 1 is
admitted may the runner start Stage 2.

Stage 2 preserves the same exact batch-1 contract and requires 8/8 uninterrupted
raw-index-24 proxy completions under a 45,000-step bound. Passing Stage 2 does
not authorize FlightSim control. The next mandatory gate is source-locked N399
official-prefix replay, followed by a separately preregistered zero-command
shadow and bounded VQ2 Training decision. VQ2 Submission remains forbidden.
