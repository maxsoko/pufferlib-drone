# LC060 phase-2 bias confirmation preregistration

LC059's paired, command-free Gate-3 milestone selected phase-2 pitch bias
`-0.0025`, improving 5/32 to 8/32 in 17.002 seconds with exact transport.

LC060 uses a new seed (`431600`) and keeps the total workload at 256 episodes:
four paired groups of 64. It compares LC048 against pitch biases `-0.00125`,
`-0.0025`, and `-0.00375`, using the same 24-gate proxy, 3,500-step bound,
held 4 Hz public progress, whole-Puffer checkpoint equivalence, and zero
teacher/FlightSim/Submission authority.

An exploratory winner requires at least two more Gate-3 passes than baseline,
no increase in pre-Gate-3 terminals, and exact transport. Prefer pass count,
then terminal count, then bias norm. Any winner still requires a different-seed
paired full 24-gate screen before checkpoint promotion.
