# VQ2 LC202 phase-16/17 stacked-adapter preregistration

LC201 rejected every exact-context interpolation of LC199 while LC189 retained
17/24 proxy gates. LC202 changes architecture instead of reopening LC189: it
adds one 64-state GRU adapter that can update and emit only while public race
progress is phase 16 or 17. The existing base policy and proven phase-15
adapter are frozen byte-for-byte.

The only training source is LC193's source-locked camera/IMU/public-progress
feature corpus. Its intervention branch uses the native oracle for offline
labels and plant actions only; no oracle state enters the 256-value hidden
input and no teacher is available to the deployed actor. Train/validation is
split deterministically by agent index. Select the lowest validation MSE over
160 Adam epochs at `5e-4`; require at least 2x improvement, MSE at most `2e-4`,
finite parameters, and exact preservation of every LC189 state tensor.

Numerical admission authorizes only one teacher-free exact-256-context raw-18
screen. It does not authorize FlightSim, Submission, or any live packet.
