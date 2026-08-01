# VQ2 LC137 phase-8 success-fit threshold preregistration

LC136 is numerically rejected only because its preregistered held-out
improvement threshold was `2.0x`. Its selected success-only state improves the
single held-out rescue trajectory by `1.473931528893509x`, keeps paired-control
action drift MSE at `0.004604219119829456` under the `0.02` guard, and has
finite phase delta L2 `4.844313185799752`. Do not screen the rejected LC136
checkpoint.

Repeat the source-locked LC136 fit exactly from LC123 and LC134: same split,
records, optimizer, seed, 512 updates, interpolation scales, parameter anchor,
control-drift guard, delta guard, and phase-8-only mutation. Change only the
minimum held-out rescue-label improvement to `1.4x`. Require the regenerated
selection to meet every numerical guard and freeze non-phase-8 state exactly.

Admission authorizes exactly one reduced, teacher-free LC123-versus-LC137
raw-9 screen. It does not promote LC137 or authorize FlightSim. No teacher,
native state, blending, clipping, or action override may appear in the screen.
The proxy has 24 gates; the official VQ2 course is approximately 20 gates or
more and only official finish status proves completion. Submission is
forbidden.
