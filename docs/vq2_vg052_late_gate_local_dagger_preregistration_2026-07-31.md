# VQ2 VG052 Gate-4/5 local-start DAgger preregistration — 2026-07-31

Run exactly one offline collection tagged
`vq2_vg052_late_gate_local_dagger_vg033_visited_512`. Use 512 one-shot agents
on fixed six-gate randomized courses, 2,048 maximum steps, and local starts
uniformly restricted to active gate indices `[3,5)` at offsets `[2,5] m`.
This deliberately supplies Gate-4 and Gate-5 policy states after VG051 measured
zero Gate-5 reach across 256 uninterrupted episodes.

VG033's recurrent deterministic mean emits every complete four-channel plant
action. SF016 is queried only after each visited state and its label is written
to the offline corpus; teacher blend, teacher plant actions, and student updates
are exactly zero. Stored actor inputs contain only the legal 4,118-value camera/
IMU/action-history observation plus held public progress, for the unchanged
4,119-value deployed ABI. Gate-local provenance and native state are never
stored as actor inputs.

Admission requires exactly 512 fixed-six-gate terminal episodes, unit local
reset rate, at least 250,000 records, at least 50,000 records each for public
phases 3 and 4, no phase-0--2 records, exact action-history parity, finite and
bounded query actions, ordered 4 Hz public progress, and zero hard envelope or
transport fault. Crashes and gate completion are diagnostics for this label
collection, not deployment admission. A failed tag cannot be retried unchanged.

VG052 authorizes only a separately source-locked trust-region offline refit from
VG033. It grants no replay, shadow, FlightSim Training, or Submission authority.
Submission remains forbidden without explicit user authorization.

Source lock before the run:

- manifest: `7e246d5e...`
- collector: `a3e12b9d...`
- runner: `9e84b4fd...`
- focused test: `72021570...`
- native environment/header/binding: `246997f5...`/`6c1093b1...`/`65c1e297...`
- native regression: `f7180561...`
- VG051 evidence: `88e56a5b...`
