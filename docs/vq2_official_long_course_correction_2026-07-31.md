# VQ2 official long-course correction — 2026-07-31

The operator directly inspected the official VQ2 simulator and reports about
`20` gates, possibly more. Earlier estimates of 11--12 and every synthetic
six-gate completion assumption are superseded for official-lap planning.

## Runtime authority

- The exact course count is unknown before flight and must not be hardcoded.
- Public `active_gate_index` is ordered progress only. Encode it without a
  Gate-16 saturation ceiling and never use it as a finish predicate.
- A valid official lap requires nonnegative `race_finish_time_ns`. The final
  observed index then records the actual course count.
- VQ2 Submission remains prohibited without explicit user authorization.

## Evidence quarantine

VG070 and VG071 are five-/six-gate curriculum diagnostics only. VG071 improved
the candidate from `20/128` finishes and `47/128` crashes to `25/128` finishes
and `39/128` crashes on its fixed short proxy, but it has no long-course,
replay, shadow, FlightSim, or live authority. Its retained hashes are:

- report: `c1494b9c56bdf2cbbfead91dc34d51d45cb780cd6317bc924835a67f7cc6b5e4`
- state: `e126ec1aa12e47a0ad1ff1aa982700fb63f3964324efbb54d4ea249ba432bc91`
- checkpoint: `60677e385cefb6d6af5c957071cb846de8396d8872880a90832d987b7494f010`

## Deadline path

Approximately 36 hours remain. The immediate sequence is:

1. preserve legacy actor behavior while expanding the native gate capacity and
   public-progress representation beyond Gate 16;
2. prove the native teacher, logs, and unbounded legal observation on source-
   locked 20- and 24-gate courses;
3. benchmark serial, vectorized, and asynchronous PuffeRL throughput on Vast,
   retaining measured steps/s and wall-clock speedup;
4. train a count-agnostic recurrent full-output Puffer actor across 20--24+
   gates, using privileged state only for offline labels/reward;
5. require full-course teacher-off screens, replay/export parity, Windows
   zero-command shadow, then one uniquely preregistered bounded Training run
   whose stop authority is official finish time.

No synthetic count, including 20 or 24, proves the official lap. It only makes
the controller and infrastructure compatible with the operator-observed course.
