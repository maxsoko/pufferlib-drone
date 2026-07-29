# VQ2 C009 measured visual suffix warmed DAgger fit result — 2026-07-28

C009 is numerically admitted and authorizes only one paired true/alias offline
closed-loop screen.

- selected epoch: `16` of `16`;
- optimizer updates: `12,195`;
- warmed Gate-2 held-out weighted MSE: `0.0005419381338523211`;
- phase-zero held-out weighted MSE: `0.00042963336056655834`;
- C007 parent warmed Gate-2 weighted MSE: `0.09525888199634053`;
- all five source-specific audits pass;
- C006 true/alias weighted MSE:
  `0.0004225234184452917/0.0014346875896252848`;
- FlightSim/Submission packets: `0/0`.

Frozen artifacts:

- checkpoint SHA-256
  `f037e56ec07ae134a87b3cf86bc9e1f3312d2310918cdf4f6fb84a9e96b259f2`;
- report SHA-256
  `296adbfdb8c6e2c8360e872a47f604b54ac7265f7628944c4d8c3b7b426baf34`.

The numerical correction is not flight evidence. Run C009 once on each frozen
C005 handoff and require perfect, fault-free next-gate completion on both.
