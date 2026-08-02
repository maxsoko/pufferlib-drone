# VQ2 LC208 stacked-adapter on-policy DAgger-2 preregistration

LC207 rejects LC206 at raw 16 on 128/128 exact-context rows. LC208 repeats the
source-owned DAgger procedure on LC206's new failure distribution: one 256-row
LC206 control and one paired 256-row training-only oracle rescue from public
phase 16 through 17.

Capture teacher labels for both groups only in phases 16–17. Store only the
frozen legal-observation base Puffer's 256 recurrent values and action before
the continuation residual. Admit only if the oracle reaches raw 18 on 256/256,
the control fails, both phases are represented, pairing/transport are exact,
and all labels are finite and in envelope.

This authorizes only an offline continuation-adapter refit. FlightSim and
Submission remain unauthorized.
