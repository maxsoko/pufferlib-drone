# VQ2 LC205 stacked-adapter on-policy DAgger preregistration

LC203 and LC204 reject the LC202 endpoint and every tested output scale, while
LC189 remains 128/128 at 17/24. The next training corpus must therefore follow
LC202's actual adapter-owned failure distribution rather than reuse LC189's
states.

Run two independent 256-row LC202 actors in the exact recurrent context. The
control half uses LC202 actions throughout. The rescue half uses the
training-only native oracle from public phase 16 through phase 17. Capture
teacher labels for both halves only during phases 16–17. Stored student inputs
are the legal-observation base Puffer's 256 recurrent values and its pre-tanh
action before the continuation residual; continuation state and oracle/native
state are excluded.

Admit the dataset only if control fails raw 18, rescue succeeds 256/256 with no
paired losses, both phases have records, actions are finite and in envelope,
and transport/pairing are exact. This authorizes only an offline continuation-
adapter refit. FlightSim and Submission remain unauthorized.
