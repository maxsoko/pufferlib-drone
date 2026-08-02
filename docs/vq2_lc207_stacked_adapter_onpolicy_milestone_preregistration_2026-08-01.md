# VQ2 LC207 on-policy stacked-adapter milestone preregistration

LC206 fits LC205's adapter-owned phase-16/17 corpus with 223.6x held-out
improvement (`0.000594146` MSE), while preserving every LC189 tensor exactly.
LC207 compares one complete LC189 actor and one complete LC206 actor in the
established 256-row recurrent context on paired seed-15 exact starts.

LC189 must reproduce raw 17 on 128/128 rows. LC206 promotes only if it reaches
raw 18 on 128/128, gains all 128 paired outcomes, loses none, preserves
transport, and does not increase pre-target terminals. Both are teacher-free;
there is no oracle blend, analytic action, clip, or override.

This is offline screening only and sends no FlightSim packets. A pass would
still require independent confirmation before any later admission work.
