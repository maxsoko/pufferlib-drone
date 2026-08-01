# VQ2 LC204 stacked-adapter scale bracket preregistration

LC203 showed that the full LC202 continuation correction is too strong: LC189
reproduced raw 17 on 128/128 rows and LC202 ended at raw 16 on 128/128. LC204
keeps LC202's trained continuation GRU fixed and scales only its zero-origin
four-action output head by `0.003, 0.01, 0.03, 0.10, 0.30, 0.60`. Scale zero
is the byte-exact LC189 actor.

Run every candidate as a complete Puffer in the established 256-row recurrent
context. Select the smallest nonzero scale only if it reaches raw 18 on all
128 paired rows, gains all outcomes versus the 128/128 raw-17 baseline, loses
none, preserves transport, and does not increase pre-target terminals.

This is teacher-free offline screening only. It sends no FlightSim packets and
authorizes neither a live Training run nor Submission.
