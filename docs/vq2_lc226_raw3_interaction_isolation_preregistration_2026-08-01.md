# LC226 raw-3 interaction-isolation preregistration

LC225 proves every reset/perception/plant grouping reaches raw index `2` in
`32/32`; LC224's maximum raw index `2` therefore localizes the failure to the
Gate-3 approach, not the Gate-2 approach.

LC226 repeats LC225's six profiles against raw index `3`, using the same seed
`432224`, environment offset `287`, exact Puffer transport, and zero teacher
blend. To minimize deadline cycle time, use a paired `8+8` diagnostic and at
most `45,000` native steps. A profile passes only if both actors reach raw
index `3` in `8/8` with zero pre-target terminal. A pass is diagnostic only;
it does not establish full-24 robustness or authorize FlightSim/Submission.
