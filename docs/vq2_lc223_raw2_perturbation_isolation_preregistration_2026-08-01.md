# LC223 raw-2 perturbation-isolation preregistration

LC222 failed at raw index `1` in every episode. LC223 is a command-free offline
diagnostic that changes the target to raw index `2` and independently screens:
an unperturbed disjoint-offset control, expanded reset noise, camera-extrinsic
jitter, camera dropout, edge dropout, rolling shutter, plant gain/hover
randomization, plant lag/drag randomization, and the original combined profile.

Each profile uses the same paired `32+32` Puffer screen, seed `432223`, seed
offset `143`, at most `5,000` native steps, teacher blend zero, and exact action
transport. Passing a profile means both frozen actors reach raw `2` in `32/32`
with zero pre-target terminal. No result authorizes FlightSim control or
Submission.
