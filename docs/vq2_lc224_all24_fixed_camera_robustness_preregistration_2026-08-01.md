# LC224 all-24 fixed-camera robustness preregistration

LC223 isolated LC222's early failure to randomized camera-extrinsic rotation.
The official VQ2 camera extrinsic is a fixed calibration, measured from the
active build as `1.920944634732011 deg` optical uptilt. Randomly changing that
mount during an episode is not a deployment condition.

LC224 therefore runs one command-free, teacher-free full-24 proxy screen of the
immutable LC216 Puffer checkpoint. It combines every LC223 perturbation that
passed individually: reset-position noise `0.15/0.075 m`, camera dropout `1%`,
edge dropout `0.5%`, rolling shutter `1 ms`, rate-gain jitter `1%`, hover-thrust
jitter `0.002`, and rate-lag/linear-drag jitter `2%`. Camera roll, pitch, and
yaw jitter are exactly zero.

Use paired `32+32` episodes, seed `432224`, environment offset `287`, exact
action transport, and the ordinary `45,000`-step all-24 bound. Promotion
requires LC216 to reach raw index `24` in `32/32`, with zero pre-target terminal
and no transport fault; the LC213 prefix baseline must remain stopped at raw
index `18`. This 24-gate native course is an offline proxy for the official
approximately-20+-gate VQ2 event. It is not an official finish and grants no
FlightSim or Submission authority by itself.
