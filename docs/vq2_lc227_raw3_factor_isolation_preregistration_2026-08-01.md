# LC227 raw-3 factor-isolation preregistration

LC226 passes plant-only and perception-plus-plant at raw index `3`, while
perception-only, reset-plus-perception, reset-plus-plant, and the full
fixed-camera combination fail. LC227 splits the remaining causes.

Use the same paired `8+8` actors, seed `432224`, environment offset `287`, exact
Puffer transport, and zero teacher blend. Screen control, reset-only, each of
camera dropout `1%`, edge dropout `0.5%`, and rolling shutter `1 ms` alone, then
reset plus gain/hover and reset plus lag/drag. Successful LC226 profiles resolve
before step `2,700`; cap LC227 at `10,000` steps to classify persistent misses
without spending the full all-24 bound. A pass requires raw index `3` in `8/8`
for both actors with zero pre-target terminal. This remains command-free offline
diagnosis and grants no FlightSim or Submission authority.
