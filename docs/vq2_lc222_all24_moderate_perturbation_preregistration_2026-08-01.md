# LC222 all-24 moderate-perturbation preregistration

LC222 is an offline, teacher-free screening run of immutable LC216. It sends no
FlightSim packets and does not authorize Submission or a bounded flight.

- Compare frozen LC213 and LC216 in one paired 64-agent native CUDA vector.
- Use 32 episodes per checkpoint, seed `432222`, seed-index offset `143`, and
  require LC216 `32/32` at raw index `24`, zero pre-target terminal, zero paired
  loss, exact action transport, and LC213 stopping at raw index `18`.
- Apply `0.15/0.075 m` reset-position noise, `0.005 rad` camera-axis jitter,
  `1%` camera dropout, `0.5%` edge dropout, `1 ms` rolling shutter, and small
  plant randomization (`1%` rate gain, `0.002` hover thrust, `2%` lag/drag).
- A pass authorizes only a 128-episode disjoint-seed scale-up. A failure rejects
  perturbation admission and keeps live VQ2 control frozen.
