# VQ2 C012 C011 aggregate warmed DAgger fit result — 2026-07-28

C012 is numerically admitted and authorizes one paired offline handoff screen.

- selected epoch: `11` of `16`;
- optimizer updates: `15,983`;
- Gate-2/phase-zero held-out weighted MSE:
  `0.0006618027711765891/0.00024169211826505986`;
- all seven source-specific audits pass;
- C006 true/alias weighted MSE:
  `0.00033604980658482085/0.0013414051020804324`;
- C011 true/alias weighted MSE:
  `0.00042270107198083865/0.0003882853130139293`;
- FlightSim/Submission packets: `0/0`.

Frozen artifacts:

- checkpoint SHA-256
  `2362ed2250ce41743d9153c64d7ccd3760171a468733c1b62cf34ebb2694f9f0`;
- report SHA-256
  `d7d6a66ea9f686a53d3516a11d8651b215805835327be8a79899ea455457d8f4`.

Run C012 on the frozen true-range handoff first. Stop and reject on any miss;
run the fixed alias half only after a perfect true-range result.
