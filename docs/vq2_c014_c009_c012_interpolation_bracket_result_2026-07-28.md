# VQ2 C014 C009-C012 interpolation bracket result — 2026-07-28

C014 is rejected and the interpolation route is closed. All three candidates
were exact whole-checkpoint Puffer interpolants with zero runtime blend and
zero recurrent/action-delivery fault, but none passed Gate 2.

| Alpha | Result | Plane/final diagnostic |
|---:|---|---|
| `0.25` | `0/128`, miss | radial/right/vertical `5.488/-5.263/-1.555 m` |
| `0.50` | `0/128`, miss | radial/right/vertical `3.876/3.820/-0.615 m` |
| `0.75` | `0/128`, timeout | final `[7.561, 15.568, -0.628] m` |

Every screen had zero crash, out-of-order, envelope, nonfinite, teacher,
update, FlightSim, or Submission event.

Frozen artifacts:

- generation manifest SHA-256
  `49c652839f9eb3855222652331022fcbbea5d055281ac1fb79289346732372ea`;
- bracket report SHA-256
  `b39d32f5d2a1b469d7da6949f548758633bd6512f912c30c7d59741c6acc4e81`;
- alpha `0.25/0.50/0.75` report SHA-256:
  - `1e6504a211b5684f8c7ced3500cfcdd72a74026de8875dc36cda4bf8eb81193f`;
  - `9e3af3047d26df99ce5a053608179839091ed4b635bff3e640c6c67d2f34839e`;
  - `616450d536d2c7270c3e6b89271ca680c0e2db19a52d20c16cfad7dbfcb1a5dd`.

Do not refine this scalar line. The sign change still couples lateral and
vertical error outside the `0.75 m` aperture. The next method should optimize
closed-loop return directly on the measured true/alias fixture (recurrent PPO
with legal actor input and a training-only critic), rather than another
one-step supervised fit or checkpoint line search.
