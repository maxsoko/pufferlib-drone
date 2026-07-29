# VQ2-SF005 native-oracle PD surface preregistration — 2026-07-28

Tag: `vq2_sf005_native_oracle_pd_surface`

SF003 under-reached Gate 2 at `4.0 m/s`; SF004 overshot it at `2.5 m/s` with
the same direct position gain and weak rate damping. The center is bracketed.
Replace serial scalar guesses with one fixed common-seed surface.

## Fixed surface

- target speeds: `2.5, 3.0, 3.5, 4.0 m/s`;
- lateral rate gains: `0.1, 0.3, 0.5, 0.7`;
- lateral position gain fixed at `0.35`;
- direct world-vertical controller unchanged;
- `16` agents and exactly `16` episodes per cell;
- common seed `42005` for all 16 cells;
- `60 s` native horizon;
- six uninterrupted gates, each radius `0.75 m`;
- all SF003 fixed course/start/plant/teacher settings otherwise unchanged; and
- no checkpoint, replay, or label output.

## Fixed selection

Select lexicographically by:

1. higher ordered six-gate success rate;
2. lower collision rate;
3. lower out-of-order rate;
4. higher mean gates passed;
5. lower missed-gate rate;
6. lower terminal radial error;
7. lower completion time; then
8. lower target speed and lower derivative gain as deterministic simplicity
   tie-breakers.

Only a cell with at least one finish can seed a separately tagged independent
screen. The surface itself cannot admit the oracle or authorize label
collection. If every cell has zero finishes, retain the best progress/error
cell only as diagnostic evidence and redesign the lateral reference.

## Safety boundary

Native only. No FlightSim packet, sealed-test access, student update, or
Submission action is authorized.
