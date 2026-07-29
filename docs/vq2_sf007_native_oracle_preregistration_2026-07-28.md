# VQ2-SF007 smooth-spline speed preregistration — 2026-07-28

Tag: `vq2_sf007_native_oracle_spline_speed_surface`

SF003--SF006 show that the direct active-gate reference is structurally
unreliable: its target changes discontinuously at each official gate pass.
Before adding another trajectory generator, test the existing continuous
course spline at solve-first speeds. This is a native-oracle diagnostic only;
it writes no action labels or replay.

## Fixed surface

Use the unchanged native course-spline teacher, the fixed six-gate course and
plant, true `0.75 m` gate radius, standard full-start noise, `60 s` horizon,
and zero external actions. Screen target forward speeds
`1.5, 2.0, 2.5, 3.0, 3.5 m/s` on `16` common-seed episodes per cell using seed
`42007`. Do not vary any controller gain in this experiment.

Select lexicographically by:

1. valid completion rate;
2. lower collision and out-of-order rates;
3. mean gates passed;
4. lower missed-gate rate;
5. lower terminal crossing radial error;
6. lower completion time; and
7. lower target speed as the final solve-first tie break.

Only a cell with at least one valid finish may seed a separately tagged
independent screen. If all cells fail, reject the fixed spline and implement
one default-off smooth segment reference law rather than widening this scalar
sweep.

## Safety boundary

Native only. No FlightSim traffic, consumed-test access, student update,
checkpoint admission, label/replay write, or Submission action is authorized.
