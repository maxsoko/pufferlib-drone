# VQ2 solve-first preregistration — 2026-07-28

This document source-locks the new solve-first lineage authorized by the user
after N523--N735 was frozen. It does not resume the rejected Dreamer optimizer
as N736. Solve-first artifacts use `vq2_sfNNN_*` tags.

## VQ2-SF001: causal soft-red-mask transformer

Tag: `vq2_sf001_soft_red_mask_contract`

Purpose: implement the smallest competition-legal mapping from one official
TS-002 JPEG to the leading `4,096` values of the recurrent Puffer observation.
This is preprocessing only. It cannot emit an action and creates no FlightSim
authority.

### Fixed input and output contract

- Input is exactly one decoded `640x360` three-channel BGR frame or its JPEG
  bytes. Invalid JPEG data and any other dimensions fail closed.
- The transform is pure and per-frame. It has no history, clock, official gate
  index, target identity, future frame, or mutable detector state.
- Every pixel is scored independently for red hue, saturation, and brightness.
  No contour, component, quadrilateral, corner, range, bearing, pose,
  confidence-ranked target, or morphology operation is permitted.
- OpenCV hue is circular on `[0,180)`. Red hue weight is `1` through distance
  `15` from red, falls linearly to `0` at distance `35`, and is multiplied by
  saturation and value ramps. Saturation ramps from `0` at `40` to `1` at
  `60`; value ramps from `0` at `40` to `1` at `80`.
- Intrinsic normalization scales the full `640x360` image by `0.1` to `64x36`
  using area resampling and places it in rows `[14,50)` of a zero `64x64`
  canvas. It does not crop horizontal evidence.
- This letterbox maps official `fx=fy=320`, `cx=320`, `cy=180` exactly to
  native `fx=fy=32`, `cx=cy=32`. A `360x360` center crop is rejected because
  its horizontal field of view is only about `58.7 deg`, while VQ2's measured
  Gate-2 bearing after Gate 1 is about `30.6 deg` and the official camera
  provides a `90 deg` horizontal field of view.
- Output is a C-contiguous, row-major, flat `float32` array of shape `(4096,)`
  in `[0,1]`. Letterbox rows are exactly zero.

### Preregistered tests

The focused suite must prove:

1. invalid JPEGs and wrong image shapes fail closed;
2. output shape, dtype, contiguity, range, and repeated-call bit determinism;
3. official intrinsic mapping and exact zero padding;
4. black, gray, green, blue, and cyan pixels are rejected;
5. red hue/saturation/value response is monotonic across every soft ramp;
6. simultaneous red candidates at both horizontal edges survive, and adding a
   larger candidate never suppresses a smaller one;
7. JPEG and direct-BGR paths agree within a preregistered mean absolute error
   of `0.01` on a synthetic scene;
8. source contains no contour, component, morphology, corner, or geometry API;
9. all eight retained `640x360` v3391 official-renderer frames decode and
   produce finite, nonempty masks; and
10. the focused suite passes under the repository Linux virtual environment.

The retained official-renderer frame fixture is
`logs/sitl/n254_v3391_gate_alignment/frame_15613.jpg`, SHA-256
`a93d3238490c33a33f1878a2dc47f0bbeb6885d39a9a086b5f4a0d4cb65e2ea0`.
It is from the official v3391 VQ1 telemetry migration, not from the VQ2 course,
so it proves only renderer/JPEG/red-evidence compatibility. The previously
recorded VQ2 passive frame is not present in the current worktree; its recorded
SHA-256 remains
`8a0e9ea39ec6d2f7cfa60119e8243ad5ebfc4d5087aae4a29906741a8a33c061`.
Do not manufacture or silently substitute that missing artifact.

### Admission and next decision

VQ2-SF001 is admitted only if all focused tests pass and a report records exact
source, test, and fixture hashes. Passing admits the transformer for offline
native-policy work; it does not admit a checkpoint, shadow, reset, arm, or
bounded flight.

After VQ2-SF001, audit the current native six-gate oracle at a true `0.75 m`
aperture. Do not collect BC/DAgger labels until the teacher itself meets the
solve-first reliability requirement on held-out full-course episodes.

## Immutable safety boundary

- Never reopen, hash, score, or tune against the consumed N712 physical sealed
  test.
- No post-N522 FlightSim packet is authorized by this preregistration.
- VQ2 Submission remains forbidden.
