# VQ2 VG066 sparse early-head count-5 bracket — 2026-07-31

Run one teacher-free diagnostic tagged
`vq2_vg066_sparse_early_head_count5_bracket_001`. VG065 rejects its full
11-head fit, but held-out teacher-action MSE improves `3.16x/2.11x/1.97x` at
public phases 1/2/3. Construct a synthetic Puffer checkpoint that copies only
those three learned 4x256 rows; every base parameter stays exact and every
other indexed residual head is zero.

Screen scales `0,0.25,0.5,0.75,1.0` on 64 fresh fixed count-5 courses, seed
`429201`, episode offset 192, four native threads, and the 360 s horizon needed
by the 2 m/s teacher distribution. Every plant action is the deterministic
whole Puffer output. Teacher/native action, blend, clip, fallback, sampling,
and update are zero.

Qualify only if transport is hard-clean, Gate 1 does not regress, crashes do
not increase, and the ordered completion/Gate-5/Gate-4/Gate-3/Gate-2/mean
frontier strictly improves lexicographically. A qualified scale authorizes a
fresh long-course screen only; it grants no live authority.

Bind the pushed commit, VG033, rejected VG065 checkpoint/report/evidence,
goal, native extension/sources, generalized bracket/evaluator, wrapper,
runner, tests, and safety fields. Frozen new hashes:

- VG065 rejection: `777c2da4da877515399f9a0ace1a068ba2c81732f4750502cc0795983e5a12c8`
- generalized bracket: `8e3cbb98807b14c4480b8a1c30d0575d75853bfa22406734e368a542b0388c4c`
- wrapper: `a3b252ca11d9368119388934e94114764301a9e0b4009165984df1966ef6cea4`
- runner: `f11d2d6cccb068cf5e58679ebfcc50ed62d611033303a6823884cb626fc050ea`
- test: `fe68a45bee1ef5e876f0ae326937d902b32791698335613f5635f239559c2ff0`

FlightSim, shadow, Training, Submission, and teacher actions are zero. VQ2
Submission remains forbidden.
