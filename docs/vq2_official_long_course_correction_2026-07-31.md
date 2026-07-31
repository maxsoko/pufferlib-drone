# VQ2 official long-course correction — 2026-07-31

The operator directly inspected the official VQ2 simulator and reports about
`20` gates, possibly more. Earlier estimates of 11--12 and every synthetic
six-gate completion assumption are superseded for official-lap planning.

## Runtime authority

- The exact course count is unknown before flight and must not be hardcoded.
- Public `active_gate_index` is ordered progress only. Encode it without a
  Gate-16 saturation ceiling and never use it as a finish predicate.
- A valid official lap requires nonnegative `race_finish_time_ns`. The final
  observed index then records the actual course count.
- VQ2 Submission remains prohibited without explicit user authorization.

## Evidence quarantine

VG070 and VG071 are five-/six-gate curriculum diagnostics only. VG071 improved
the candidate from `20/128` finishes and `47/128` crashes to `25/128` finishes
and `39/128` crashes on its fixed short proxy, but it has no long-course,
replay, shadow, FlightSim, or live authority. Its retained hashes are:

- report: `c1494b9c56bdf2cbbfead91dc34d51d45cb780cd6317bc924835a67f7cc6b5e4`
- state: `e126ec1aa12e47a0ad1ff1aa982700fb63f3964324efbb54d4ea249ba432bc91`
- checkpoint: `60677e385cefb6d6af5c957071cb846de8396d8872880a90832d987b7494f010`

## Deadline path

Approximately 36 hours remain. The immediate sequence is:

1. preserve legacy actor behavior while expanding the native gate capacity and
   public-progress representation beyond Gate 16;
2. prove the native teacher, logs, and unbounded legal observation on source-
   locked 20- and 24-gate courses;
3. benchmark serial, vectorized, and asynchronous PuffeRL throughput on Vast,
   retaining measured steps/s and wall-clock speedup;
4. train a count-agnostic recurrent full-output Puffer actor across 20--24+
   gates, using privileged state only for offline labels/reward;
5. require full-course teacher-off screens, replay/export parity, Windows
   zero-command shadow, then one uniquely preregistered bounded Training run
   whose stop authority is official finish time.

No synthetic count, including 20 or 24, proves the official lap. It only makes
the controller and infrastructure compatible with the operator-observed course.

## LC001 throughput correction

The first LC001 remote launcher scoped `OMP_NUM_THREADS=1` around the entire
runner to stabilize a pre-existing bit-exact PyTorch unit test. This also
forced both nominally 32-thread vector rollouts to one CPU worker. LC001 remains
valid teacher/long-course evidence if its predicates pass, but is rejected as a
throughput proof and must not be relabeled. LC002 confines one-thread execution
to the flaky exact test and baseline, requires `OMP_NUM_THREADS=32` plus
`OMP_DYNAMIC=FALSE` inside each vector report, and uses fresh seeds and a new
evidence directory.

LC001 completed with `64/64` collision-free ordered finishes at both 20 and 24
gates, mean native completion times `370.688721 s` and `447.664062 s`, and
maximum unsaturated progress sources `20/6` and `24/6`. Its projected vector
speedup was only `1.177568x`, exactly exposing the inherited one-thread launch
mistake. Count report hashes are `5f7c1b21...` and `b1a83357...`; aggregate
SHA-256 is `c4fa43cf4af7a20ed3728d5ba869a67bc6b7d85c2662ef105d58c0ce151f1903`.
It sends zero FlightSim packets and grants no actor or live authority.

LC002 corrects the thread contract and again passes `64/64` at both 20 and 24
gates. It reaches `219,916` and `193,337` native agent steps/s, or `6.702488x`
the one-agent baseline, with exact `OMP_NUM_THREADS=32` evidence. Aggregate
SHA-256 is `874da287be8451fb84055001d87abec288b6f573783b4d51f8d82c912e208e4c`.
This is a large improvement but misses the preregistered `10x` gate, so LC002
is rejected as the requested Puffer throughput proof. LC003 now measures the
actual persistent-buffer CUDA rollout-plus-optimizer path; it is explicitly
throughput-only because its native benchmark actor consumes the privileged
training suffix and no weight is saved.

LC003 confirms actual CUDA training but rejects at `87,337` end-to-end SPS,
`2.627739x` baseline, and 19% peak GPU. Its profile spends `10.926120 s` in
rollout and only `0.375917 s` in optimizer work; report SHA-256 is
`7646bea6d071aaa02c0ef92844c7be6e37e65c30151754233a5e677ae6a07383`.
LC004 causally scales agents/buffers/threads/minibatch by four to occupy the
full host and feed the GPU. It retains all throughput-only and no-checkpoint
restrictions.

LC004 stopped at wrapper import because the repository root was absent from
`sys.path`; no environment, CUDA context, rollout, optimizer update, or report
was created. LC005 repairs only that executable bootstrap, uses a fresh tag and
seed 431050, and preserves the full LC004 scale and admission contract.
