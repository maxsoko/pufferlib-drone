# VQ2 VG026 accelerated fresh teacher-free screen preregistration — 2026-07-30

## Candidate and one-run decision

Screen exactly one numerically admitted VG025 checkpoint under tag
`vq2_vg026_variable_gate_recurrent_teacher_free_256`. Bind VG025 checkpoint,
training report, and admission evidence SHA-256 values
`85671c31...`/`b9d22d28...`/`633a34d5...`.

This is an offline native screen only. It authorizes neither FlightSim packets
nor a Windows shadow, Training run, or Submission selection.

## Thread-acceleration parity gate

Before the first admission episode, run one immutable parity probe tagged
`vq2_vg026_screen_thread_parity_001` on 64 count-5 agents, seed `429111`, for
exactly 128 steps. Execute the same deterministic mean recurrent actor once with
4 native vector threads and once with 32.

Admit acceleration only if discrete episode/action/phase/safety fields match
exactly; all metrics and action summaries differ by at most `1e-7`; both runs
execute exactly 8,192 policy actions with zero executed-action error; and
optimizer steps, policy-state writes, teacher actions, and FlightSim packets
are zero. The report binds the source commit, runtime, compiled extension,
candidate, admission, checker, wrapper, generic evaluator, and this
preregistration. A failed parity probe forbids VG026.

## Fresh screen contract

- Fixed counts `(5, 8, 11, 12)`, 64 terminal episodes per count, and seeds
  `429111/429114/429117/429118`. These seeds are disjoint from all prior
  teacher-free screens.
- Use 32 native CPU threads and two buffers for each sequential 64-agent count.
  The parity gate is the sole authority for superseding the historical
  four-thread execution path.
- The only plant action is the VG025 recurrent deterministic mean. The actor
  observes the unchanged 4,118 legal values plus one 4 Hz held public-progress
  scalar. Both recurrent state and previous-action history advance normally.
- Teacher blend/action emission, sampling, clipping, analytic fallback, and
  student updates are zero. Crossing margin at `0.50 m` remains diagnostic;
  the official aperture is `0.75 m`.
- State is written before count 5 and after every immutable count report.
  Resume skips only source-identical completed counts. A terminal aggregate
  cannot be rerun unchanged.

## Admission

Admit only with at least `231/256` ordered full-course finishes and all four
count screens complete. Require zero crash, out-of-order, non-finite action,
action-envelope, wire-rate, thrust-envelope, phase off-tick/decrease/skip,
phase-encoding, teacher, and action-history parity fault. Executed action error
must remain at most `5e-5`.

Failure is terminal rejection evidence and authorizes only causal offline
diagnosis or a newly tagged visited-state collection. Admission authorizes only
source-locking the recurrent callable and command-free replay/parity work.

## Remote/source gate

Require the exact pushed commit, clean tracked worktree, preserved environment,
CUDA, Clang/OpenMP, ccache, at least 32 CPUs, about 64 GiB RAM, 15 GiB free
disk, both native regression suites, a fresh architecture-matched float32
native build, and the complete focused evaluator/parity test shard.

Frozen new source surface:

- `scripts/eval_vq2_vg026_variable_gate_recurrent_policy.py` —
  `e0df8de31d60e7565483574f1a841777c7c9cee058876ade9e43a14c4c70bbe8`
- `scripts/check_vq2_vg026_screen_thread_parity.py` —
  `48985cdf5399136eea6c67a8c731f6dce790924b7ee09c4bc69f311a712ecf0a`
- `scripts/run_vq2_vg026_vast.sh` —
  `13771ba9de7450583b5f9c224dee8728757fb4b54af872a997f9bc7d1eb45f48`
- `tests/test_eval_vq2_vg026_variable_gate_recurrent_policy.py` —
  `720fcb6b723b589b97ab3f332852ac80c4a727b1691fe35876fc8d622236b02f`
- `tests/test_check_vq2_vg026_screen_thread_parity.py` —
  `7cb6c936bfdca24446b9f49fd29166efc0652b3f42a86c5aaca472ecb7cb49a4`

## Safety

Privileged actor inputs, teacher plant actions, optimizer updates, FlightSim
packets, sealed test accesses, shadow authority, Training authority, and
Submission authority are all zero.
