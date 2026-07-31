# VQ2 VG048 offset-16 paired count-5 confirmation — 2026-07-31

Run exactly one resumable offline confirmation tagged
`vq2_vg048_vg047_offset16_paired_count5_confirmation_001`. Compare VG033
against VG047 alpha `0.025` on 64 concurrent count-5 episodes each, four
threads, 2,560 steps, declared label seed `429162`, and native
`evaluation_episode_offset=16`.

Both measured behavior projections must differ from the offset-8 parent and
candidate hashes `faa02134...` and `b742e602...`. Every action must be the
deterministic mean of one full-output recurrent PufferLib policy over the
legal 4,119-value ABI and held public phase. Teacher action/blend, sampled
action, clipping, analytic fallback, optimizer updates, FlightSim packets,
and sealed-test access are zero.

VG047 qualifies for a separately preregistered count-11 diagnostic only if
both actors prove offset 16 and behavior distinctness, all hard predicates
pass, Gate-1/2 reach and crash count do not regress, and Gate-3 reach or
finishes strictly improve. Failure rejects VG047 without unchanged retry.

Source SHA-256 values:

- parent/candidate manifests:
  `6c9bdb6f07a21944b56faec9c63223185cccd4b14400149d10b9d0445dbbc453` /
  `dcfa3477731a578a6c5a76b49289cb764a10c69f7115317d49cbd4889ad2df47`
- VG047 admission:
  `3204cee23dcc9f28d7e4599fd8a1324441e254a5a3dabf74c2f10dc05eae7852`
- offset-aware component:
  `85f7e2a5c813d816dc2aa974e2df07429a1b525181d2d45b7740772511ec35e1`
- shared comparator:
  `053228cc1955b53c75265a99cd5b575e16e4a08d9e8ff2ec601b0e1291a60d2d`
- offset-16 wrapper:
  `8c7ce186daeb329eda0359aa5b28cb8331d6208aec4331f035ed80354f81c08d`
- runner:
  `347d7a754173b74ea238addd37ca0f37bc218fdad627ecd52c5685093ff207c2`
- dedicated test:
  `9481bf5cbdeb1dc9779bc5471cc0867ae326a7082bc9e2cf8058143625e8bab0`

Require the exact pushed commit, retained Vast environment, both native
regression suites, fresh SM89 float32 build, CUDA, focused tests, and exact
source hashes. This is offline-only. Shadow, Training, and Submission
authority are zero; VQ2 Submission remains forbidden without explicit user
authorization.
