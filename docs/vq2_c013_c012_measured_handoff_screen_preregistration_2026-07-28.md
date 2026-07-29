# VQ2 C013 C012 measured handoff screen preregistration — 2026-07-28

Test C012 on the unchanged C005 fixture with no training or simulator action.

Frozen inputs:

- evaluator SHA-256
  `4f86f65e1ff9f31bfea19137602efed13b68ca068b0b8d593031cf81402df34c`;
- evaluator-test SHA-256
  `fb7d31556511ec96f229adfdc680ac2c04f98f6be2946147b99b86fc380a55c3`;
- focused suite: `21/21` passing;
- C012 checkpoint/report SHA-256:
  - `2362ed2250ce41743d9153c64d7ccd3760171a468733c1b62cf34ebb2694f9f0`;
  - `d7d6a66ea9f686a53d3516a11d8651b215805835327be8a79899ea455457d8f4`.

Run up to this pair, stopping on first failure:

1. `vq2_c013_c012_true_range_exact_128`, seed `43013`, zoom/hold `1/0`;
2. `vq2_c013_c012_live_alias_exact_128`, seed `43014`, zoom/hold
   `1.741775393486023/16`.

Each executed half requires `128/128` ordered next-gate completion, zero
failure/crash/timeout/miss/out-of-order event, warm/action-delivery error
`<=5e-5`, zero envelope/nonfinite fault, and zero FlightSim/teacher/update/
Submission action. Passing both authorizes a larger offline audit only.
