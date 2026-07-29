# VQ2 C003R handoff recovery preregistration — 2026-07-28

C003A/B are execution-invalid and produced no plant step or artifact. The
failure diagnosis is frozen at
`docs/vq2_c003_handoff_execution_failure_2026-07-28.md`, SHA-256
`a0f6a222dba7e88621d9a8acaa3f0c01f8fa2736e166aefc181923e1e49af2e2`.

The recovery evaluator SHA-256 is
`895599890516a0735ec69169d347f54ea12f70b496a94863730107f1f7cbf3d2`.
It differs only by warming SF066 with 208 calls to its deployment-time
`forward_step`, which replays the C002R suffix outputs with `0.0` maximum
error. N294's warm path is unchanged. Focused tests now pass `17/17`; test
source SHA-256 is
`7445fcb491f661e6aae1d6b0418f898f0c52efcaaf2430e4b1ed483e82de4b43`.

Run the exact original screens once under new tags:

1. `vq2_c003r_true_range_exact128`: zoom `1.0`, hold `0`;
2. `vq2_c003r_live_alias_exact128`: zoom `1.741775393486023`, hold `16`.

Agents (`128`), seed (`43003`), CUDA, every source/state/action input,
fixture-valid criterion, policy admission criterion, and safety rule remain
exactly as preregistered in the pre-run C003 document, SHA-256
`647dd305839f5bc9907a379c2d4bb0a59a54a49dd14a4e5a3dd84d903c51351c`.

No further evidence-writing recovery is authorized for this fixture.
