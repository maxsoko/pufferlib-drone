# VQ2 VG053 late-gate trust-region refit preregistration — 2026-07-31

Run exactly one offline recurrent refit from frozen VG033, tagged
`vq2_vg053_late_gate_trust_refit_001`. Add admitted VG052 as a tenth causal
source without changing the Puffer architecture, 4,119-value legal ABI, action
normalization, or recurrent deployment contract.

Use one epoch, AdamW learning rate `5e-7`, 256-step causal BPTT, four exposures
for chunks containing a public gate transition, and exact source weights:
clean/VG009/recovered-VG012/VG016/VG019/VG024/VG027/VG032/VG039/VG052 =
`0.30/0.03/0.03/0.06/0.08/0.08/0.12/0.15/0.05/0.10`. The first eight sources
retain `0.85` mass; VG052 receives only `0.10`. This is deliberately much
smaller than VG042's four epochs at `2e-6`, whose rollout direction was not
robust.

Numerical admission requires epoch 1 to improve the source-balanced validation
objective and VG052 validation strictly, preserve every prior source cap, keep
VG052 weighted MSE at most `0.20`, reproduce exact source-weight accounting,
and maintain four transition exposures. Numerical admission authorizes only a
new independent six-gate multi-offset rollout screen; it is not rollout or live
admission.

Every training input remains legal camera/IMU/action-history plus public
progress. Native state is present only in the frozen offline SF016 labels.
Teacher blend, FlightSim packets, shadow, Training, and Submission are zero.
Submission remains forbidden without explicit user authorization.

Source lock before the run:

- VG052 admission/report/metadata: `95c3f6e6...`/`687141ac...`/`03b71103...`
- trainer: `a91640c5...`
- runner: `92fd257c...`
- focused test: `1d7efeb1...`
