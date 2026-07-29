# VQ2-SF063 current-distribution DAgger fit — 2026-07-28

Tag: `vq2_sf063_current_dagger_fit_001`

Fine-tune frozen SF059 as one recurrent 4,119-input, four-output Puffer actor.
Use only two source distributions: one copy of the exact measured-course SF049
oracle prefix as the nominal anchor and six logical copies of admitted SF062 as
the current-policy correction. Interleave real episode boundaries on the agent
axis; never splice recurrent histories. The sixfold current source makes its
short Gate-2 segment comparable in row count to SF049's long clean segment.

Use seed `42063`, CUDA, `16` epochs, learning rate `2e-5`, final eight local
agents from each of the seven logical source groups for validation, phase-zero
loss weight `1`, Gate-2 loss weight `2`, and train the encoder, phase embedding,
recurrent core, action head, and phase residual. Keep the four-output ABI,
teacher blend zero, and all deployment constraints unchanged.

After the aggregate selection, separately audit held-out local agents `56:64`
from both SF049 and SF062. Require each source's phase-zero and Gate-2 weighted
MSE at most `0.01`, every action-channel MSE at most `0.05`, finite outputs,
and the ordinary aggregate numerical gate. Failure rejects SF063 without a
rollout screen. Passing permits only a separately source-locked exact native
screen. Send zero FlightSim packets, never access N712, and do not authorize a
shadow, bounded flight, or Submission.
