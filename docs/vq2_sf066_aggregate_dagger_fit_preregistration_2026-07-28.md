# VQ2-SF066 aggregate current-distribution DAgger fit — 2026-07-28

Tag: `vq2_sf066_aggregate_dagger_fit_001`

Fine-tune frozen SF063 as the unchanged recurrent 4,119-input, four-output
Puffer actor. Use one logical SF049 clean-anchor group, three SF062 prior-crash
groups, and three SF065 safe-under-turn groups. The two failure distributions
then contribute approximately the same combined Gate-2 row count as the long
clean trajectory. Interleave intact episode histories on the agent axis; never
splice recurrent sequences.

Use seed `42066`, CUDA, `16` epochs, learning rate `2e-5`, the final eight local
agents from each of seven logical groups for validation, phase-zero loss weight
`1`, Gate-2 loss weight `2`, and the existing trainable actor boundary. Keep the
actor ABI, public held phase, and complete four-output action unchanged.

After aggregate checkpoint selection, separately audit held-out local agents
`56:64` from SF049, SF062, and SF065. Require each source's phase-zero and
Gate-2 weighted MSE at most `0.01`, each action-channel MSE at most `0.05`,
finite outputs, and the ordinary aggregate numerical gate. Failure rejects
SF066 without a rollout screen. Passing permits only a separately source-locked
exact screen. Send zero FlightSim packets, never access N712, and do not
authorize a shadow, bounded flight, or Submission.
