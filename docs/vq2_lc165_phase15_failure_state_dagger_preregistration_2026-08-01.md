# VQ2 LC165 phase-15 failure-state DAgger preregistration

LC164 exhausts two constant-bias generations at phase 15. Run 256 exact
seed-15 LC162 control trajectories and 256 paired training-only alignment-oracle
interventions to raw index 16 or 27,000 steps. Record legal Puffer state plus
oracle action labels on both the control failure distribution and rescue
distribution; oracle actions may actuate only the intervention half.

Admit the dataset only if control is 0/256, intervention is 256/256, all 512
rows query phase 15, pairing and transport are exact, and every label is finite
and in envelope. This is offline training infrastructure only. It sends no
FlightSim packets and grants no live or Submission authority. The proxy has 24
gates and the official VQ2 course has approximately 20 gates or more.
