# VQ2 VG014 three-source recurrent refit preregistration — 2026-07-30

## Candidate and immutable sources

Run one epoch-resumable offline refit tagged
`vq2_vg014_variable_gate_three_source_refit_001`, seed `429065`. Start from
admitted VG010 checkpoint SHA-256
`a093475371a94bc284310bbe64609fe515a35e50be0e68a7fdfb781b55f03822`.
Use only these legal recurrent datasets:

- VG003 clean oracle trajectories: 2,396,907 records;
- admitted VG009 VG005-visited states: 123,992 records; and
- VG013-recovered VG012 VG010-visited states: 171,903 records.

VG007/VG008 remain quarantined. The recovered VG012 arrays are loaded through
the external admitted VG013 report SHA-256
`38358a5813b7d6811d186006bc44a2e4470e51ad5a3d9c462bef4516b2b069c9`;
the original failed VG012 state is never resumed or rewritten.

## Fixed optimization contract

Every optimizer update pairs one recurrent chunk from all three sources. Each
source loss is normalized by its own valid records before mixing at exact
weights `0.50` clean, `0.25` VG009, and `0.25` recovered VG012. Raw corpus
volume cannot set source weight. The clean stream defines one epoch; both
failure streams cycle only at complete episode-batch boundaries.

- 12 epochs, four agents per source chunk, 256-step BPTT;
- three identical optimizer exposures for any chunk triplet containing a
  causal public-phase transition;
- AdamW learning rate `2e-5`, weight decay `1e-5`, gradient cap `1.0`;
- temporal smoothness weight `1e-4` and action weights `(1,1,4,1)`; and
- disjoint validation: final 32 VG003 agents and final 64 agents from each
  failure corpus.

The unchanged VG010 parent is epoch zero. Select the epoch with minimum
`0.50/0.25/0.25` validation weighted MSE. Numerical admission requires a child
epoch, lower balanced validation than VG010, strictly lower recovered-VG012
validation MSE, clean validation MSE at most `0.02`, VG009 validation MSE at
most `0.02`, exact source weights in every epoch, and at least `3.0x`
transition exposure.

## Source lock, resume, and safety

Before epoch zero, the state binds the current Git commit, trainer/runner/
preregistration, all dataset metadata and admitted reports, VG009/VG010/VG013
admission evidence, parent checkpoint/report, model/runtime contract, source
splits and weights, phase audits, optimizer/RNG state, and zero-authority
safety fields. Once training state exists, resume performs no install, build,
test, or source mutation. A completed resume additionally binds checkpoint,
report, and terminal state hashes.

Remote preflight requires exact evidence hashes, a clean pushed commit, CUDA
on the retained supported GPU, at least 32 CPUs, about 64 GiB RAM, 20 GiB free
disk, both native regression suites, a fresh float32 vision build/ABI, and all
focused tests.

This is numerical fitting only. Teacher plant actions, FlightSim packets,
sealed N712 access, command-free shadow, bounded Training, and Submission are
all zero. Admission can authorize only a new teacher-free native screen on
fresh course seeds.
