# VQ2 VG063 horizon-corrected teacher intervention — 2026-07-31

Run exactly one resumable offline collection tagged
`vq2_vg063_horizon_corrected_intervention_features_001`. VG062 proved that the
admitted SF016 teacher is collision-free and centers Gates 2--4, but its 64 s
collection horizon admitted only `1/512` full courses and never observed public
indices 6--11. VG062 is terminally rejected and cannot retry unchanged.

VG063 changes only the fresh seed (`429198`), horizon (`23,040` native steps,
360 s), minimum feature count (`1,000,000`), experiment identity, and the
corresponding 99% success floor. The 360 s horizon matches the already-admitted
VG002 5--12-gate variable-course oracle contract. VG033 still emits the entire
plant action through held public phase 0; admitted SF016 emits the entire
training-only plant action at phases 1+. The compact 550-byte feature ABI,
legal recurrent warm-up, course distribution, no-blend action selection, and
every other VG062 safety/transport predicate remain exact.

Admit only with at least 99% full-course success, zero crash/hard native fault,
exact 512 uniform 5--12-gate episodes, nonzero records at public phases 1--11,
at least one million records, and every VG062 feature/action/public-phase/
authority predicate. The dataset remains training-only and non-deployable.

Before launch bind the pushed commit, VG062 rejection, all inherited VG062
sources/evidence, the wrapper, runner, preregistration, test, runtime, and fresh
float32 extension. Require the retained RTX 4090 worker, 32 CPUs, 20 GB free
disk, CUDA, both native suites, a fresh SM89 build, and focused tests.

Frozen hashes:

- VG062 rejection: `397b614af2bf3aac3d8ef6e9a1dd2180a08af865d17b53d0685d4d309916ce3f`
- shared collector: `1c0b361bb71d3803ce5daa6ac61443f9d16474f96407503545e0319a99ecc70c`
- VG063 wrapper: `31fa9eafb993d54772cd6264e6d1974663d82c9016cc209fbfb21b093e12d3f2`
- runner: `aea2ef193006cc4f8e9a27888667f5e7d7fc4960698b45b5d160c031662d6f94`
- test: `637577d7839b4b895b871ecbb14621c7df08087a696050828e60ec7be3f2c063`

FlightSim packets, shadow, VQ2 Training, Submission, sealed-test access, and
student updates are zero. Runtime teacher authority remains false. Admission
authorizes only a separately preregistered teacher-free indexed Puffer
distillation. VQ2 Submission remains forbidden.
