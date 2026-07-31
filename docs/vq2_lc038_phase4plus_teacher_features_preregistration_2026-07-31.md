# VQ2 LC038 phase-4+ teacher continuation corpus — 2026-07-31

LC037 promotes a whole-Puffer parent at mean progress `3.71875`, raw index 7,
and crash rate `0.03125`, but full-start student-state data becomes sparse after
phase 4. A source-locked native 24-gate oracle is `64/64`, collision-free.
Use that oracle only as training infrastructure to turn each student-reached
phase-4 prefix into labels for every remaining phase in one vector pass.

Run 128 full-start 24-gate episodes at seed `431380`, 32 threads, and at most
32,000 steps. The exact LC037 recurrent Puffer owns every plant action while
held public progress is 0--3. From held phase 4 onward, the offline oracle owns
the training plant while the same Puffer recurrent state continues to advance
on legal observations. Record phase-head labels only for phases 4--23.

Require all 128 episodes, at least 15% phase-4 reach, at least 10% completed
teacher continuations, crash at most 20%, at least 400,000 finite labels, at
least 1,000 labels for every phase 4--23, exact action/progress transport, and
both action sources partitioned exactly. This corpus may train Puffer weights;
the teacher and privileged state remain forbidden in every admission screen,
runtime controller, replay, shadow, and FlightSim attempt.

LC038 sends zero FlightSim packets and grants no live or Submission authority.
