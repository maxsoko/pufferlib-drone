# LC117 exact training-oracle rescue horizon ladder preregistration

LC115 and LC116 show that the exact LC105 continuation cannot be rescued when
the training-only alignment oracle begins at phase 9 or phase 8: both paired
screens remain 0/128 at raw progress 10 with exact transport. Do not fit either
failed label window.

Run a descending paired horizon ladder at teacher starts `7,6,5`, in that
order. Every rung reuses the saved LC105 actor, full 256-row actor execution,
two identical 128-seed cohorts, seed `432050`, 24 proxy gates, 32 CPU threads,
CUDA inference, and the 12,000-step raw-10 stop. The control cohort remains
pure Puffer. The candidate cohort uses the training-only alignment oracle from
its rung start through held phase 9. Stop the ladder immediately at the first
rung that creates at least one paired raw-10 gain with no loss, no added
pre-target terminal, and exact teacher/transport bounds.

The first successful rung is the latest measured recoverable horizon and may
authorize source-locked intervention collection for only that phase window.
It does not admit a Puffer checkpoint or authorize FlightSim/Submission. If
all three rungs fail, reject this oracle through phase 5 and diagnose an earlier
horizon or a different teacher before fitting more labels.
