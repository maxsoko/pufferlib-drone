# VQ2 C006 measured visual suffix DAgger result — 2026-07-28

C006 is admitted and authorizes one anchored suffix fit from SF066.

- legal records: `78,293`;
- episode lengths: `526..687`;
- terminal count: exactly `1` per agent, always final;
- true-range agents: `64`;
- live-alias agents: `64`;
- N294/SF066 warm replay maximum error: `0.0/0.0`;
- executed SF066 plant-action maximum error: `0.0`;
- phase decreases/action-envelope/nonfinite faults: `0/0/0`;
- teacher actions executed: `0`;
- student updates: `0`;
- FlightSim/Submission actions: `0/0`.

The training-only oracle labels have mean
`[-0.2976451, -0.3098501, -0.9042886, 0]`; all four complete action channels
are stored even though yaw labels are zero.

Frozen dataset:

- report SHA-256
  `e3c26dc8d8b6812a56994212ee1b392260cbc24bcb112a11140547bdd7ab4f79`;
- metadata SHA-256
  `77c0fe74a93ebc1afd501109e5cb3080a254a5758db7db7b5b4361fc37d8768f`;
- mask SHA-256
  `9a536058b88099232ea52868535fb6cfe46368c58e1505dce3c3fa5faa4c8a9d`;
- tail SHA-256
  `17d9aac1f4a04e210f08d921a2083bd4d853031fde889f99636bc1658083ae84`;
- action SHA-256
  `a666c61e1ea62a5cb1083fa8815747d6ddfb7d4532882130d172b0d12e959b15`;
- terminal SHA-256
  `0f5e44be7cf0e34611780f4d6953fd4a2b65062da5eaf78c85f019204e0c296a`;
- valid SHA-256
  `dbec7d553fc228d308f816aae8c7972d516501d0241869788552dd3f1c4e12b4`.

Train one child from SF066 with five intact recurrent source groups: SF049
clean, SF062 prior-crash, SF065 under-turn, C006 true-range, and C006
live-alias. Use the final eight local agents of each group for validation.
Do not splice sequences or update N294.
