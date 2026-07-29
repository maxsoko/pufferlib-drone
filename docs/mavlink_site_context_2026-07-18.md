# MAVLink Documentation Relevance Audit — 2026-07-18

This is the durable result of a site-wide relevance scan of the English
MAVLink documentation at <https://mavlink.io/en/> for the AI Grand Prix v3385
controller. It is protocol context, not a replacement for the simulator's
authoritative `VADR-TS-002` contract or measured live behavior.

## Scope

The scan covered the complete English documentation navigation tree:
implementations and language bindings; MAVLink versions, framing,
serialization, signing, routing, redundancy, packet loss, telemetry, XML and
dialects; the standard message catalogs; and every listed microservice. The
pages below were selected for deep review because they can affect this
repository's UDP telemetry, reset lifecycle, or offboard flight control.

The remaining microservices—missions, parameters, manual control, cameras,
gimbals, FTP, landing target, battery, terrain, Open Drone ID, UTM, and similar
payload services—do not participate in the measured v3385 race-control path.
They should not be added without simulator evidence that the corresponding
messages exist.

## Actionable Protocol Contract

### Discovery, identity, and routing

- `HEARTBEAT` advertises liveness and system/component identity. Its rate and
  disconnect threshold are transport-specific rather than fixed by MAVLink.
  Continue transmitting the controller heartbeat even in passive periods.
- Learn the simulator target from received traffic/heartbeat. Do not identify
  component type from component ID alone; use heartbeat type/autopilot fields.
- `target_system=0` is network broadcast and `target_component=0` is broadcast
  to every component of an addressed system. Preserve a learned component zero
  because it is valid for the simulator.
- Keep one controller/source on the UDP link. MAVLink has no general
  cross-link de-duplication, and sequence numbers are per channel.

Sources: [heartbeat protocol](https://mavlink.io/en/services/heartbeat.html),
[routing](https://mavlink.io/en/guide/routing.html), and
[ID assignment](https://mavlink.io/en/services/mavlink_id_assignment.html).

### Command `31000` reset

- `COMMAND_LONG` confirmation `0` means first transmission; `1..255` denotes a
  retransmission. The installed simulator contract's seven zero parameters
  remain correct for its custom command `31000`.
- Standard command reliability uses a matching `COMMAND_ACK`, but the learned
  simulator reset is a custom, potentially non-idempotent command. Never add a
  blind automatic retry merely because no ACK arrives: a duplicate reset can
  start another race epoch.
- A command ACK proves command delivery/result, not the postcondition needed by
  the race runner. Keep the stronger measured proof already in use: normal
  disarm, wait 100 ms, send one learned-target command `31000`, then require
  `sim_boot_time_ms` rollback greater than 1000 ms and a fresh nonnegative race
  start at official index 0/finish -1. If v3385 ever emits `COMMAND_ACK 31000`,
  record it as additional diagnostic evidence only.

Sources: [command protocol](https://mavlink.io/en/services/command.html),
[COMMAND_LONG and COMMAND_ACK](https://mavlink.io/en/messages/common.html), and
[redundancy/de-duplication](https://mavlink.io/en/guide/redundancy_deduplication.html).

### Offboard attitude control

- `SET_ATTITUDE_TARGET` carries a w,x,y,z quaternion, body rates in rad/s, and
  normalized thrust. Bit `128` (`ATTITUDE_IGNORE`) is the correct type mask for
  the deployed body-rate-plus-thrust mode; the identity quaternion is ignored.
- The message timestamp is milliseconds since the sender's boot. The adapter's
  monotonic timestamp is appropriate for the controller process.
- MAVLink defines `ATTITUDE_TARGET` as the reported current commanded target.
  v3385 has not been observed to publish it. If it appears, capture it to check
  applied-command equality; do not require it for flight without a live probe.
- Offboard stacks commonly require a continuing setpoint stream and may leave
  offboard mode if it stops. This supports keeping the independent, fixed-rate
  publisher. It does not define v3385's exact timeout or authorize `>=100 Hz`.

Sources: [offboard control](https://mavlink.io/en/services/offboard_control.html)
and [common message definitions](https://mavlink.io/en/messages/common.html).

### Telemetry validity and transport diagnostics

- `HIGHRES_IMU.fields_updated` says which sensor values changed. The measured
  simulator value `63` marks all three accelerometers and all three gyros as
  updated, which is exactly the deployed estimator subset. A future value that
  omits gyro bits `8|16|32` must not be treated as a fresh rate sample.
- MAVLink continuous telemetry is lossy. Track packet sequence gaps, including
  8-bit wraparound, to distinguish UDP loss from policy/controller faults. The
  metric is valid here only while the runner receives one simulator channel;
  routing or merged redundant links can make sequence gaps misleading.
- MAVLink 2 receivers must reject packets with unknown incompatibility flags;
  generated `pymavlink` parsing supplies framing, CRC, truncation, and version
  handling. Do not replace it with an ad-hoc wire decoder.
- Standard systems can expose a one-shot extra message with
  `MAV_CMD_REQUEST_MESSAGE`, or set a stream interval with
  `MAV_CMD_SET_MESSAGE_INTERVAL`; both should return `COMMAND_ACK`. If we probe
  `ATTITUDE_TARGET`, `AUTOPILOT_VERSION`, or `SYSTEM_TIME`, do it before a
  passive shadow, log the ACK, and first prove v3385 support. Do not alter
  telemetry rates during an official attempt.
- `PING` can measure UDP round-trip latency independently of policy inference.
  It is diagnostic only: current recurrent cadence variance is already
  localized to the controller's outer loop, not a command/telemetry round trip.

Sources: [HIGHRES_IMU](https://mavlink.io/en/messages/common.html),
[packet-loss calculation](https://mavlink.io/en/guide/packet_loss.html), and
[serialization](https://mavlink.io/en/guide/serialization.html). See also
[requesting messages/rates](https://mavlink.io/en/mavgen_python/howto_requestmessages.html)
and [PING](https://mavlink.io/en/services/ping.html).

### Clocks and reset epochs

- A `TIMESYNC` request is `tc1=0, ts1=<requester ns>`; a response mirrors
  `ts1` and places responder time in `tc1`. The adapter's previously reversed
  request arguments were corrected on 2026-07-18.
- Reliable synchronization requires repeated exchanges plus filtering. Current
  official runs report no simulator TIMESYNC stream, so do not couple the
  recurrent policy clock to TIMESYNC or pretend camera and MAVLink clocks are
  synchronized.
- Use local monotonic elapsed time for the new fixed-rate recurrent state
  schedule. Continue using the simulator's custom race-status boot rollback as
  reset authority. Standard `SYSTEM_TIME.time_boot_ms` rollback or a HIL reset
  flag would be useful supplementary reset evidence only if v3385 starts
  publishing them.

Sources: [time synchronization](https://mavlink.io/en/services/timesync.html)
and [common message definitions](https://mavlink.io/en/messages/common.html).

### Security

MAVLink 2 signing protects authentication and replay on untrusted networks.
The current simulator uses a local unsigned UDP endpoint, so signing adds no lap
performance value and is out of scope. Reassess only if control traffic crosses
an untrusted network.

Source: [message signing](https://mavlink.io/en/guide/message_signing.html).

## Repository Audit Result

| Concern | Current status | Required action |
|---|---|---|
| Learned target IDs | Correct; component zero preserved | Keep |
| Heartbeat | Correct controller identity and periodic send | Keep |
| Reset command | Correct one-shot confirmation/params plus state proof | Keep; optionally log ACK if observed |
| Body-rate target mask | Correct value `128` | Keep |
| Setpoint time | Monotonic controller milliseconds | Keep |
| TIMESYNC request | Arguments were reversed | Corrected and unit-tested |
| TIMESYNC policy clock | Simulator support absent | Do not use |
| IMU validity flags | Simulator reports deployed six fields updated | Add a fail-closed diagnostic if live flags change |
| Packet sequence loss | Not yet exposed in run reports | Add as a diagnostic before attributing low cadence to UDP loss |
| `ATTITUDE_TARGET` echo | Not observed/parser not enabled | Probe opportunistically; never gate current flight on it |
| Extra message/rate requests | Not proven in v3385 | Probe only outside official attempts and require matching ACK |

## Consequence for the Current Candidate

The MAVLink audit does not explain the N229 prefix variance as a malformed
attitude command or reset. It strengthens the current N230 hypothesis: the wire
publisher should remain independent and continuous, while recurrent hidden
state must advance on a deterministic local monotonic clock. Shadow/parity must
still prove the implementation before another official command-`31000` run.

## 2026-07-19 Complete Rescan and Live Discovery Result

The English documentation source tree was rescanned from the current
`mavlink/mavlink-devguide` default branch: 97 files / about 2.2 MiB, including
every guide, language page, standard/dialect catalog, and microservice in the
site navigation. The earlier relevance selection remains correct, with these
additional conclusions:

- Numeric command `31000` is the standard enum slot
  `MAV_CMD_WAYPOINT_USER_1`, whose parameters and behavior are explicitly
  user-defined. MAVLink itself does **not** define it as reset. The simulator's
  reset meaning and seven-zero-parameter contract come only from the AI-GP
  example and measured v3385 behavior. Preserve the existing one-shot reset
  and postcondition proof; do not generalize from the enum name.
- `MAV_CMD_REQUEST_MESSAGE` is the current one-shot discovery command.
  Requesting `MESSAGE_INTERVAL` (`244`) with the desired message ID in request
  parameter 1 is the non-mutating way to ask whether a stream is disabled,
  unavailable, or active. Response-target value `1` addresses a response back
  to the requester. `MAV_CMD_SET_MESSAGE_INTERVAL` would mutate the stream and
  was deliberately not used.
- VADR-TS-002 issue 00.02 explicitly excludes GPS/absolute global position and
  lists only `HEARTBEAT`, `ATTITUDE`, `HIGHRES_IMU`, and `TIMESYNC` in the
  simulator-to-client table. `ODOMETRY` appeared only in superseded issue
  00.01 and was removed. Therefore `SIM_STATE`, HIL truth, GPS, odometry,
  local-position, and hidden track-geometry requests are outside the current
  legal observation contract and were not probed.

`scripts/probe_mavlink_discovery.py` now implements the compliant diagnostic.
Its default contract sends heartbeats, three non-mutating interval queries,
one-shot protocol metadata/`ATTITUDE` requests, one valid `TIMESYNC` request,
and one `PING`. It sends zero reset, arm, disarm, attitude/position setpoint,
persistent stream change, or position/simulator-truth request. Eighteen focused
tests pass. Script/test SHA-256 values are
`3271c80883fb9002a088404b6f32d5d62ef3d476ea50af49c8360a0abd375720` and
`d545937e9727c6522158c8160df4790c63eb43cbdaf075a3628c6144f1d8d4c5`.

The first probe ran on the terminal post-collision screen and is diagnostic
only. The UI was then recovered without relaunching responsive PID `24476`;
the active VQ1/R1 proof is
`logs/sitl/race_start_mavlink_discovery_active_recovery_001.json`, SHA-256
`20160c8b9ed01dbfa2fa3d5b05878445624dad5a8383abb9d6eb12b9436a34eb`,
with base/status `193/4`, official index `0`, and finish `-1`.

The authoritative active-state probe is
`logs/sitl/mavlink_discovery_probe_active_2026-07-19.json`, SHA-256
`f9f3bc4b7494d66ea51c474c0b3a9d48fe9cddd4637392354850742cfa8fbee6`.
Its two-second baseline received `223 HIGHRES_IMU`,
`183 ACTUATOR_OUTPUT_STATUS`, `20 HEARTBEAT`, and eight encapsulated race-status
messages. Across every request window v3385 emitted:

- no `COMMAND_ACK` for command `512`;
- no `MESSAGE_INTERVAL` response for `ATTITUDE`, `HIGHRES_IMU`, or `TIMESYNC`;
- no `AUTOPILOT_VERSION` or `PROTOCOL_VERSION`;
- no one-shot `ATTITUDE`;
- no `TIMESYNC` response; and
- no `PING` response.

This closes standard MAVLink discovery as a source of missing Gate-4 state on
v3385. Do not add persistent message-rate commands or repeat these requests in
official attempts. The legal live estimator remains camera + `HIGHRES_IMU`
gyro + actuator/race status + action history, on a local monotonic clock.
