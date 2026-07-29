# VQ2 command-free interface inventory — 2026-07-26

## Package and event identity

- User-supplied executable:
  `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3391\AIGP_3391\FlightSim.exe`
- Launcher SHA-256:
  `0d3217fa72e9fee847b2c154432476a687f21b79f0ab6b910728a6254b4dce32`
- Shipping executable SHA-256:
  `68dfd80d5c9057ec92785baad61194bf5d178ddde8d6df66a6add4da5d83332b`
- `FlightSim-WindowsNoEditor.pak`: `4574381151` bytes, SHA-256
  `5d424b4ee0de36053914461da56696cfff10c1ed9fab2c6bd883ace58e85883f`.
- The authenticated UI exposes VQ1, VQ2 Submission, and VQ2 Training. Only VQ2
  Training was selected. Submission was not selected or started.
- Active game process during inventory: PID `22692`, exact shipping executable
  under the package above. The window was resized for reliable UI navigation;
  this does not alter simulator physics or network behavior.

## Zero-command contract

The raw and decoded probes sent exactly zero:

- heartbeats;
- TIMESYNC replies;
- metadata/message requests;
- reset commands;
- arm or disarm commands;
- position, velocity, attitude, or body-rate setpoints.

VQ2 Training remained stationary at official `active_gate_index=0`, with
`race_finish_time_ns=-1`. This is an interface inventory, not a flight attempt.

## Raw transport evidence

Twenty seconds on UDP `14550` and `5600` produced:

| Stream | Observation |
|---|---:|
| MAVLink v2 datagrams | 3,215 |
| MAVLink v1 datagrams | 0 |
| MAVLink bytes | 127,479 |
| Camera chunks | 39,822 |
| Camera bytes | 54,910,556 |
| Valid `<IHHIIQ` camera headers | 39,822 |
| Camera header failures | 0 |

Artifact:
`C:\Users\anon\code\pufferlib-drone\logs\sitl\vq2_training_raw_stream_inventory_001.json`
(SHA-256
`73cb40eaec41aad175c4176e3ec0da9ccf765df1d23cc1e0abfee0a88f1c43ea`).

## Decoded MAVLink availability

The receive-only 12-second decoded inventory observed:

| Message/field | Count | Rate | Deployable status |
|---|---:|---:|---|
| HEARTBEAT | 119 | 9.917 Hz | Available |
| HIGHRES_IMU | 655 | 54.583 Hz | Finite accel and gyro available |
| ACTUATOR_OUTPUT_STATUS | 1,127 | 93.917 Hz | Available |
| ENCAPSULATED_DATA | 48 | 4.000 Hz | Race status available |
| ATTITUDE | 0 | 0 Hz | Unavailable |
| LOCAL_POSITION_NED | 0 | 0 Hz | Unavailable |
| ODOMETRY | 0 | 0 Hz | Unavailable |
| Track handshake/transfer | 0 | 0 Hz | Unavailable |
| TIMESYNC | 0 | 0 Hz | Not published in this window |

Race status is usable as the official progress authority. It does not reveal
the VQ2 gate count in this stationary window, so do not normalize progress by a
guessed denominator. Magnetometer, pressure, altitude, and temperature fields
inside HIGHRES_IMU are non-finite; accelerometer and gyroscope fields are
finite. No malformed message, unknown message, or telemetry dropout occurred.

The absence of a COLLISION message means no collision occurred during this
stationary probe; it does not prove the event-driven collision channel is
disabled.

## Camera evidence

- Header/framing remains TS-002 `<IHHIIQ>` on UDP `5600`.
- Decoded JPEG dimensions are `640x360`.
- The 12-second probe reconstructed 720 complete JPEG instances with zero
  malformed chunks, chunk loss, stalls, or reconstruction failures.
- Frame IDs and simulation timestamps arrived in duplicate pairs. This implies
  approximately 360 unique frames over 12 seconds, or nominal `30 Hz`.
  Consumers must deduplicate on `(frame_id, sim_time_ns)`.
- The UI displays `cam 20°`, but six independent command-free UDP frames now
  measure optical uptilt `1.920944634732011°` (range
  `1.7381973099618586°--2.1260258956233695°`). The aggregate calibration is
  `vq2_training_passive_camera_calibration_008.json`, SHA-256
  `bcc0372559f297382a177392d82e3cb2c96b810d550c47954596028450d75164`.
  The stationary IMU's approximately `17.803°` mount pitch is neither vehicle
  attitude nor camera tilt.
- Complete-frame deduplication on `(frame_id, sim_time_ns)` is implemented in
  `drone_camera_receiver.py` (SHA-256 `b2ea1df4...`). A six-second live passive
  probe emitted `180` unique frames, suppressed `6125` duplicate chunks, and
  recorded zero out-of-order frames, reconstruction failures, or chunk loss.

Decoded artifact:
`C:\Users\anon\code\pufferlib-drone\logs\sitl\vq2_training_passive_interface_inventory_001.json`
(SHA-256
`7e68308cec371c339476bf004aa5c53f7cee8a017efbd16a527470c3312d3279`).

First frame:
`C:\Users\anon\code\pufferlib-drone\logs\sitl\vq2_training_passive_interface_frame_001.jpg`
(SHA-256
`8a0e9ea39ec6d2f7cfa60119e8243ad5ebfc4d5087aae4a29906741a8a33c061`).

Probe implementation root/Windows SHA-256:
`fe6943a411c72155557f8be61ee497dbed51a5f87aa2b3266618b31bce49763d`.

## Decision

The VQ1 N283 controller is not a VQ2 candidate: its public-pose governor cannot
operate when vehicle pose and track transfer are absent. Preserve it only as
VQ1 evidence.

The active VQ2 mechanism is a recurrent PufferLib policy using camera, finite
HIGHRES_IMU accel/gyro, previous action, and official race progress. N294
passes exact Gate-1 `512/512`, perturbed `422/512`, zero crash/timeout, then
passes a `520/520` full-rate Windows shadow with maximum replay error
`4.33e-7` and zero lifecycle/control commands.

N295 subsequently passed official Gate 1 exactly once in VQ2 Training with the
frozen N294 checkpoint. Official Gate-1 race time is `3.228605508 s`; the
bounded index stop fired at `3.250 s`. It delivered `207` policy setpoints at
`63.995 Hz`, with zero in-run collision, invalid state, rate violation,
telemetry dropout, or drain-limit hit, and exact deployment hashes. A separate
receive-only follow-up proves disarm (`base_mode=65`, status `3`) while sending
zero packets. Its post-disarm terminal ground contact is outside the accepted
timed prefix. See the competitive ledger for exact report hashes. N295 cannot
be retried unchanged; further Training commands require a source-locked and
separately preregistered PufferLib Gate-2 continuation.

VQ2 Submission remains untouched and unauthorized during development.
