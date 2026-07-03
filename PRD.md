# PRD: Drone Gate Navigation Challenge (Source of Truth)

Last updated: 2026-07-03
Owner: `PufferLib/native-v4-drone-port`

## Source of Truth Policy

- This file is the active product and execution PRD for the race program.
- Legacy hover-era and historical experiment logs remain preserved in git history on the stabilized `3.0` checkpoint branch.
- If conflicts exist between this file and archived material, this file wins.

## Challenge Goal

Build a fully autonomous drone stack that passes all gates in strict order and minimizes valid completion time in the official virtual competition environment through the TS-002 observable interface: MAVLink v2 telemetry/control plus the forward camera stream.

## Program Goal And Completion Contract

Goal:

- Win, or place competitively enough to credibly contend for winning, by submitting a fully autonomous TS-002-compliant drone racing stack that completes the official DCL course in strict gate order with a competitive valid time.
- The primary submitted behavior should be backed by a PufferLib v4-trained policy using the competition-shaped observation/action contract.
- A conservative visual-servo controller remains an explicit fallback and validation baseline so the program can still produce a valid autonomous run if the learned policy is not yet reliable enough.

Completion definition:

- The project is not complete when a native checkpoint scores well. It is complete only when the stack runs through the official SITL/MAVLink + camera interface, without privileged native state, and produces deterministic evidence that it can complete the course.
- A submission candidate exists only after it:
  - maintains heartbeat at `>=2 Hz`;
  - sends accepted commands with an engineering target `>=50 Hz` and strict cap `<100 Hz`;
  - consumes TS-002-confirmed telemetry and camera/perception-derived gate state;
  - passes gates in strict order with no human interaction during the timed run;
  - writes deterministic JSON/CSV reports with completion time, ordered gate passes, invalid-run/crash status, command-rate metrics, telemetry dropouts, and vision-stream health;
  - demonstrates reliability first, then competitive speed.

Milestone ladder:

1. First-gate proof: pass the first gate in SITL using telemetry plus camera/perception-derived gate pose.
2. Policy proof: train `drone_race_competition` so a PufferLib v4 policy learns stable gate approach without privileged native target vectors.
3. Transfer proof: run a trained policy through the MAVLink/SITL adapter using the same normalized body velocity/yaw-rate contract.
4. Course proof: complete the official course in strict order with no human input.
5. Winning proof: tune completion time only after full-course reliability is stable enough that speed, not survival, is the limiting factor.

Work-continuity rule:

- Always work on the highest-priority unmet milestone above. Do not promote native-only checkpoints, R4/full-course curriculum, Q8/fake-quant, or speed tuning ahead of the SITL/camera/telemetry acceptance boundary unless the work is explicitly labeled as support-track diagnostics.

## Competition Strategy Reset

Decision as of 2026-05-21:

- The competition-winning path is a spec-compliant SITL controller first, not native `drone_race` score optimization in isolation.
- Native PufferLib remains valuable as a high-throughput training, regression, and teacher-signal backend, but native success is not promotable unless the behavior transfers through the MAVLink/SITL adapter and official telemetry/camera streams.
- The first competition-facing baseline should be conservative and inspectable: TS-002 camera ingestion, gate detection/pose estimation using known pinhole intrinsics and gate dimensions, telemetry-based stabilization, and bounded MAVLink setpoints at a verified command rate below `100 Hz`.
- Learned policies may augment perception, planning, or control after this baseline exists; they should not be the only path to a valid autonomous run.
- Speed optimization is explicitly secondary to reliable, ordered, spec-compliant gate traversal with correct command-rate, heartbeat, telemetry, and vision-health reporting.

## Current Completion Snapshot

Completed:

- `native-v4-drone-port` is now the GitHub default branch and active source of truth.
- Native v4 baseline exists for `drone` and `drone_race`, including C bindings, CPU smoke/eval paths, local regression checks, and Linux CUDA/NCCL validation on Vast.ai.
- `drone_race` uses representative quadrotor motor/RK4 dynamics instead of direct velocity/yaw-rate kinematics.
- Race and hover observations are aligned at `23` features, enabling hover-to-race native checkpoint warm starts.
- Staged native race curriculum exists for H1, R1, R3, and R4, with promotion checks that stop on failed thresholds.
- Reproducible checkpoint eval artifacts are supported via `scripts/eval_drone_race_checkpoint.py` with paired JSON/CSV outputs.
- Linux GPU support track ran on Vast RTX 4090: native precheck/validation passed, short `drone_race_competition` train completed, and checkpoint eval artifacts were produced (`logs/drone_race_competition_eval.{json,csv}`).
- Useful native checkpoints exist through H1 and R1; R3 has a reproducible partial-success checkpoint but is not promotion-ready.
- Q8/mixed-precision PufferNet inference scaffolding, export, latency/drift benchmarking, and closed-loop hover comparison harnesses exist.
- MAVLink/SITL adapter scaffold, local dry-run path, TS-002 command-rate validation, and testable inbound telemetry parser/reporting scaffold exist.
- First-pass competition-shaped policy contract exists for native training and SITL wiring: `config/drone_race_competition.ini` trains the compiled `drone_race` backend with TS-002-shaped observations and body velocity/yaw-rate policy actions, while `scripts/drone_policy_contract.py` defines the matching Python decode path to MAVLink local-NED setpoints.
- First-pass TS-002 visual-servo geometry helper exists: detected square-gate corners can be converted into relative body/NED pose and conservative local-NED velocity/yaw commands.
- Local competition smoke entrypoint now exists (`scripts/drone_sitl_competition_smoke.py`): it wires MAVLink telemetry, TS-002 camera ingest, first-pass square-gate detection, visual-servo or policy-contract command emission, and deterministic JSON/CSV reporting in one run loop.
- First-pass square-gate detector now exists (`scripts/drone_gate_detector.py`) and is integrated into the camera/perception boundary used by the smoke entrypoint.
- Windows-local official simulator runtime is now validated on the RTX 3070 workstation with a lightweight `.venv-win` Python stack (`pymavlink`, `numpy`, `opencv-python` only for SITL). The observed simulator topology is MAVLink v2 telemetry from `127.0.0.1:14560` to client port `14550`, TS-002 camera packets on client port `5600`, and simulator-owned sockets on `14560` and `5601`.
- On 2026-06-04, Windows-local official simulator smoke runs validated telemetry+camera ingestion and controller-side visual gate-pass heuristics: `logs/sitl/competition_smoke_gate1_windows_local_drain.json` reports nonzero telemetry/camera, `command_rate_violations=0`, `telemetry_dropouts=0`, `completed_frames=5183`, and `detector_detections=2282`.
- On 2026-07-02, official simulator `v1.0.3379` replaced the unsupported `v1.0.3364` runtime on this workstation. The supported executable is `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3379\AIGP_3379\FlightSim\Binaries\Win64\DCGame-Win64-Shipping.exe`; the old `v1.0.3364` app now shows a service/version error and is not usable for promotion evidence.
- Official simulator `v1.0.3379` still uses the Windows-local topology: simulator-owned sockets on `14560` and `5601`, client MAVLink on `14550`, and client camera on `5600`. The current reliable operator flow is clean launch, submit saved login, select `AI-GP Virtual Qualifier R1`, click `RACE`, then start the controller locally from `.venv-win`.
- The `v1.0.3379` bundled `PyAIPilotExample-v2` and live traffic confirm that `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, and track geometry should not be assumed for current runs; live official validation receives `HEARTBEAT`, `HIGHRES_IMU`, `ACTUATOR_OUTPUT_STATUS`, and encapsulated race status. Yaw/local position must therefore remain optional inputs.
- Current command-frame evidence favors `SET_POSITION_TARGET_LOCAL_NED` with the original yaw setpoint behavior as the conservative baseline. A `MAV_FRAME_BODY_NED` trial (`logs/sitl/official_gate1_validation_summary_3379_body_frame_official_progress_001.json`) produced valid streams and commands but no controller-side pass and flew off course. The default helper path now keeps frame selection explicit while defaulting to `local_ned`.
- The current best `v1.0.3379` official run is `logs/sitl/official_gate1_validation_summary_3379_local_ned_restored_yaw_002.json`: probe passed with nonzero MAVLink and TS-002 camera packets, heartbeat was `2 Hz`, command rate was `50 Hz` with `0` violations, telemetry dropouts were `0`, the controller-side vision tracker emitted `ordered_gate_passes=1` at `17.687s`, but official race status stayed at `active_gate_index=0` and `last_gate_race_time=-1`.
- The SITL smoke report now emits `approach_diagnostics` with closest range, confidence, body-vector estimate, visual-servo command, and bounded pose/command samples. The first diagnostic run (`logs/sitl/official_gate1_validation_summary_3379_approach_diag_001.json`) showed the controller never got inside the vision pass threshold: closest estimated camera range was `2.611632m` with low confidence `0.111434`, and official race status remained gate `0`.
- Early command-semantics probes on `v1.0.3379` did not prove first-gate advancement. `--visual-servo-desired-standoff-m 0.0` still missed (`logs/sitl/official_gate1_validation_summary_3379_standoff0_001.json`, closest `2.681855m`). Constant forward local-NED policy mode also missed with both yaw-active and official-example yaw-ignored masks; the yaw-ignored run recorded collisions and actuator outputs near `0.05` (`logs/sitl/official_gate1_validation_summary_3379_constant_forward_ignore_yaw_001.json`).
- The SITL adapter now supports the simulator example's `SET_ATTITUDE_TARGET` body-rate command shape: attitude ignored, body roll/pitch/yaw rates active, and normalized thrust active. `scripts/drone_sitl_competition_smoke.py`, `scripts/run_official_gate1_validation.py`, and `scripts/watch_official_gate1_validation.py` expose this as `--control-mode attitude-rates`, plus a camera-steered `--control-mode visual-servo-attitude` that maps detected gate pose to body pitch/yaw rates and thrust. `scripts/run_sitl_control_calibration.py` runs a fixed velocity/attitude-rate/steered-attitude matrix and emits per-case JSON/CSV plus a summary ranking official gate progress, closest range, confidence, collisions, and command-rate compliance.
- First live `SET_ATTITUDE_TARGET` diagnostics on `v1.0.3379` prove the simulator obeys body-rate/thrust commands. `logs/sitl/control_calibration_live_attitude_t060_probe_01_attitude_rates_pitch_negative_t060.json` reached `closest_range_m=1.251098`, `highest_confidence=0.970796`, zero collisions, and no command-rate violations in an 8s run, but official `active_gate_index` stayed `0`. A longer non-reset run (`logs/sitl/official_gate1_validation_attitude_t060_long.json`) produced multiple controller-side gate-pass events and real actuator outputs near `0.59/0.61`, but official race status still did not advance. The follow-up steered-attitude artifact (`logs/sitl/official_gate1_validation_visual_servo_attitude_001.json`) started from the disturbed post-diagnostic state and saw no gate detections, so it is not promotion evidence; rerun it only after a clean manual race reset.
- Fresh-race arming now requires receiving an inbound simulator heartbeat before sending `MAV_CMD_COMPONENT_ARM_DISARM`. `scripts/drone_sitl_competition_smoke.py` now waits for a pre-arm heartbeat by default, sends configurable arm attempts, and records `arm_on_start`, `arm_commands_sent`, and `prearm_heartbeats_seen` in `control_inputs`. The first armed visual-servo-attitude run (`logs/sitl/official_gate1_validation_visual_servo_attitude_prearm_001.json`) confirmed the fix: `prearm_heartbeats_seen=1`, `arm_commands_sent=5`, `base_mode=193`, `system_status=4`, and actuator outputs around `0.71-0.79`.
- The armed visual-servo-attitude run produced controller-side pass events but still missed official progress: `ordered_gate_passes=2`, first controller-side pass at `6.485s`, closest range `1.598883m`, zero command-rate violations, zero telemetry dropouts, but official `active_gate_index=0` and `last_gate_race_time=-1`. The controller is now applying thrust; the remaining blocker is approach geometry/control, not transport or arming.
- The yaw-search `visual-servo-attitude` path can now reacquire an off-screen first gate. `logs/sitl/official_gate1_watch_yaw_search_after_relaunch_latest.json` waited through menu/no-camera states, then ran after manual R1/RACE start with nonzero streams, `prearm_heartbeats_seen=1`, `base_mode=193`, `command_rate_violations=0`, `telemetry_dropouts=0`, `detector_detections=2772`, and `ordered_gate_passes=6`. It still failed official promotion (`official_active_gate_index=0`, `last_gate_race_time=-1`) and recorded one collision, so the result proves search/reacquisition and actuation, not gate completion.
- The detector is locking onto the real first gate in the current v3379 scene. `logs/sitl/after_yaw_search_after_relaunch_snapshot_latest/first_detection_annotated.jpg` shows the annotated square on the gate, with the vehicle low and obstacle-bound. The remaining controller problem is to climb/yaw/center before pitching forward through the gate plane.
- `scripts/drone_sitl_competition_smoke.py` now records the actual attitude target used for each approach diagnostic sample (`actual_command`) and supports opt-in `visual-servo-attitude` yaw-search plus center-gated forward motion. Use `--attitude-servo-search-yaw-rate-rad-s`, `--attitude-servo-forward-yaw-tolerance-rad`, `--attitude-servo-forward-z-tolerance-m`, and `--attitude-servo-uncentered-forward-scale 0.0` to rotate/climb until the gate is centered before applying forward pitch.
- `scripts/run_official_gate1_validation.py` now performs a short pre-smoke race-start check. When the simulator is streaming camera but `race_start_boot_time_ms=-1`, it returns `blocked_race_not_started` without arming or sending controller commands. `logs/sitl/official_gate1_validation_race_start_check_latest.json` proves the fast path: MAVLink/camera were nonzero, `race_started=false`, `smoke=null`, and elapsed time was `2.260205s`.
- Windows race restart automation now has a safe wrapper: `scripts/start_windows_aigp_race.ps1` relaunches v3379, optionally clicks the saved login/R1/RACE screen coordinates, and then calls `scripts/wait_official_race_start.py`. The helper only reports success when `race_start_boot_time_ms>=0`; if the coordinate click path misses, it fails before controller commands are sent.
- The restart helper is live-validated. `logs/sitl/race_start_helper_full_relaunch_retry_test.json` shows a full process relaunch plus UI retry loop reaching `race_started=true`, `base_mode=193`, `system_status=4`, and `race_start_boot_time_ms=45563`. A failed coordinate pass correctly produced nonzero exit with `race_started=false`, so the safety gate works.
- The live detector now has a color-required aperture path for the v3379 red gate. `scripts/drone_gate_detector.py` first finds a red gate contour, then returns a substantial inner opening contour when visible instead of treating the red outer frame/panel as the scoring target. This removed the earlier gray-ceiling/grid false positives and eliminated the misleading local pass events caused by clipping a red gate edge.
- Aperture-detector strict runs prove the remaining first-gate issue is controller geometry, not connectivity or arming. `logs/sitl/competition_smoke_aperture_hybrid_latest.json` kept nonzero MAVLink/camera streams, `command_rate_violations=0`, `telemetry_dropouts=0`, `ordered_gate_passes=0`, and official `active_gate_index=0`; the closest aperture estimate was `3.595506m` with the aperture high-left in `logs/sitl/debug_gate1_aperture_hybrid_latest/closest_detection_annotated.jpg`.
- The best close-range aperture run so far is `logs/sitl/competition_smoke_aperture_final06125_latest.json`: final approach latched at `3.922s` with trigger pose `range_camera_m=3.490909`, `image_center_px=[89.5,123.0]`, command `body_pitch_rate=-0.3`, `body_yaw_rate=-0.3`, `thrust=0.6`, and no command-rate violations, no telemetry dropouts, no collisions, but official race status still stayed at `active_gate_index=0`, `last_gate_race_time=-1`.
- The first loosened center-gated crawl attempt (`logs/sitl/official_gate1_validation_center_gated_crawl_latest.json`) still failed official progress with `official_active_gate_index=0`, but improved evidence: nonzero streams, `command_rate_violations=0`, `telemetry_dropouts=0`, zero collisions, `detector_detections=4820`, and closest range `1.580692m`. The miss occurred with high closest yaw error (`0.467369 rad`) while crawl was enabled, so the next attempt should keep the wider vertical tolerance but remove forward crawl unless yaw is centered.
- The armed local-NED constant-forward retest (`logs/sitl/official_gate1_watch_velocity_forward_armed.json`) is not a promotion result. It did start/move the race view, but the vehicle did not approach the first gate (`closest_range_m=11.958532`) and official gate progress remained `0`; treat local-NED constant forward as a poor primitive for gate traversal unless future axis sweeps show a better sign/frame mapping.
- Live TS-002 camera ingestion required draining multiple UDP chunks per smoke-loop iteration; `scripts/drone_camera_receiver.py`, `scripts/drone_sitl_competition_smoke.py`, and `scripts/run_official_gate1_validation.py` now support bounded camera packet draining via `--camera-max-packets-per-loop` (default `512`).
- The simulator-bundled `PyAIPilotExample` confirmed the MAVLink/camera endpoints, camera header, custom reset command `31000`, and race-status/track-info encapsulated data formats. The SITL adapter now parses official race status, local position, odometry, actuator output, and collisions.
- Manual simulator reset is currently the reliable start-state path. `logs/sitl/reset_snapshot_manual_001/snapshot_summary.json` captured a healthy reset state with normal local position near the origin, non-black camera frames, `race_status.active_gate_index=0`, and `188` detector hits in `5s`; the first detected frame shows the first gate centered in view.
- The experimental MAVLink reset command `31000` can leave the simulator in an invalid/blank-camera state in the current session. `logs/sitl/reset_snapshot_auto_001/snapshot_summary.json` recorded black frames, zero detections, and an implausible local NED `z` near `-9342m`; do not use auto-reset as promotion evidence until this is understood.

Not complete / blocking:

- Official simulator traffic validation is proven, the manually reset first-gate camera view is healthy, and the detector sees the first gate. Official first-gate race progress is not yet proven. After adding simulator race-status parsing, a strict `--require-official-race-progress` run on 2026-06-04 failed with `active_gate_index=0`, `last_gate_race_time=-1`, and blockers `insufficient_gate_passes:0<1` plus `insufficient_official_gate_progress:0<1`.
- As of 2026-07-02 on simulator `v1.0.3379`, the blocker is narrower: connectivity, camera, heartbeat, arm state, actuator output, and command-rate compliance are proven, and the vision-side pass heuristic can trigger, but official race status still does not advance beyond gate `0`. Treat controller-side `ordered_gate_passes` as diagnostic only until `official_active_gate_index>=1`.
- As of 2026-07-03, aperture-based detection is stable and no longer reports clipping-related false passes, but `active_gate_index` remains `0` despite repeated vision-side gate events. The approach geometry blocker appears to be lateral correction timing (for example, low/right aperture alignment in the closest annotated frame). We added image-space roll steering (`--attitude-servo-k-image-roll`) with defaults that use full-frame normalization for tiny detections, and updated controller harnesses so the new arg is passed through safely in calibration/watch/race validation. Next action: run a clean manual-reset sweep in this order: `0.0`, `-0.05`, `0.05`, `-0.12`, `0.12`, then tune sign and magnitude until first-gate `official_active_gate_index` advances.
- The current winning hypothesis should shift toward a conservative visual-servo/SITL baseline, with native RL used as support rather than the submission path by itself.
- R3 three-gate stability is still below promotion thresholds; latest best 4k deterministic eval reached `success_rate=0.8775`, `crash=0.1225`, `gates_passed=2.7847`, still above the crash threshold.
- R4/full-course race promotion is blocked until R3 reliability improves.
- Existing promoted/partial native checkpoints were trained through the legacy privileged native observation/action interface; the competition-shaped contract is implemented but not yet trained or validated in the simulator.
- MAVLink telemetry parsing for `HEARTBEAT`, `ATTITUDE`, and `HIGHRES_IMU` is validated against the official Windows simulator traffic on port `14550`; `TIMESYNC` was not observed in the 2026-06-04 runs and should remain optional/diagnostic until confirmed.
- Local SITL evaluation with fixed course/start conditions is not implemented.
- Historical 2026-05-22 smoke run observed zero simulator telemetry/camera traffic; this is superseded by replay-backed first-gate passing artifacts on 2026-05-30 and Windows-local stream/controller validation on 2026-06-04.
- Local strict acceptance passes with TS-002 mock/replay streams and official Windows simulator traffic for stream/controller health. Official first-gate proof now requires simulator race-status gate advancement, not only controller-side vision pass heuristics.
- Deterministic UDP capture/replay tooling now exists for SITL regression without official traffic (`scripts/sitl_udp_capture.py`, `scripts/sitl_udp_replay.py`), and policy-mode smoke wiring can be exercised with a competition-shaped callable (`scripts/policy_callable_gate_pid.py`).
- Local offline regression now includes replay-driven policy smoke acceptance from captured traffic artifacts, reducing dependence on live simulator availability for day-to-day controller/perception/reporting validation.
- First-gate proof is now reproducible on deterministic replay traffic using telemetry+camera only, with canonical artifacts in `logs/sitl/competition_smoke_gate1.{json,csv}` (`ordered_gate_passes=1`, no crashes, no command-rate violations).
- Deterministic replay reliability now meets promotion thresholds with frozen gate-pass defaults: `N=30` normal and `N=30` degraded Profile A both report `30/30` valid runs for visual-servo and checkpoint policy modes.
- TS-002 vision ingestion now includes UDP runtime integration, high-rate chunk draining, frame reassembly, first-pass detector hookup, and time-sync error tracking in local smoke runs. Live simulator stream validation passes; official gate advancement remains pending.
- Native gate geometry is still too abstract for TS-002 clearance: current circular gate radii and point-mass crossing checks do not model the `1500x1500x260 mm` inner opening, `2700x2700x260 mm` frame, or `280x280x160 mm` chassis.
- Q8/fake-quant training is not promoted; closed-loop race parity and action drift remain unresolved.

Blocking work breakdown:

- Spec-compliant SITL baseline:
  - Current state: Windows-local official simulator traffic is reachable on supported simulator `v1.0.3379`, official race-status messages are parsed, and strict runs can prove nonzero MAVLink/camera with compliant heartbeat and command rates. The live topology is MAVLink telemetry on client port `14550`, TS-002 camera on `5600`, and outbound setpoints through `udpin:0.0.0.0:14550`.
  - Next action: use clean simulator launch/manual R1/RACE start, require simulator race-status advancement with `--require-official-race-progress`, and tune the center-gated `visual-servo-attitude` controller until `active_gate_index` advances past `0`. Keep MAVLink auto-reset disabled for promotion runs until the black-frame/high-altitude reset behavior is explained.
  - Acceptance signal: first-gate acceptance is `acceptance_passed=true` from a reset official simulator state with nonzero race-status messages, `official_active_gate_index>=1`, command-rate, telemetry-dropout, vision-health, crash, and completion-time fields.
- Visual-servo competition baseline:
  - Current state: the first-pass square-gate detector feeds the pose helper in the live SITL loop, and `v1.0.3379` local-NED/yaw validation produced a controller-side pass at `17.687s` without official race advancement. Body-NED is not the current promotion default because the latest body-frame trial produced no controller-side pass and a clear off-course run.
  - Next action: use official race status as the validation judge; continue from the aperture-detector runs and add image-space lateral control. First calibrate `SET_ATTITUDE_TARGET` roll-rate sign/effect from a clean race start, then add a bounded roll or earlier yaw correction that moves the aperture from high-left toward camera center before the final forward pitch. Promote only a run where official `active_gate_index` advances past `0`; do not treat local vision pass events as promotion evidence.
  - Acceptance signal: telemetry+camera-only baseline advances official `active_gate_index` at least once in a fixed SITL course without native privileged gate vectors, then graduates to ordered two-gate validation.
- R3 native race reliability:
  - Current state: best non-promoted R3 candidate is `checkpoints/drone_race/1777433724949/0000000026279936.bin`, but larger eval still fails the crash threshold.
  - Diagnostic result: crashes are low-altitude dives, not high-altitude or horizontal out-of-bounds (`crash_low` accounts for essentially all crashes).
  - Terminal detail: low crashes cross the floor just below `z=-1.0` but with high downward velocity (`avg_low_crash_vz` around `-3.5` to `-4.5`), so simply lowering `crash_height` does not solve the failure.
  - Route-position diagnostic result: low crashes are not isolated to one route segment. On the 4k best-candidate eval, low-crash distribution was about `34.5%` before gate 1, `19.6%` while targeting gate 2, and `45.9%` while targeting gate 3.
  - Recent shaping result: strong/mild altitude-floor penalties, control-penalty brackets (`w_ctrl=0.014/0.016`), and lower time pressure (`w_time=0.1`) produced 1k false positives or collapsed later; none passed a 4k deterministic eval.
  - Latest curriculum result: raised-gate R3 continuation with `gate_altitude=1.6` produced the best non-promoted 4k result so far, but still missed crash threshold (`crash=0.1225`).
  - Latest diagnostic step: native logs now include pre-impact time-to-floor, stopping-margin, and max-up-accel metrics so low crashes can be classified as vertical-energy/stopping-distance failures instead of altitude-only failures.
  - Latest local attempt: restored the raised-gate lead checkpoint from `vast_backup_20260428`, but macOS cannot run the required default-native PufferL eval/training backend because `nvcc`/CUDA is absent; the CPU `_C` rebuild is diagnostic-only and does not expose `create_pufferl` for `scripts/eval_drone_race_checkpoint.py`.
  - Next action: when Vast/Linux CUDA is reachable, rebuild `drone_race` with default native precision and re-evaluate the raised-gate lead with the new diagnostics; if low crashes show negative pre-impact stopping margin, try a minimal vertical stop-margin reward or checkpoint filter before more scalar altitude tuning.
  - Acceptance signal: deterministic JSON/CSV eval reaches the R3 promotion threshold with `success_rate >= 0.85` and `crash <= 0.10`.
- R4/full-course promotion:
  - Current state: blocked; direct 4-gate and weak 3-gate continuations have regressed.
  - Next action: do not run R4 as a promotion attempt until either R3 passes deterministic eval under native governance or the SITL visual-servo baseline demonstrates ordered multi-gate behavior. Use R4 only for diagnostic probes if explicitly labeled as non-promotable.
  - Acceptance signal: R4 resumes only from a promoted R3 checkpoint or from a verified SITL baseline, with checkpoint/controller lineage and eval artifacts recorded.
- Privileged native-state dependency:
  - Current state: native `drone_race` has an opt-in `interface_mode = 1` path with a 23-float TS-002-shaped observation contract and four normalized body velocity/yaw-rate actions; short Linux GPU smoke checkpoint exists and replay reliability is now non-inferior to baseline at `N=30`.
  - Next action: continue Linux GPU/PufferLib checkpoint iteration while preserving replay reliability, but do not make the Unreal simulator the main training loop unless a fast, repeatable headless stepping/reset API is available. Use the official simulator for validation, targeted data capture, and transfer checks.
  - Acceptance signal: policy-mode official-traffic SITL runs pass first-gate acceptance with telemetry+camera only and deterministic JSON/CSV evidence.
- MAVLink telemetry and controller contract:
  - Current state: heartbeat, setpoint, and inbound telemetry parsing/reporting are validated against official simulator `HEARTBEAT`, `HIGHRES_IMU`, `ACTUATOR_OUTPUT_STATUS`, and encapsulated race-status traffic. The selected first-pass model output is normalized body forward/right/down velocity plus yaw-rate, decoded to MAVLink velocity/yaw setpoints; the live helper defaults to `local_ned` because that matches the simulator example and current evidence.
  - Next action: keep `TIMESYNC`, `ATTITUDE`, `LOCAL_POSITION_NED`, and `ODOMETRY` optional until observed/confirmed in the current simulator, then connect `scripts/drone_policy_contract.py` to the live SITL adapter for policy-mode official runs. Do not assume simulator navigation-reference or `ODOMETRY` availability until organizers confirm it, because TS-002 telemetry bullets omit the TS-001 navigation-reference/ODOMETRY row. Treat yaw-ignore command masks as diagnostic-only until they can move without collision/idle behavior.
  - Acceptance signal: SITL bridge can log confirmed TS-002 telemetry, emit accepted commands with an engineering target of `>=50 Hz` while strictly enforcing `<100 Hz`, and produce a deterministic run report with command-rate/dropout metrics.
- Local SITL eval runner:
  - Current state: `scripts/drone_sitl_competition_smoke.py` can connect to the official Windows simulator endpoint, run visual-servo or policy-contract command loops, and emit deterministic JSON/CSV artifacts with telemetry/camera health metrics.
  - Next action: document or automate fixed course/start-condition reset control, then add ordered multi-gate/completion extraction so it can gate promotion directly.
  - Acceptance signal: one-command SITL smoke/eval produces success, ordered gate passes, completion time, crash/invalid-run status, command rates, and telemetry dropout fields.
- Vision ingestion:
  - Current state: TS-002 defines a `30 Hz`, `640x360` JPEG-over-UDP stream on default/client port `5600`, chunked as a 24-byte little-endian metadata header plus JPEG payload. The live simulator sends high-rate chunk traffic; the receiver now supports bounded packet draining so smoke runs can reconstruct current frames instead of falling behind stale chunks.
  - Next action: keep packet-drain defaults tuned for live traffic, sync `sim_time_ns` with MAVLink `TIMESYNC` if/when the simulator emits it, and add regression coverage for high-rate chunk draining.
  - Acceptance signal: camera adapter publishes reconstructed frames with `sim_time_ns`, applies TS-002 pinhole intrinsics/extrinsics, and supports separate telemetry-only vs telemetry+vision eval tracks.
- Native geometry and gate fidelity:
  - Current state: native course logic is useful for curriculum training, but it has not been audited against TS-002 chassis, gate frame, inner opening, obstacle, and boundary dimensions.
  - Next action: compare native gate crossing/collision helpers, obstacle definitions, and course geometry against the TS-002 drone chassis `280x280x160 mm`, gate outer `2700x2700x260 mm`, and gate inner `1500x1500x260 mm` dimensions. Replace point/circular gate success checks with a spec-faithful clearance model before treating native success as transferable.
  - Acceptance signal: native and SITL geometry tests cover gate opening, frame depth, drone chassis clearance, obstacle/boundary collisions, and ordered crossing validity.
- Official simulator runtime topology:
  - Current state: native training and Vast.ai validation remain Linux-oriented, while the official DCL simulator runs on Windows 11. The validated local topology is Windows simulator plus Windows-local Python `.venv-win` controller on the RTX 3070 workstation; earlier WSL relay attempts were blocked by firewall/VPN routing and are not the preferred path.
  - Next action: document the Windows-local runbook (`.venv-win`, MAVLink client port `14550`, camera port `5600`, simulator reset requirement) and keep Linux/Vast only for training/checkpoint generation.
  - Acceptance signal: documented Windows-local topology can run the controller against the official simulator without assuming a Linux-hosted DCL simulator.
- Q8/fake-quant promotion:
  - Current state: Q8 export/runtime and benchmarks exist, but action drift and closed-loop race parity are not promotion-ready.
  - Next action: defer fake-quant RL until FP32 closed-loop behavior is reliable through the SITL/camera/telemetry path; then add QAT or post-training calibration against closed-loop hover/race metrics.
  - Acceptance signal: Q8 policy matches FP32 closed-loop SITL success/crash metrics within agreed tolerance and meets worst-case latency requirements.

## Competition-Aligned Requirements

- Gate recognition from provided sensor/visual streams.
- Precision drone control (thrust/orientation/rates) with reliability-first safety behavior.
- Efficient planning/navigation under realistic vehicle limits.
- No manual control and no hardware advantages.

## Official Virtual Qualifier Spec Alignment

Reference: `docs/260508_Technical_Spec_0002.pdf` (`VADR-TS-002`, issue `00.02`, cover date `2026-05-08`; revision history `2026-05-04` note: `camera`).

Decision:

- Native PufferLib v4 remains the high-throughput training and regression backend, not the primary definition of competition readiness.
- Native `_C` execution is the required path for drone training and evaluation; `--slowly`/PyTorch backend runs are not acceptance criteria.
- Final competition integration must communicate through MAVLink v2 over UDP via a MAVSDK-compatible SITL bridge.
- Training interfaces must converge toward the official observable interface instead of relying permanently on privileged absolute state.
- Native `drone_race` success is a training milestone only; it is not a qualifier-equivalent result until the policy/controller runs through the SITL/MAVLink interface.
- The first qualifier-facing implementation should be a conservative telemetry+camera visual-servo controller. RL/native checkpoints should be used to improve or replace pieces only after the observable pipeline is running.
- Linux/Vast remains valid for training throughput, but official simulator integration must account for the TS-002 Windows 11 DCL simulator runtime.

Spec constraints:

- Physics target rate: `120 Hz`.
- Qualifier-facing command rate: engineering target `>=50 Hz`, with strict TS-002 cap `<100 Hz`.
- Minimum heartbeat rate: `2 Hz`.
- Round 1 maximum run duration: `8 minutes` (`480s`).
- Simulator uses a local Cartesian internal frame; no GPS simulation or absolute global position is exposed.
- Supported inbound control messages include `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET`.
- Supported simulator-to-client MAVLink messages include `HEARTBEAT`, `ATTITUDE`, `HIGHRES_IMU`, and `TIMESYNC`.
- Relevant telemetry bullets include vehicle attitude, orientation, linear velocities, and system status flags. TS-002 no longer lists the TS-001 simulator navigation-reference/`ODOMETRY` row, so any dependency on nav-reference/ODOMETRY data needs organizer clarification.
- Vision is a forward-facing first-person camera stream over UDP: default port `5600`, `30 Hz`, `640x360`, little-endian packet metadata, 24-byte header, and JPEG payload chunks keyed by `frame_id`, `chunk_id`, `total_chunks`, `jpeg_size`, `payload_size`, and `sim_time_ns`.
- Camera model: pinhole, no lens distortion, `fx=fy=320`, `cx=320`, `cy=180`, `VFoV=90 deg`, camera/body same origin, camera tilted `20 deg` upward relative to body.
- Drone chassis: `280x280x160 mm`.
- Gate geometry: outer frame `2700x2700x260 mm`; inner square opening `1500x1500x260 mm`.
- Official DCL simulator runtime: Windows 11, standard PC with a decent GPU around `8GB` VRAM; Linux OS is currently not supported for the simulator.
- Course geometry, physics parameters, and environmental conditions are deterministic.
- Human interaction during a submitted timed run remains grounds for immediate disqualification.

Current alignment assessment:

- Status: useful research backbone, not yet on a competition-winning implementation path until the SITL/camera baseline exists.
- Aligned:
  - Native v4 `_C` is the required high-throughput training/eval backend.
  - `drone_race` now uses representative quadrotor motor/RK4 dynamics instead of direct velocity/yaw-rate kinematics.
  - Native race timing is set around the official `120 Hz` physics target and `480s` maximum run.
  - MAVLink v2/UDP bridge scaffolding exists for heartbeat and setpoint commands, but needs the TS-002 `<100 Hz` command cap enforced.
  - The program is reliability-first: speed optimization is blocked until valid-run stability improves.
- Not yet compliant:
  - Official stream/controller health is validated, but simulator race-status first-gate advancement is not yet proven.
  - No camera-based gate detector or integrated visual-servo baseline is implemented; corner-based pose/command geometry exists but is not live.
  - Current policies still train on native privileged state vectors; competition-facing policies must converge to telemetry plus forward-camera observations.
  - Telemetry parsing for attitude/orientation, IMU, status, and time sync is scaffolded but not validated against official simulator traffic.
  - Navigation-reference/`ODOMETRY` availability is ambiguous under TS-002 and must not be assumed until clarified.
  - The policy/controller output contract to official MAVLink setpoints is not finalized.
  - Vision ingestion now has a tested TS-002 header parser/reassembler scaffold, but UDP runtime smoke testing, `sim_time_ns`/MAVLink time sync, and policy/perception integration are not complete.
  - Native geometry, obstacle, and gate-fidelity assumptions have not been audited against TS-002 chassis/gate dimensions.
  - Official simulator runtime topology is resolved for the local RTX 3070 workstation: run the simulator and lightweight Python controller on Windows, while retaining Linux/Vast for training.
  - Current native race policy is not yet stable enough for full-course promotion.

Qualifier acceptance boundary:

- A run is qualifier-representative only if it:
  - communicates with the simulator over MAVLink v2/UDP through the SITL bridge;
  - maintains heartbeat at `>=2 Hz`;
  - emits accepted setpoint/control messages with an engineering target of `>=50 Hz` and a hard cap `<100 Hz`;
  - consumes official telemetry/observable streams rather than native privileged state;
  - consumes the TS-002 forward-camera stream through a JPEG reassembler when using vision;
  - respects the `480s` Round 1 maximum duration;
  - has no human interaction during the submitted timed run;
  - records deterministic report artifacts with success, ordered gate passes, completion time, crash/invalid-run status, command rates, telemetry dropouts, and vision-stream health.
- Native `_C` race metrics remain necessary for training throughput and regression, but are not sufficient for competition acceptance.

## Program-Level Success Metrics

- `valid_run_rate`: percentage of runs that complete all gates in order.
- `median_valid_time`: median completion time over valid runs.
- `ordered_gate_pass_rate`: percentage of accepted in-sequence gate crossings.
- `crash_rate`: fraction of runs that terminate via crash/unsafe state.
- Future SITL/vision metrics: camera FPS, frame jitter/stalls, JPEG reconstruction failures, chunk loss, `sim_time_ns`/MAVLink time-sync error, and command-rate enforcement violations above the TS-002 `<100 Hz` cap.

## Architecture Decision (Active)

Use a spec-first competition stack with native v4 as the training/regression backend:

1. SITL bridge: maintain heartbeat, parse confirmed TS-002 MAVLink telemetry, enforce command rate `<100 Hz`, and emit deterministic run reports.
2. Camera/perception interface: reconstruct TS-002 JPEG frames, synchronize `sim_time_ns`, apply camera intrinsics/extrinsics, and expose frames to perception.
3. Baseline perception/planning: detect the visually distinctive square gates, estimate relative gate pose/scale from image geometry, and sequence gates conservatively.
4. Controller interface: emit local setpoints or attitude targets compatible with MAVLink command messages, with bounded velocity/attitude/thrust behavior before speed tuning.
5. Native training envs: keep `ocean/drone` for hover/control prior and `ocean/drone_race` for race progression, teacher signals, and regression under increasingly spec-faithful observations/geometry.
6. Deterministic evaluation: keep fixed-seed native smoke/eval harnesses for regression, but require SITL/camera/telemetry reports for qualifier readiness.
7. Edge inference: keep policies small enough for deterministic inference and add quantized PufferNet runtimes only after FP32 closed-loop SITL behavior is reliable.

## Edge Inference and Quantization Track

Decision:

- Start with quantized, mixed-precision `PufferNet` inference for the existing `DefaultEncoder + MinGRU + DefaultDecoder` policy.
- Do not introduce Transformer policies until native state-vector control is reliable and camera/vision requirements justify the added complexity.
- Treat quantization as a closed-loop control problem, not just action-MSE compression.

Initial target:

- Observations are normalized/clamped in native envs (`23` race features, `23` hover features), making them suitable for fixed-point encoding and hover-to-race checkpoint warm starts.
- Use Q8 weights with per-output-channel scales for linear layers.
- Use `int32` accumulators for low-bit MACs.
- Keep recurrent state and nonlinear gate math mixed precision initially, then tighten to Q15/Q8 only after closed-loop parity is measured.
- Keep final actions as float or high-resolution fixed point until actuator command tolerances are known.

Validation metrics:

- Float vs Q8 inference latency.
- Float vs Q8 action drift.
- Hover score / gate success rate.
- Crash and invalid-run rates.
- Worst-case inference latency under recurrent state.

Implementation surface:

- `src/puffernet.h`: canonical float C inference path.
- `src/puffernet_q8.h`: Q8/mixed-precision inference path.
- `tests/bench_puffernet_edge.c`: deterministic float-vs-Q8 latency and drift benchmark.

## Scope Boundaries

In scope:

- `ocean/drone/*` for native hover/control-prior benchmarking.
- `ocean/drone_race/*` for native race dynamics and reward/metric logging.
- `config/drone.ini` and `config/drone_race.ini`.
- `scripts/eval_drone_hover_native.py` and `scripts/eval_drone_race_native.py`.
- MAVLink/SITL adapter code, TS-002 camera ingestion, visual gate perception, controller integration, and submission/eval harness.

Out of scope for active race implementation:

- Rebuilding the old v3 Python `pufferlib.environments.drone_race` runtime.
- Hardware flight implementation.

## Training Backbone and Governance Model

Decision:

- Use native v4 `drone` hover/control training as a stability prior and teacher source.
- Use native v4 `drone_race` as a support track for dynamics, robustness, and reward diagnostics, not as the competition acceptance boundary.
- Use SITL/camera/telemetry outputs as the primary promotion surface once the baseline controller exists.
- Add L1-L9 as promotion governance gates over SITL outputs, with native evals retained as preflight/regression gates.

### Tier Backbone

- H0: native `drone` hover/control prior and controller sanity checks.
- R0: native `drone_race` support-track diagnostics under increasingly spec-faithful geometry.
- S0: SITL/MAVLink adapter smoke with telemetry parsing, command-rate enforcement, and deterministic reports.
- S1: SITL/MAVLink + TS-002 camera ingestion with visual gate detection and first-gate visual servoing.
- S2: SITL/MAVLink + vision/perception multi-gate sequencing.
- S3: speed optimization after reliable valid completion.

### L1-L9 Governance

- L-levels do not replace tier configs.
- L-levels decide promotion, remediation, checkpoint/controller acceptance, and whether a native result is allowed to influence the competition-facing path.
- Remediation policy: max 3 attempts per level, targeted remediation between attempts.

## L1-L9 Definitions (Operational)


| Level | Intent                                 | Primary Metric                 | Gate           |
| ----- | -------------------------------------- | ------------------------------ | -------------- |
| L1    | Stable launch/hold behavior            | `valid_run_rate` (easy setup)  | >= 0.90        |
| L2    | Reach first gate reliably              | `first_gate_pass_rate`         | >= 0.90        |
| L3    | Pass one gate reliably                 | `first_gate_pass_rate`         | >= 0.95        |
| L4    | Two-gate sequencing reliability        | `two_gate_pass_rate`           | >= 0.90        |
| L5    | Sequence robustness (turning pressure) | `success_rate` (tier2)         | >= 0.80        |
| L6    | Full lap reliability                   | `success_rate` (tier3)         | >= 0.60        |
| L7    | Speed optimization after reliability   | `median_completion_time_valid` | <= target time |
| L8    | Consistency                            | `max_consecutive_successes`    | >= 10          |
| L9    | Generalization                         | `success_rate` on strict suite | >= 0.50        |


Notes:

- L7 target time remains configurable per track and evaluation suite.
- L9 generalization currently uses strict fixed-seed suite until multi-track interfaces land.

## Execution Strategy

1. Build the observable SITL/camera/telemetry pipeline before treating native race metrics as competition progress.
2. Establish a conservative visual-servo controller that can pass gates using TS-002 camera frames and confirmed MAVLink telemetry.
3. Keep native hover/race training as a support track for control priors, robustness, diagnostics, and teacher signals.
4. Tighten native race timing and geometry to match TS-002 before using native scores as transfer signals.
5. Optimize speed only after sustained valid completion through the SITL interface.

Current priority order:

1. Run short Linux GPU smoke training on `drone_race_competition` to verify the TS-002-shaped policy contract learns anything before spending large training budget.
2. Wire TS-002 vision ingestion into the SITL path: JPEG reassembly, timestamp sync, camera calibration, frame health, and perception boundary.
3. Build a conservative visual gate detector and integrate the existing pose/visual-servo helper for first-gate and short-course qualification.
4. Connect `scripts/drone_policy_contract.py` decoded velocity/yaw-rate outputs to the SITL adapter and deterministic JSON/CSV reports.
5. Add a local/remote SITL eval runner with fixed course/start conditions.
6. Audit and tighten native geometry/gate/chassis fidelity against TS-002 dimensions.
7. Continue R3 native race stability work as a support track, especially vertical-energy diagnostics, but do not let it block SITL baseline work.
8. Clarify/validate telemetry availability before depending on navigation-reference or `ODOMETRY` fields.
9. Define the Windows 11 simulator + Linux/native training integration topology.
10. Continue Q8/fake-quant work only after FP32 closed-loop SITL behavior is reliable.

## Strong and Reusable Patterns (Adopted)

From cross-project analysis, we adopt these patterns:

- Strict level gating with explicit thresholds and budgets.
- Gate-geometry helper as a single tested source for crossing validity.
- Deterministic submission-style local harness behavior.
- Domain-randomization/noise profiles as progressive robustness knobs.

## Extended Implementation Plan

### Phase A: Native v4 Baseline

- Import upstream PufferLib v4 core on `native-v4-drone-port`.
- Add native `ocean/drone_race` C environment and binding.
- Add native hover/race smoke eval helpers.
- Validate native CPU build and native eval smoke for local correctness.
- Add Linux CUDA/NCCL native validation runner (`scripts/validate_native_drone_gpu.sh`).
- Validate CUDA native build on a CUDA/NCCL-capable machine.

### Phase B: Qualifier Timing and Metrics

- Align native `drone_race` config with `120 Hz` physics target and `480s` max run.
- Add dedicated native gate-crossing regression tests or standalone C/Python parity harness.
- Replace race kinematic shortcut with representative quadrotor motor/RK4 dynamics.
- Add hover-compatible race observation contract and staged first-gate to full-race curriculum runner.
- Emit native eval report artifacts (JSON + CSV) per checkpoint.
- Audit native gate/chassis geometry against TS-002 and add spec-faithful clearance tests.
- Improve 3-gate race reliability to promotion threshold as a support track.
- Promote to 4-gate/full-course curriculum only after R3 passes deterministic eval or the SITL baseline demonstrates reliable multi-gate behavior.

### Phase C: MAVLink/SITL Adapter

- Add MAVLink client scaffold for UDP MAVLink v2.
- Maintain heartbeat at `>=2 Hz` in scaffold/dry-run.
- Send `SET_POSITION_TARGET_LOCAL_NED` and/or `SET_ATTITUDE_TARGET` with an engineering target of `>=50 Hz` and strict `<100 Hz` cap in scaffold/dry-run.
- Parse attitude, orientation, linear velocity, status flags, and `TIMESYNC`.
- Do not rely on simulator navigation-reference or `ODOMETRY` data until organizer clarification confirms availability under TS-002.
- Use the selected normalized body velocity/yaw-rate model contract and decode it to MAVLink local-NED velocity/yaw-rate setpoints.
- Add telemetry dropout, command-rate, heartbeat, and time-sync metrics to every SITL run report.

### Phase D: Vision/Perception Path

- Add TS-002 forward-camera UDP ingestion on default port `5600`.
- Implement 24-byte metadata header parsing, JPEG chunk reassembly, chunk-loss detection, and reconstruction-failure reporting.
- Sync frame `sim_time_ns` to telemetry/control logs.
- Integrate pinhole intrinsics (`fx=fy=320`, `cx=320`, `cy=180`, `VFoV=90 deg`, no distortion) and the `20 deg` upward camera tilt relative to body.
- Implement a first-pass square-gate detector and relative pose estimator from image geometry.
- Implement a conservative visual-servo route: center the gate in image space, control closure rate, pass through slowly, and only then increase speed.
- Keep privileged/native state available only for training diagnostics and teacher signals.
- Train/evaluate telemetry-only and telemetry+vision variants separately.

### Phase E: Submission Harness

- Add deterministic submission entrypoint for SITL.
- Add local SITL eval runner with fixed course/start conditions.
- Record and enforce no-human-interaction compliance assumptions for submitted timed runs.
- Add SITL report fields for vision FPS, frame jitter/stalls, JPEG reconstruction failures, chunk loss, `sim_time_ns` sync error, and command-rate cap enforcement.

### Phase F: Edge Inference

- Benchmark native float `forward_puffernet` latency on local CPU.
- Add Q8/mixed-precision PufferNet runtime.
- Add float-vs-Q8 latency and action-drift benchmark.
- Add exporter for trained float weights to packed quantized weights.
- Add closed-loop hover comparison harness for trained `.bin` checkpoints.
- Add fake quantization during native RL fine-tuning.
- Promote Q8 only on closed-loop hover/race metrics, not action drift alone.

## Immediate Sprint (Current)

- Rerun `scripts/watch_official_gate1_validation.py` after a known-good simulator launch and manual R1/RACE start using the discovered Windows-local ports (`--mavlink-port 14550`, `--camera-port 5600`, `--endpoint udpin:0.0.0.0:14550`). Keep `--race-start-check-s 1` enabled so menu/post-crash camera streams return `blocked_race_not_started` before smoke.
- Use `scripts/start_windows_aigp_race.ps1 -Tag <run_id>` as the preferred restart path for this workstation. It may use coordinate clicks for the Unreal UI, but the MAVLink race-start check is the safety gate; do not run validation if the helper exits nonzero.
- Run `scripts/run_sitl_control_calibration.py` after a clean manual race start to find the official simulator command primitive before training more policy checkpoints against the SITL adapter.
- Rerun `--control-mode visual-servo-attitude` with yaw-search and center-gated forward motion immediately after `scripts/start_windows_aigp_race.ps1` reports `race_started=true`. Suggested conservative starting point: `--detector-min-area-px 300 --detector-max-aspect-error 0.8 --detector-min-fill-ratio 0.1 --attitude-servo-search-yaw-rate-rad-s 0.45 --attitude-servo-k-pitch 0.025 --attitude-servo-max-pitch-rate-rad-s 0.12 --attitude-servo-k-yaw 2.2 --attitude-servo-max-yaw-rate-rad-s 1.0 --attitude-servo-k-thrust 0.08 --attitude-servo-max-thrust 0.80 --attitude-servo-forward-yaw-tolerance-rad 0.18 --attitude-servo-forward-z-tolerance-m 0.90 --attitude-servo-uncentered-forward-scale 0.0`.
- Keep pre-arm heartbeat acquisition enabled. Fresh race runs must show `prearm_heartbeats_seen>=1`, `base_mode=193`, `system_status=4`, and nonzero actuator outputs before control tuning evidence is trusted.
- Use simulator race status (`active_gate_index`, `last_gate_race_time`, `race_finish_time_ns`) as the first-gate validation judge; controller-side vision pass heuristics are diagnostic only.
- Use manual simulator reset for promotion runs; keep `--send-sim-reset` diagnostic-only until the blank-camera/high-altitude reset state is fixed.
- Tune the telemetry+camera visual-servo baseline from the healthy manual-reset first-gate view until official `active_gate_index` advances past `0`, then stabilize repeated first-gate validation before speed tuning.
- Connect trained policy outputs through `scripts/drone_policy_contract.py` into the live Windows-local SITL adapter and compare policy mode against the visual-servo baseline.
- Train policies primarily in PufferLib/native `drone_race_competition` for throughput; use the official simulator for validation/data capture/transfer checks unless an organizer-supported fast training API appears.
- Add regression coverage for high-rate TS-002 camera chunk draining and detector behavior on live/captured official frames.
- Document the Windows-local simulator runbook for Linux-trained controllers: `.venv-win`, RTX 3070 workstation, MAVLink telemetry client port `14550`, camera client port `5600`, simulator-owned `14560/5601`, and reset-state requirement.
- Audit native geometry/gate/obstacle fidelity against TS-002 dimensions.
- Continue R3 native stabilization on GPU as a support track, not as the only critical path.

## Promotion and Remediation Policy

- Promotion requires passing all required metrics for the current level.
- If failed:
  1. Run diagnostics over fixed-seed suite.
  2. Apply targeted remediation.
  3. Retry.
- Max attempts per level: 3.

## Experiment Logging Requirements

For each training/eval block, record:

- Checkpoint path and config used.
- Suites evaluated and seed ranges.
- L-level pass/fail outcomes and blocking metric.
- Commands run and status.
- Native checkpoint eval artifacts should be produced with `scripts/eval_drone_race_checkpoint.py` and saved as paired JSON/CSV files under `logs/`.

## Current Vast GPU Run Log

- Vast workspace path: `/root/pufferlib-drone`.
- Active branch/commit used for these runs: `native-v4-drone-port` at `9ec46207` (`add drone race curriculum warm start`).
- 2026-04-28 Vast restart:
  - New RTX 4090 instance: `35774236`.
  - Direct SSH used by the local workstation: `root@89.221.67.131 -p 16604`.
  - Remote workspace restored to `/root/pufferlib-drone`.
  - Local backup mirror created on 2026-04-28: `/Users/anon/code/rl/puffer_ai/vast_backup_20260428`.
- 2026-05-11 Vast check: the previous direct SSH endpoint `root@89.221.67.131 -p 16604` reset during key exchange, so remote repo/checkpoint sync and deterministic R3 re-eval could not be run from this session.
- 2026-05-14 local R3 continuation attempt:
  - Restored `checkpoints/drone_race/1777433724949/0000000026279936.bin` locally from `/Users/anon/code/rl/puffer_ai/vast_backup_20260428/checkpoints/drone_race/1777433724949/0000000026279936.bin`.
  - Rebuilt `drone_race` with `bash build.sh drone_race --cpu` using Homebrew LLVM/libomp and confirmed `scripts/test_drone_race_native_regressions.sh` passes.
  - Deterministic checkpoint eval did not produce JSON/CSV artifacts: the preexisting local `_C` lacked `create_pufferl`, the CPU rebuild still only exposes the VecEnv surface, and the required default-native PufferL backend cannot be built locally because this macOS workstation has no `nvcc`/CUDA.
  - No R3 checkpoint was promoted and no training continuation was run. Resume on Vast/Linux CUDA with a default `bash build.sh drone_race`, not `--cpu` or `--float`.
- Important checkpoint compatibility note: native training checkpoints from the curriculum are default native precision checkpoints. Do not evaluate or continue them from a `_C` build made with `./build.sh drone_race --float`; rebuild with `./build.sh drone_race` first.
- Before shutting down an ephemeral Vast instance, copy at least `checkpoints/`, `logs/`, and any changed source/docs to persistent storage or the local workstation. Do not rely on the instance root disk surviving destroy/relaunch.
- Local backup mirror created on 2026-04-25: `/Users/anon/code/rl/puffer_ai/vast_backup_20260425` (`checkpoints/` and `logs/`, about `44M`).
- Minimum backup from local machine:
  ```bash
  mkdir -p vast_backup
  rsync -av -e "ssh -i ~/.ssh/id_ed25519 -p 26520" \
    root@ssh1.vast.ai:/root/pufferlib-drone/checkpoints/ vast_backup/checkpoints/
  rsync -av -e "ssh -i ~/.ssh/id_ed25519 -p 26520" \
    root@ssh1.vast.ai:/root/pufferlib-drone/logs/ vast_backup/logs/
  ```
- If using a mounted persistent volume on the Vast box, mirror in-place before shutdown:
  ```bash
  rsync -av checkpoints/ /persistent/checkpoints/
  rsync -av logs/ /persistent/logs/
  ```

### Native Race Curriculum Checkpoints

- Native hover/control prior:
  - `checkpoints/drone/1777138104363/0000000039976960.bin`
- Solved H1 one-gate race bridge:
  - `checkpoints/drone_race/1777139155996/0000000049938432.bin`
  - Metrics: `success_rate=0.9995`, `crash=0.0005`, `gates_passed=0.9995`.
- Strong R1 two-gate bridge; use this as the canonical restart point for 3-gate experiments:
  - `checkpoints/drone_race/1777139257574/0000000052494336.bin`
  - Metrics: `success_rate=0.9823`, `crash=0.0177`, `gates_passed=1.9812`.
- Best current 3-gate stabilization checkpoint:
  - `checkpoints/drone_race/1777139626939/0000000052494336.bin`
  - Metrics: `success_rate=0.7990`, `crash=0.2006`, `gates_passed=2.6586`.
  - Not promotion-ready; continue 3-gate stability work before returning to 4 gates.
- Best reproducible 2026-04-28 default-native eval from restored R3 run:
  - `checkpoints/drone_race/1777139626939/0000000039387136.bin`
  - Eval artifact: `logs/drone_race/scan_r3_0000000039387136_1000.json`.
  - Metrics: `success_rate=0.7521`, `crash=0.2479`, `gates_passed=2.5122`.
  - The old dashboard-final R3 metrics remain useful context, but deterministic JSON/CSV eval should govern future promotion.
- Best current R3 checkpoint after 2026-04-28 remediation probes:
  - `checkpoints/drone_race/1777415131824/0000000039387136.bin`
  - Eval artifact: `logs/drone_race/scan_r3_ctrl12_0000000039387136_1000.json`.
  - Metrics: `success_rate=0.8629`, `crash=0.1371`, `gates_passed=2.6843`.
  - Not promotion-ready because crash remains above the `<= 0.10` threshold, but this is the current best R3 lineage candidate.
- R3 low-crash diagnostics added after the above candidate:
  - Metrics now include `crash_low`, `crash_high`, `crash_xy`, `crash_low_z`, and `crash_low_vz`.
  - Re-eval artifact: `logs/drone_race/eval_r3_best_low_crash_detail_2000.json`.
  - Finding: R3 failures are low-altitude dives (`crash_low` accounts for essentially all crashes), averaging just below the floor with high downward velocity.
  - Mild altitude-floor shaping produced one promising 1k checkpoint, `checkpoints/drone_race/1777416809505/0000000026279936.bin` (`success_rate=0.8782`, `crash=0.1218`), but its 2k re-eval regressed to `success_rate=0.8458`, `crash=0.1542`; do not promote it.
- R3 route-position diagnostics added after low-crash diagnostics:
  - Metrics now also include `crash_low_progress`, `crash_low_time`, and low-crash next-gate buckets: `crash_low_next_gate0`, `crash_low_next_gate1`, `crash_low_next_gate2`, `crash_low_next_gate3plus`.
  - Re-eval artifact: `logs/drone_race/eval_r3_best_route_detail_4096.json`.
  - Best-candidate 4k metrics: `success_rate=0.8029`, `crash=0.1971`, `gates_passed=2.6283`; all crashes were low-altitude crashes.
  - Low-crash route split: about `34.5%` while targeting gate 1, `19.6%` while targeting gate 2, and `45.9%` while targeting gate 3; average low-crash progress was `0.4218` and average low-crash time was `4.485s`.
  - Control-penalty bracket probes from the best candidate were not robust:
    - `w_ctrl=0.016` produced a 1k false positive at `checkpoints/drone_race/1777432567340/0000000026279936.bin` (`success_rate=0.9019`, `crash=0.0981`), but 4k re-eval failed at `success_rate=0.7465`, `crash=0.2535`; do not promote it.
    - `w_ctrl=0.014` produced a 1k false positive at `checkpoints/drone_race/1777432719880/0000000039387136.bin` (`success_rate=0.9040`, `crash=0.0960`), but 4k re-eval failed at `success_rate=0.7808`, `crash=0.2192`; do not promote it.
  - Lower time pressure also regressed: `w_time=0.1`, `w_ctrl=0.012` from the best candidate collapsed late, and its best 1k scan was only `success_rate=0.8475`, `crash=0.1525`; do not promote it.
- R3 raised-gate altitude-margin curriculum probes:
  - Goal: train R3 with higher gate altitude as a curriculum split, then evaluate checkpoints back on the original R3 course (`gate_altitude=1.0`) to test whether altitude margin transfers.
  - `gate_altitude=1.35` improved the best-candidate 4k re-eval but did not pass: `checkpoints/drone_race/1777433609185/0000000026279936.bin`, artifact `logs/drone_race/eval_r3_alt135_transfer_best_4096.json`, `success_rate=0.8381`, `crash=0.1619`, `gates_passed=2.7309`.
  - `gate_altitude=1.6` is the best current non-promoted R3 lead: `checkpoints/drone_race/1777433724949/0000000026279936.bin`, artifact `logs/drone_race/eval_r3_alt160_transfer_best_4096.json`, `success_rate=0.8775`, `crash=0.1225`, `gates_passed=2.7847`.
  - `gate_altitude=1.8` regressed and should not be used; its best 1k transfer scan was only `success_rate=0.8514`, `crash=0.1486`.
  - Combining `gate_altitude=1.6` with `w_ctrl=0.014` also regressed; its best 1k transfer scan was `success_rate=0.8511`, `crash=0.1489`.
  - Do not promote any raised-gate candidate yet. The useful signal is that altitude-margin curriculum helps, but all larger evals still fail `crash <= 0.10`.
- R3 vertical-energy/time-to-floor diagnostics:
  - Goal: explain low-altitude dives as time-to-impact and stopping-distance failures, not just altitude-threshold misses.
  - Code now emits `crash_low_floor_margin_pre`, `crash_low_ttf_pre`, `crash_low_stop_margin_pre`, `crash_low_max_up_accel_pre`, `floor_impact_risk`, `floor_stop_violation`, `floor_risk_steps`, `floor_stop_violation_steps`, `floor_risk_sampled`, `min_floor_ttf`, and `min_floor_stop_margin`; `scripts/eval_drone_race_checkpoint.py` also writes conditional averages such as `env/avg_crash_low_ttf_pre`.
  - Local verification: `SDKROOT="$(xcrun --show-sdk-path)" CFLAGS="-O2 -DNDEBUG -isysroot $(xcrun --show-sdk-path)" bash scripts/test_drone_race_native_regressions.sh` passed, including a forced low-floor crash test with negative stopping margin.
  - Deterministic checkpoint eval remains blocked locally: the target checkpoint is now restored under `PufferLib/checkpoints`, but the local CPU `_C` build does not provide the PufferL `create_pufferl`/rollout/load-weight API, and default-native rebuild requires CUDA/NVCC. No candidate is promotable from this diagnostic-only step.
  - Next action: on Vast/Linux CUDA, sync these code/docs, rebuild `drone_race` with `bash build.sh drone_race`, run a 1k diagnostic eval of `checkpoints/drone_race/1777433724949/0000000026279936.bin` with the established R3 config, then run 4k only if it still passes the scan.
- Latest attempted continuation regressed and should not be used for warm start:
  - `checkpoints/drone_race/1777161616339/0000000078708736.bin`: 3 gates, `success_rate=0.3055`, `crash=0.6944`, `gates_passed=1.6427`.
- `scripts/train_drone_race_curriculum.sh` now includes H1, R1, R3, and R4 stages with promotion thresholds. If a stage misses its success/crash threshold, the script exits instead of automatically continuing from a regressed checkpoint.
- Do not continue from these regressed runs unless explicitly investigating failure modes:
  - `checkpoints/drone_race/1777139303583/0000000078708736.bin`: 4 gates, `success_rate=0.5402`, `crash=0.4566`.
  - `checkpoints/drone_race/1777139353818/0000000099942400.bin`: 4 gates, `success_rate=0.4035`, `crash=0.5898`.
  - `checkpoints/drone_race/1777139537143/0000000099942400.bin`: 3 gates, `success_rate=0.4637`, `crash=0.5363`.
  - `checkpoints/drone_race/1777411909778/0000000049938432.bin`: invalid 2026-04-28 continuation made from a `--float` `_C` build; ignore for lineage.
  - `checkpoints/drone_race/1777413297251/0000000099942400.bin`: default-native R3 continuation regressed to `success_rate=0.2017`, `crash=0.7976`.
  - `checkpoints/drone_race/1777413469422/0000000104923136.bin`: relaxed R3 retry regressed to `success_rate=0.4377`, `crash=0.5623`.
  - `checkpoints/drone_race/1777414869351/0000000069992448.bin`: increased `invalid_penalty=55` collapsed to `success_rate=0.0117`, `crash=0.9883`.
  - `checkpoints/drone_race/1777414942471/0000000069992448.bin`: lower `learning_rate=0.00002` final run regressed, though its early checkpoint became an intermediate improvement.
  - `checkpoints/drone_race/1777415036469/0000000069992448.bin`: zero-entropy continuation final run regressed, though its early checkpoint improved R3 before the control-penalty probe.
  - `checkpoints/drone_race/1777416711275/0000000069992448.bin`: strong altitude-floor/descent shaping collapsed to `success_rate=0.0`, `crash=1.0`.
  - `checkpoints/drone_race/1777416809505/0000000069992448.bin`: mild altitude-floor/descent shaping also collapsed late, despite one promising early checkpoint.

## Risks and Controls

- Risk: speed tuning before reliability leads to invalid-run collapse.
  - Control: enforce L6/L8 gates before aggressive speed optimization.
- Risk: native training overfits privileged state and fails the official observable interface.
  - Control: keep MAVLink telemetry/vision adapter as an explicit promotion gate.
- Risk: v4 native CUDA build cannot be validated on local macOS.
  - Control: use native CPU build/eval only for local correctness; run CUDA/NCCL native training on Linux GPU before trusting throughput claims.
- Risk: timing mismatch versus qualifier simulator.
  - Control: use `120 Hz` / `480s` defaults in native race config and SITL tests, and enforce the TS-002 command cap `<100 Hz`.
- Risk: native geometry or obstacle simplifications produce a policy that clears native gates but fails official gate/chassis clearance.
  - Control: audit native gate, frame-depth, chassis, obstacle, and boundary geometry against TS-002 dimensions before treating native results as transferable.
- Risk: telemetry contract assumes TS-001 navigation-reference/`ODOMETRY` data that TS-002 no longer lists.
  - Control: build telemetry-only paths on confirmed TS-002 telemetry fields and request organizer clarification before adding nav-reference dependencies.
- Risk: camera packet loss or timestamp drift destabilizes perception/control in SITL.
  - Control: gate telemetry+vision promotion on JPEG reconstruction, chunk-loss, frame-jitter, and `sim_time_ns` sync metrics.
- Risk: official simulator runtime differs from the Linux training environment.
  - Control: keep Linux/Vast for training, but validate the competition bridge against a Windows 11 DCL simulator host with the required GPU class.
- Risk: old v3 hover checkpoint is not weight-compatible with v4 native models.
  - Control: use v3 hover results as teacher/reference metrics, not as direct v4 checkpoint input.
- Risk: quantized inference passes open-loop action checks but destabilizes closed-loop flight.
  - Control: require hover/race closed-loop parity, crash-rate checks, and worst-case latency checks before using Q8 for competition-facing control.

## Hover Legacy Snapshot (Reference Only)

- Role in active program:
  - The old v3 hover work remains the strongest evidence of stable control behavior.
  - Native v4 `drone` hover is now the control-prior benchmark before native `drone_race`.
- Canonical historical record:
  - `docs/prd_archive_legacy_2026-02-20.md` (full hover R&D timeline and run log).
- Best robust checkpoint lineage (historical):
  - `experiments/drone_hover_robust_176940098863.pt`
  - `experiments/drone_hover_robust_177049920214.pt` (post curriculum-counter persistence fix).
- Recorded robust evaluation outcomes (fixed seeds, deterministic):
  - Wind `0.3`, gains `kp_xy=3.8`, `kd_xy=3.5`: `100/100` success, mean hover `30.000s`.
  - Wind `0.4`, gains `kp_xy=3.8`, `kd_xy=3.5`: `90/100` success, mean hover `27.380s`.
  - Wind `0.4`, gains `kp_xy=4.2`, `kd_xy=3.9`: `94/100` success, mean hover `28.299s`.
- Operational caveat:
  - Best historical performance depends on PD-assisted residual control (`pd_assist=True`, `pd_assist_scale=0.02`, `residual_penalty=0.5`).
  - No-assist hover policies under randomized settings underperformed.
- Current workspace note:
  - Hover CSV/log artifacts are present under `experiments/`, but historical `.pt` checkpoints may be absent locally and may need regeneration.
- Race handoff usage:
  - v3 `.pt` checkpoints are not assumed compatible with v4 native `.bin`/model layouts.
  - Use old hover behavior as teacher/reference data or benchmark, not a direct weight load path.

## Current Program Status

- Branch `3.0` contains the stabilized v3 checkpoint and historical Python race/hover work.
- Branch `native-v4-drone-port` contains the native v4 migration.
- Native `drone` hover CPU build and native eval smoke pass locally.
- Native `drone_race` now uses representative quadrotor motor/RK4 dynamics instead of direct velocity/yaw-rate actions.
- Native `drone_race` CPU build, eval smoke, and gate-crossing/physics/timing regressions pass locally.
- Native CUDA/NCCL validation passes on Vast.ai, and native hover training produced usable `.bin` checkpoints.
- Full native `drone_race` from scratch collapsed at `crash=1.000`; the current path is staged hover-warm-started race curriculum.
- Q8 PufferNet benchmark and closed-loop hover comparison harness compile locally; initial trained-hover Q8 comparison shows useful score retention but action drift remains too high for promotion.
- MAVLink/SITL scaffold dry-run passes locally, with TS-002 `<100 Hz` command validation.
- TS-002 camera packet parser/reassembler scaffold passes synthetic unit tests locally.
- TS-002 corner-based gate pose and visual-servo command helper passes synthetic geometry tests locally.
- TS-002-shaped native policy contract now exists in `drone_race` via `interface_mode = 1`, with local C regressions covering observation layout and velocity/yaw-rate action mapping.
- `drone_race_competition` config loads as a profile over the compiled `drone_race` backend through `backend_env_name = drone_race`.
- Python SITL-side policy action decoding now exists in `scripts/drone_policy_contract.py` and passes unit coverage.
- `scripts/drone_sitl_competition_smoke.py` now emits deterministic artifacts with robust ordered-gate pass/completion fields (confidence-gated, hysteresis/rearm, debounce over missed detections, cooldown, rejection counters); local run artifact paths remain `logs/sitl/competition_smoke_gate1.json` and `logs/sitl/competition_smoke_gate1.csv`.
- Latest canonical replay-backed smoke run (2026-05-30) sent compliant heartbeats/commands, received telemetry/camera traffic, and passed first gate with acceptance enabled (`ordered_gate_passes=1`) in `logs/sitl/competition_smoke_gate1.{json,csv}`.
- The smoke runner now supports strict acceptance gating (`--require-telemetry --require-camera --min-gate-passes 1`) and returns explicit blocker reasons when first-gate conditions are unmet.
- A fast stream preflight tool now exists (`scripts/sitl_stream_probe.py`) to verify inbound MAVLink/camera packet presence before running longer smoke evaluations.
- Windows-local official simulator validation now works on the RTX 3070 workstation. The live ports discovered on 2026-06-04 are MAVLink v2 telemetry from simulator `127.0.0.1:14560` to client port `14550`, TS-002 camera chunks on client port `5600`, and simulator-owned sockets on `14560/5601`.
- Windows-local official simulator smoke runs on 2026-06-04 validated stream/controller health and vision detection, but official first-gate race progress remains blocked. After `PyAIPilotExample` race-status parsing was added, `logs/sitl/official_gate1_validation_summary_windows_local_auto_reset_official_progress_001.json` failed strict official progress with `active_gate_index=0` and `last_gate_race_time=-1`.
- Manual reset snapshot `logs/sitl/reset_snapshot_manual_001/snapshot_summary.json` confirms the current simulator can start in a valid first-gate view: local position near origin, non-black frames, `188` detections in `5s`, and `race_status.active_gate_index=0`. Auto-reset snapshot `logs/sitl/reset_snapshot_auto_001/snapshot_summary.json` is invalid for promotion evidence because it produced black frames and implausible altitude.
- Live camera traffic required bounded packet draining because each frame is chunked into many UDP packets; the smoke runner now drains up to `--camera-max-packets-per-loop` packets per loop (default `512`) and processes the newest completed frame.
- A local mock stream harness now exists (`scripts/mock_ts002_stream.py`) and was used to produce a strict passing artifact run on 2026-05-30 (`ordered_gate_passes=1`, nonzero telemetry/camera, acceptance passed) in `logs/sitl/competition_smoke_gate1_mocked_required.{json,csv}`.
- UDP capture/replay now includes metadata sidecars (`capture_id`, SHA256, timing profile, generator params), with replay-time hash verification and deterministic impairment injection (loss/reorder/latency/jitter) for degraded-stream regression.
- Deterministic regression runner now exists (`scripts/sitl_replay_regression.py`) to run repeated smoke acceptance and paired visual-servo vs policy comparisons on identical replay streams.
- Checkpoint-backed policy callable now exists (`scripts/policy_callable_checkpoint.py`) for wiring trained `drone_race_competition` checkpoints into `--policy-callable`.
- Detector stress suite now exists (`scripts/detector_stress_suite.py`) with frozen threshold config for noise/blur/compression/occlusion/scale.
- `--slowly`/PyTorch backend runs are considered debug-only and are not part of the accepted training path.
- Native CUDA build remains unavailable on local macOS; use Linux GPU hardware for accepted throughput/training validation.
- Latest implementation result (2026-06-04): Windows-local official simulator telemetry/camera traffic is validated, high-rate camera draining works, simulator race status is parsed, and manual reset gives a healthy first-gate camera view. Next required implementation is official first-gate advancement (`active_gate_index>=1`) through the SITL controller from manual reset, while continuing PufferLib/native policy training, R3 stabilization, and quantization as support tracks.
