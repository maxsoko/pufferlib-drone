# Windows Official SITL Runbook

Last updated: 2026-06-04

Use this path for official simulator validation on the local RTX 3070 workstation.
Keep native PufferLib training and checkpoint generation separate from this lightweight
Windows controller runtime unless a policy checkpoint is ready to test.

## Validated Topology

- Simulator: `C:\Users\anon\Desktop\AI-GP Simulator v1.0.3364\AIGP_3364`
- Simulator process: `DCGame-Win64-Shipping.exe`
- Controller repo: `C:\Users\anon\code\pufferlib-drone`
- Controller Python: `C:\Users\anon\code\pufferlib-drone\.venv-win\Scripts\python.exe`
- MAVLink client port: `14550`
- Camera client port: `5600`
- Simulator-owned sockets observed: `14560` and `5601`
- Controller endpoint: `udpin:0.0.0.0:14550`

## First-Gate Validation

Reset the simulator/course so the drone starts facing the first gate, then run:

```powershell
cd C:\Users\anon\code\pufferlib-drone
.\scripts\run_windows_official_gate1_validation.ps1 -Tag windows_local_reset_001
```

For competition-progress evidence, require simulator race-status advancement:

```powershell
.\scripts\run_windows_official_gate1_validation.ps1 `
  -Tag windows_local_official_progress_001 `
  -RequireOfficialRaceProgress
```

The simulator's bundled `PyAIPilotExample` also documents a custom MAVLink reset
command, `31000`. The project wrapper exposes it as an optional switch:

```powershell
.\scripts\run_windows_official_gate1_validation.ps1 `
  -Tag windows_local_auto_reset_001 `
  -SendSimReset
```

Treat `-SendSimReset` as experimental until repeated trials prove it returns the
course to the same initial first-gate view as a manual reset.

The run writes deterministic artifacts under `logs\sitl\`:

- `official_gate1_validation_summary_<tag>.json`
- `stream_probe_official_<tag>.json`
- `competition_smoke_gate1_official_<tag>.json`
- `competition_smoke_gate1_official_<tag>.csv`

Acceptance requires nonzero MAVLink telemetry, nonzero TS-002 camera frames,
`heartbeat_hz >= 2`, command rate below `100 Hz`, no command-rate violations,
and at least one ordered gate pass. For official progress, also require
`official_active_gate_index >= 1`; vision-only gate-pass events are diagnostic
until the simulator race status advances.

## Policy Mode

Use policy mode only after a competition-shaped PufferLib checkpoint is exported
for `scripts/policy_callable_checkpoint.py`.

```powershell
$env:PUFFER_POLICY_CHECKPOINT_PATH = "C:\path\to\checkpoint.bin"
.\scripts\run_windows_official_gate1_validation.ps1 `
  -Tag windows_local_policy_001 `
  -ControlMode policy `
  -PolicyCallable "scripts/policy_callable_checkpoint.py:infer"
```

## Repeatability Notes

- Reset the simulator/course before each timed validation trial.
- Use a unique `-Tag` per trial to avoid overwriting artifacts.
- If probe passes but smoke sees zero detections, the simulator is probably not
  at the initial first-gate view.
- If vision pass events appear but `official_active_gate_index` remains `0`, the
  controller has not produced a valid simulator-scored gate pass yet.
- Do not tune speed until repeated reset trials pass reliably.

## Bundled Example Findings

The simulator ships a Python example at:

`C:\Users\anon\Desktop\AI-GP Simulator v1.0.3364\PyAIPilotExample`

Useful confirmed details:

- MAVLink controller endpoint uses `udpin:<host>:14550`.
- Camera binds on `0.0.0.0:5600`.
- Camera packet header is little-endian `<IHHIIQ`.
- Custom simulator reset command is MAVLink command `31000`.
- Race status is sent via `ENCAPSULATED_DATA` type `1`.
- Track information is sent via `DATA_TRANSMISSION_HANDSHAKE` plus
  `ENCAPSULATED_DATA` type `2`.
