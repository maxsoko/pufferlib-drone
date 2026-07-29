param(
    [string]$Python = ".\.venv-win\Scripts\python.exe",
    [string]$Checkpoint = "",
    [switch]$RefinePlant,
    [switch]$RunTransfer,
    [int]$NRepeats = 1,
    [string]$Tag = "track_b_b1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

if ($Checkpoint -ne "") {
    $env:PUFFER_POLICY_CHECKPOINT_PATH = $Checkpoint
}

$argsList = @("scripts/track_b_runner.py", "--phase", "B1")
if ($RefinePlant) { $argsList += "--refine-plant" }
if ($RunTransfer) { $argsList += "--run-transfer" }
$argsList += @("--n-repeats", "$NRepeats", "--transfer-tag", $Tag)

& $Python @argsList
