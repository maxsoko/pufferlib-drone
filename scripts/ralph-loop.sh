#!/usr/bin/env bash
set -euo pipefail

# Gist-style "Ralph loop" for Codex:
# - reads PRD + progress
# - runs Codex iteratively
# - asks Codex to append progress updates each iteration
#
# Safe by default:
#   uses: codex exec --full-auto
# Unsafe mode (matches gist spirit):
#   set RALPH_UNSAFE=1 to add --dangerously-bypass-approvals-and-sandbox

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRD_FILE="${PRD_FILE:-PRD.md}"
PROGRESS_FILE="${PROGRESS_FILE:-progress.txt}"
STOP_FILE="${STOP_FILE:-.ralph-stop}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
MODEL="${MODEL:-}"
EXTRA_INSTRUCTIONS="${EXTRA_INSTRUCTIONS:-}"

if [[ ! -f "$PRD_FILE" ]]; then
  echo "Missing PRD file: $PRD_FILE" >&2
  exit 1
fi

if [[ ! -f "$PROGRESS_FILE" ]]; then
  {
    echo "# Ralph Progress"
    echo
    echo "Initialized: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  } > "$PROGRESS_FILE"
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found in PATH" >&2
  exit 1
fi

echo "Starting Ralph loop in: $ROOT_DIR"
echo "PRD: $PRD_FILE"
echo "Progress: $PROGRESS_FILE"
echo "Iterations: $MAX_ITERATIONS"
echo "Stop early: touch $STOP_FILE"

for ((i=1; i<=MAX_ITERATIONS; i++)); do
  if [[ -f "$STOP_FILE" ]]; then
    echo "Stop file detected: $STOP_FILE"
    break
  fi

  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[$timestamp] Iteration $i/$MAX_ITERATIONS"

  prompt_file="$(mktemp)"
  {
    echo "You are in an iterative execution loop for this repository."
    echo
    echo "Inputs:"
    echo "- PRD: $PRD_FILE"
    echo "- Progress log: $PROGRESS_FILE"
    echo
    echo "Goals for this iteration:"
    echo "1. Read the PRD and progress log."
    echo "2. Pick the single highest-priority unfinished task."
    echo "3. Implement it end-to-end with concrete repo changes."
    echo "4. Run relevant validation commands."
    echo "5. Append a concise update to $PROGRESS_FILE with:"
    echo "   - iteration number and UTC timestamp"
    echo "   - what changed"
    echo "   - files touched"
    echo "   - commands run and pass/fail"
    echo "   - blockers/next step"
    echo
    echo "Constraints:"
    echo "- Be incremental, avoid large risky rewrites."
    echo "- If blocked, record blockers and best next action in progress log."
    if [[ -n "$EXTRA_INSTRUCTIONS" ]]; then
      echo
      echo "Extra instructions:"
      echo "$EXTRA_INSTRUCTIONS"
    fi
  } > "$prompt_file"

  codex_args=(exec --cd "$ROOT_DIR" --full-auto)
  if [[ -n "$MODEL" ]]; then
    codex_args+=(-m "$MODEL")
  fi
  if [[ "${RALPH_UNSAFE:-0}" == "1" ]]; then
    codex_args+=(--dangerously-bypass-approvals-and-sandbox)
  fi

  if ! codex "${codex_args[@]}" - < "$prompt_file"; then
    echo "Iteration $i failed. See output above."
    rm -f "$prompt_file"
    exit 1
  fi

  rm -f "$prompt_file"
  sleep "$SLEEP_SECONDS"
done

echo "Ralph loop complete."
