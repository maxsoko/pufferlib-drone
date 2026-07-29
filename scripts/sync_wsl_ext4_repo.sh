#!/usr/bin/env bash
# Sync repo from Windows mount to WSL ext4 for stable native training.
set -euo pipefail

DEST="${1:-$HOME/pufferlib-drone}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$DEST"
rsync -a --delete \
    --exclude .venv \
    --exclude checkpoints \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$SRC/" "$DEST/"

echo "Synced $SRC -> $DEST"
echo "Next:"
echo "  cd $DEST"
echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
echo "  TIMESTEPS=131072 bash scripts/train_drone_race_sitl_plant_local.sh"
