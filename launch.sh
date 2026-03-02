#!/usr/bin/env bash
# Launch HRM training as a SkyPilot managed job.
#
# Usage:
#   ./launch.sh sdk_full_lr3e4_Lc8
#   ./launch.sh sdk_1k_lr7e5_wd1 --dryrun
set -euo pipefail

cd "$(dirname "$0")"

EXPERIMENT=""
DRYRUN=false
for arg in "$@"; do
  case "$arg" in
    --dryrun) DRYRUN=true ;;
    -*) echo "Unknown flag: $arg"; exit 1 ;;
    *) EXPERIMENT="$arg" ;;
  esac
done

if [ -z "$EXPERIMENT" ]; then
  echo "Usage: ./launch.sh <experiment> [--dryrun]"
  echo "Available experiments:"
  ls config/experiment/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/^/  - /'
  exit 1
fi

if [ ! -f "config/experiment/${EXPERIMENT}.yaml" ]; then
  echo "ERROR: config/experiment/${EXPERIMENT}.yaml not found"
  exit 1
fi

# Generate job name: {experiment}-{date}-{git_short_hash}
DATE=$(date +%Y%m%d)
GIT_HASH=$(git rev-parse --short=8 HEAD)
JOB_NAME="${EXPERIMENT}-${DATE}-${GIT_HASH}"

CMD=(
  sky jobs launch sky.yaml -y -d
  -n "$JOB_NAME"
  --env "HRM_EXPERIMENT=${EXPERIMENT}"
  --env WANDB_API_KEY
)

if [ "$DRYRUN" = true ]; then
  echo "[DRY RUN] ${CMD[*]}" | sed "s/WANDB_API_KEY=[^ ]*/WANDB_API_KEY=***/"
  exit 0
fi

echo "Launching: $JOB_NAME"
"${CMD[@]}"
