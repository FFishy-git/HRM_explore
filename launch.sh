#!/usr/bin/env bash
# Launch HRM Sudoku training on SkyPilot (Kubernetes)
#
# Usage:
#   ./launch.sh                                    # 1K subset (default)
#   ./launch.sh sudoku_full_lr3e4_Lc8              # Full dataset
#   ./launch.sh sudoku_1k_lr7e5_wd1 --dryrun      # Dry run
#
# Required:
#   - WANDB_API_KEY env var (or set in .env file)
#   - SkyPilot installed: pip install "skypilot-nightly[kubernetes]"
#   - kubectl configured for your cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT=""
DRYRUN=false

# Parse args
for arg in "$@"; do
  case "$arg" in
    --dryrun) DRYRUN=true ;;
    -*) echo "Unknown flag: $arg"; exit 1 ;;
    *) EXPERIMENT="$arg" ;;
  esac
done

EXPERIMENT="${EXPERIMENT:-sudoku_1k_lr7e5_wd1}"

# ---------- Validate ----------

# Load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
  echo "Loading .env"
  set -a; source "$SCRIPT_DIR/.env"; set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WARNING: WANDB_API_KEY not set. W&B will run in offline mode."
  echo "  Set it: export WANDB_API_KEY=<your-key>"
  echo "  Or add to .env file"
  echo ""
  read -r -p "Continue without W&B logging? [y/N] " response
  if [[ ! "$response" =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# Verify experiment config exists
if [ ! -f "$SCRIPT_DIR/config/experiment/${EXPERIMENT}.yaml" ]; then
  echo "ERROR: Experiment config not found: config/experiment/${EXPERIMENT}.yaml"
  echo "Available experiments:"
  ls "$SCRIPT_DIR/config/experiment/"*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/^/  - /'
  exit 1
fi

# Verify SkyPilot is installed
if ! command -v sky &> /dev/null; then
  echo "ERROR: SkyPilot not installed."
  echo "  Install: pip install \"skypilot-nightly[kubernetes]\""
  exit 1
fi

# ---------- Launch ----------

echo "=== HRM Sudoku Training Launch ==="
echo "Experiment:  ${EXPERIMENT}"
echo "Config:      config/experiment/${EXPERIMENT}.yaml"
echo "Cluster:     kubernetes (H100:8)"
if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "W&B:         enabled (LoopTF-4-CSPs/hrm-sudoku)"
else
  echo "W&B:         offline"
fi
echo ""

CMD=(
  sky launch sky.yaml
  --env "WANDB_API_KEY=${WANDB_API_KEY:-}"
  --env "HRM_EXPERIMENT=${EXPERIMENT}"
  --yes
)

if [ "$DRYRUN" = true ]; then
  echo "[DRY RUN] Would execute:"
  echo "  ${CMD[*]}" | sed "s/WANDB_API_KEY=[^ ]*/WANDB_API_KEY=***/"
  exit 0
fi

echo "Launching..."
cd "$SCRIPT_DIR"
"${CMD[@]}"

echo ""
echo "=== Post-launch commands ==="
echo "  sky status                        # Check cluster status"
echo "  sky logs hrm-sudoku               # Stream logs"
echo "  sky down hrm-sudoku               # Tear down when done"
