#!/bin/bash
# Ralph Wiggum - Long-running AI agent loop
# Usage: ./ralph.sh [--tool amp|claude] [max_iterations]
# Example: ./ralph.sh --tool claude 15

set -e

# Parse arguments
TOOL="amp"  # Default to amp for backwards compatibility
MAX_ITERATIONS=10

while [[ $# -gt 0 ]]; do
  case $1 in
    --tool)
      TOOL="$2"
      shift 2
      ;;
    --tool=*)
      TOOL="${1#*=}"
      shift
      ;;
    *)
      # Assume it's max_iterations if it's a number
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS="$1"
      fi
      shift
      ;;
  esac
done

# Validate tool choice
if [[ "$TOOL" != "amp" && "$TOOL" != "claude" ]]; then
  echo "Error: Invalid tool '$TOOL'. Must be 'amp' or 'claude'."
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
ARCHIVE_DIR="$SCRIPT_DIR/archive"
LAST_BRANCH_FILE="$SCRIPT_DIR/.last-branch"
LOG_FILE="$SCRIPT_DIR/ralph.log"

# Archive previous run if branch changed
if [ -f "$PRD_FILE" ] && [ -f "$LAST_BRANCH_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  LAST_BRANCH=$(cat "$LAST_BRANCH_FILE" 2>/dev/null || echo "")
  
  if [ -n "$CURRENT_BRANCH" ] && [ -n "$LAST_BRANCH" ] && [ "$CURRENT_BRANCH" != "$LAST_BRANCH" ]; then
    # Archive the previous run
    DATE=$(date +%Y-%m-%d)
    # Strip "ralph/" prefix from branch name for folder
    FOLDER_NAME=$(echo "$LAST_BRANCH" | sed 's|^ralph/||')
    ARCHIVE_FOLDER="$ARCHIVE_DIR/$DATE-$FOLDER_NAME"
    
    echo "Archiving previous run: $LAST_BRANCH"
    mkdir -p "$ARCHIVE_FOLDER"
    [ -f "$PRD_FILE" ] && cp "$PRD_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$PROGRESS_FILE" ] && cp "$PROGRESS_FILE" "$ARCHIVE_FOLDER/"
    echo "   Archived to: $ARCHIVE_FOLDER"
    
    # Reset progress file for new run
    echo "# Ralph Progress Log" > "$PROGRESS_FILE"
    echo "Started: $(date)" >> "$PROGRESS_FILE"
    echo "---" >> "$PROGRESS_FILE"
  fi
fi

# Track current branch
if [ -f "$PRD_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  if [ -n "$CURRENT_BRANCH" ]; then
    echo "$CURRENT_BRANCH" > "$LAST_BRANCH_FILE"
  fi
fi

# Initialize progress file if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# Ralph Progress Log" > "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

echo "Starting Ralph - Tool: $TOOL - Max iterations: $MAX_ITERATIONS"
echo "Log file: $LOG_FILE"
echo "" > "$LOG_FILE"  # Reset log file

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "==============================================================="
  echo "  Ralph Iteration $i of $MAX_ITERATIONS ($TOOL)"
  echo "==============================================================="
  echo "" >> "$LOG_FILE"
  echo "===============================================================" >> "$LOG_FILE"
  echo "  Ralph Iteration $i of $MAX_ITERATIONS ($TOOL) - $(date)" >> "$LOG_FILE"
  echo "===============================================================" >> "$LOG_FILE"

  # Run the selected tool with the ralph prompt
  if [[ "$TOOL" == "amp" ]]; then
    OUTPUT=$(cat "$SCRIPT_DIR/prompt.md" | amp --dangerously-allow-all 2>&1 | tee -a "$LOG_FILE" /dev/stderr) || true
  else
    # Claude Code: use --dangerously-skip-permissions for autonomous operation
    # -p sends the prompt as a task, --output-format text captures the final output
    PROMPT=$(cat "$SCRIPT_DIR/CLAUDE.md")
    OUTPUT=$(claude -p "$PROMPT" --dangerously-skip-permissions --output-format stream-json --verbose 2>&1 | tee -a "$LOG_FILE") || true
  fi
  
  # Check for completion signal
  # With stream-json, the signal appears inside JSON text fields. We check
  # that ALL stories in prd.json have passes:true as the authoritative signal,
  # rather than relying on grep which can false-positive on quoted instructions.
  ALL_PASS=$(python3 -c "
import json, sys
with open('$PRD_FILE') as f:
    prd = json.load(f)
stories = prd.get('userStories', [])
if all(s.get('passes', False) for s in stories) and len(stories) > 0:
    print('yes')
else:
    print('no')
" 2>/dev/null || echo "no")
  if [ "$ALL_PASS" = "yes" ]; then
    echo ""
    echo "==============================================================="
    echo "  All tasks complete! Generating summary report..."
    echo "==============================================================="
    echo "" >> "$LOG_FILE"
    echo "===============================================================" >> "$LOG_FILE"
    echo "  Summary Report Generation - $(date)" >> "$LOG_FILE"
    echo "===============================================================" >> "$LOG_FILE"

    REPORT_FILE="$SCRIPT_DIR/report.md"
    SUMMARY_PROMPT="Read scripts/ralph/prd.json and scripts/ralph/progress.txt. Write a concise summary report to scripts/ralph/report.md covering:
1) What was built — list all files created or modified with a one-line description of each
2) Key architectural decisions made during implementation
3) Codebase patterns discovered (from the Codebase Patterns section in progress.txt)
4) Test results — number of tests, what they cover, pass status
5) Any open items, manual steps needed, or recommendations for follow-up

Keep it concise and factual. Use markdown formatting."

    if [[ "$TOOL" == "amp" ]]; then
      echo "$SUMMARY_PROMPT" | amp --dangerously-allow-all 2>&1 | tee -a "$LOG_FILE" /dev/stderr || true
    else
      claude -p "$SUMMARY_PROMPT" --dangerously-skip-permissions --output-format stream-json --verbose 2>&1 | tee -a "$LOG_FILE" || true
    fi

    echo ""
    echo "Ralph completed all tasks at iteration $i of $MAX_ITERATIONS!"
    if [ -f "$REPORT_FILE" ]; then
      echo "Summary report: $REPORT_FILE"
    fi
    exit 0
  fi
  
  echo "Iteration $i complete. Continuing..."
  sleep 2
done

echo ""
echo "Ralph reached max iterations ($MAX_ITERATIONS) without completing all tasks."
echo "Check $PROGRESS_FILE for status."
exit 1
