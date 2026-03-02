#!/bin/bash
# Watch Ralph's log in a human-readable format
# Usage: ./scripts/ralph/watch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/ralph.log"

if [ ! -f "$LOG_FILE" ]; then
  echo "No log file found at $LOG_FILE"
  exit 1
fi

echo "Watching Ralph log: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo "---"

tail -f "$LOG_FILE" | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        t = d.get('type','')
        m = d.get('message',{})
        for c in m.get('content',[]):
            if c.get('type') == 'tool_use':
                print(f'[TOOL] {c[\"name\"]}({json.dumps(c[\"input\"])[:100]})')
            elif c.get('type') == 'text':
                print(f'[TEXT] {c[\"text\"][:200]}')
    except: pass
"
