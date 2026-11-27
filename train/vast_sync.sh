#!/usr/bin/env bash
RUN=${1:-dan}
EVERY=${2:-10800}
CODE_REPO=jaay211/dan
cd /workspace/dan

while true; do
    sleep "$EVERY"
    latest=$(ls -t checkpoints/"$RUN"/step_*.pt 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        hf upload "$CODE_REPO" "$latest" "checkpoints/$RUN/latest.pt" \
            --commit-message "Sauvegarde $RUN $(basename "$latest")"
    fi
    hf upload "$CODE_REPO" "checkpoints/$RUN" "checkpoints/$RUN" \
        --include "*.csv" --include "samples.txt" --commit-message "Logs $RUN"
    [ -f "train_$RUN.log" ] && hf upload "$CODE_REPO" "train_$RUN.log" "checkpoints/$RUN/train_$RUN.log"
    date
done
