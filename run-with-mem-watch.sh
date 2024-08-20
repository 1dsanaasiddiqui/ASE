#!/usr/bin/env bash

# Script to run with given params while watching and logging memory.

source ./venv/bin/activate
echo "Activated venv"

function watch_mem {
    while ps -p $WATCHED_PID --no-headers --format "%mem" >> $LOG_FNAME; do
        sleep 0.1s
    done
}

echo "With 15 abs nodes"
python verify_net.py $@ &
WATCHED_PID=$!
LOG_FNAME=logs/mem_on_$(date +%Y_%B_%d_%H_%M_%S)
echo "launched, pid was $WATCHED_PID"
watch_mem
echo done
