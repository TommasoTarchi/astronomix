#!/bin/bash
# usage: wait_jobs.sh jobid... ; exits when every job log has a RESULT or Traceback line
while true; do
  done_all=1
  for j in "$@"; do
    if ! pq log "$j" 2>&1 | grep -qE "^RESULT|Traceback|Error:"; then done_all=0; fi
  done
  [ $done_all = 1 ] && break
  sleep 30
done
for j in "$@"; do pq log "$j" 2>&1 | grep -E "^RESULT|Traceback|Error:" | tail -1; done
