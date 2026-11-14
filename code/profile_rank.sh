#!/bin/bash
set -eu
cd /home/dmq/lab3-20261114
strategy=$1
if [ "${PMI_RANK:-0}" = 0 ]; then
  exec /usr/bin/time -v -o "results/time_${strategy}_rank0.log" \
    perf stat -e task-clock,context-switches,page-faults \
    -o "results/perf_${strategy}_rank0.log" \
    ./main --graph graph.bin --only "$strategy" --no-check --warmup 1 --repeats 10
else
  exec ./main --graph graph.bin --only "$strategy" --no-check --warmup 1 --repeats 10
fi
