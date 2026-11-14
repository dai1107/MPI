#!/bin/bash
set -eu
cd /home/dmq/lab3-20261114
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# Run after the performance suite to avoid CPU contention between experiments.
while pgrep -f '^python3 -u run_experiments.py$' >/dev/null; do sleep 5; done
mpic++ -O3 -std=c++17 -Wall serial_profile.cpp -o serial_profile
for graph in graph intra010 intra095 skew01 skew10; do
  ./serial_profile "$graph.bin" 64 2 > "results/serial_${graph}_h64_l2.log"
done
for pair in '64 1' '64 4' '32 2' '128 2'; do
  read -r hidden layers <<< "$pair"
  ./serial_profile graph.bin "$hidden" "$layers" > "results/serial_graph_h${hidden}_l${layers}.log"
done
/usr/bin/time -v -o results/time_serial.log \
  perf stat -e task-clock,context-switches,page-faults -o results/perf_serial.log \
  ./serial_profile graph.bin 64 2 > results/profile_serial.log 2>&1
for h in ecs-hw-0002 ecs-hw-0003 ecs-hw-0004; do
  scp -q profile_rank.sh "$h:/home/dmq/lab3-20261114/"
done
for strategy in mpi1 mpi2 allgather; do
  mpirun -f hostfile_spread -np 4 bash ./profile_rank.sh "$strategy" > "results/profile_${strategy}.log" 2>&1
done
echo PROFILES_ALL_DONE
