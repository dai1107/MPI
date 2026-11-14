cd /home/dmq/lab3-20261114
{
  date -Iseconds
  uname -a
  cat /etc/os-release
  lscpu
  free -m
  /usr/local/mpich/bin/mpirun --version
  gcc --version
  python3 --version
  python3 -c 'import numpy; print("numpy", numpy.__version__)'
  perf --version
  perf list
  cat /proc/sys/kernel/perf_event_paranoid
  sha256sum main main.cpp main.h
  for h in ecs-hw-0001 ecs-hw-0002 ecs-hw-0003 ecs-hw-0004; do
    ssh "$h" 'hostname; nproc; free -m | head -2; sha256sum /home/dmq/lab3-20261114/main'
  done
  cd /home/dmq/mpi-check-20261113
  /usr/local/mpich/bin/mpirun -f hostfile_spread -np 4 ./hello
  /usr/local/mpich/bin/mpirun -f hostfile_single -np 4 ./hello
} > /home/dmq/lab3-20261114/results/environment.log 2>&1
cd /home/dmq/lab3-20261114
perf stat -e cycles,instructions true > results/perf_hardware_probe.log 2>&1
perf stat -e task-clock,context-switches,page-faults true > results/perf_software_probe.log 2>&1
tail -7 results/suite.log
cat results/perf_software_probe.log
