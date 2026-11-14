set -eu
cd /home/dmq/lab3-20261114
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
make -B > results/build.log 2>&1
sha256sum main > results/final-binary.sha256
for h in ecs-hw-0002 ecs-hw-0003 ecs-hw-0004; do
  ssh "$h" 'sha256sum /home/dmq/lab3-20261114/main' >> results/final-binary.sha256
done
cat results/build.log
cat results/final-binary.sha256
cat results/profiles.log
grep -E 'CORRECTNESS_ALL_PASS|PERFORMANCE_ALL_PASS' results/suite.log
ps -C main -o pid,stat,etime || true
