cd /home/dmq/lab3-20261114
date
ps -C main -o pid,stat,etime,time,pcpu,rss,wchan:24
free -m
tail -6 results/suite.log
perf stat -e cycles,instructions true 2>&1
perf list | head -30
sha256sum main main.cpp main.h
for h in ecs-hw-0002 ecs-hw-0003 ecs-hw-0004; do
  ssh "$h" 'hostname; sha256sum /home/dmq/lab3-20261114/main'
done
