cd /home/dmq/lab3-20261114
cat results/profiles.log
ls -l results/serial* results/profile* results/time* results/perf*rank0* 2>/dev/null
ps -C serial_profile -o pid,etime,pcpu,rss
