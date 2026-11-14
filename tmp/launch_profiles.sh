cd /home/dmq/lab3-20261114
nohup bash run_profiles.sh > results/profiles.log 2>&1 < /dev/null &
sleep 1
tail -2 results/suite.log
