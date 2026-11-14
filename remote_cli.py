"""SSH utility. Supply the password via stdin; it is never saved."""
import sys
import warnings
warnings.filterwarnings('ignore')
import paramiko
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
password = sys.stdin.readline().rstrip('\r\n')
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('114.116.195.186', username='dmq', password=password,
          look_for_keys=False, allow_agent=False, timeout=15)
op = sys.argv[1]
if op in ('exec', 'execfile'):
    if op == 'execfile':
        from pathlib import Path
        command = Path(sys.argv[2]).read_text(encoding='utf-8')
    else:
        command = sys.argv[2]
    _, out, err = c.exec_command(command)
    while not out.channel.exit_status_ready() or out.channel.recv_ready() or out.channel.recv_stderr_ready():
        if out.channel.recv_ready():
            print(out.channel.recv(65536).decode('utf-8','replace'), end='')
        if out.channel.recv_stderr_ready():
            print(out.channel.recv_stderr(65536).decode('utf-8','replace'), end='')
        import time
        time.sleep(0.05)
    status = out.channel.recv_exit_status()
elif op == 'snapshot':
    _,out,err=c.exec_command('cd /home/dmq/lab3-20261114 && tar -czf results-snapshot.tar.gz results hostfile_single hostfile_spread hostfile_spread8')
    status=out.channel.recv_exit_status()
    if status:
        print(err.read().decode('utf-8','replace'))
    else:
        with c.open_sftp() as s:
            s.get('/home/dmq/lab3-20261114/results-snapshot.tar.gz',sys.argv[2])
        print('SNAPSHOT_OK')
elif op == 'install':
    stdin,out,err=c.exec_command('sudo -S -p "" yum -y install perf time')
    stdin.write(password+'\n'); stdin.flush()
    print(out.read().decode('utf-8','replace'))
    print(err.read().decode('utf-8','replace'))
    status=out.channel.recv_exit_status()
elif op in ('put','get'):
    with c.open_sftp() as s:
        getattr(s,op)(sys.argv[2],sys.argv[3])
    print('TRANSFER_OK')
    status=0
c.close()
sys.exit(status)
