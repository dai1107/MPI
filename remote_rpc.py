"""Session-only SSH controller. Password arrives on stdin, never on disk."""
import json
import getpass
import sys
import threading
import paramiko
import msvcrt

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.RejectPolicy())
password = getpass.getpass('SSH password: ')
client.connect('114.116.195.186', username='dmq', password=password,
               look_for_keys=False, allow_agent=False, timeout=15)
print('SSH_READY')
def requests():
    while True:
        chars = []
        while True:
            char = msvcrt.getwch()
            if char in ('\r', '\n'):
                break
            chars.append(char)
        if chars:
            yield ''.join(chars)
for line in requests():
    request = json.loads(line)
    try:
        if request['op'] == 'install':
            stdin, stdout, stderr = client.exec_command('sudo -S -p "" yum -y install perf time')
            stdin.write(password + '\n')
            stdin.flush()
            print(stdout.read().decode('utf-8', 'replace'))
            print(stderr.read().decode('utf-8', 'replace'))
            print('RPC_EXIT', stdout.channel.recv_exit_status())
        elif request['op'] == 'exec':
            _, stdout, stderr = client.exec_command(request['cmd'])
            def forward(stream):
                for chunk in iter(lambda: stream.read(2048), b''):
                    print(chunk.decode('utf-8', 'replace'), end='', flush=True)
            t = threading.Thread(target=forward, args=(stderr,))
            t.start()
            forward(stdout)
            t.join()
            print('\nRPC_EXIT', stdout.channel.recv_exit_status())
        elif request['op'] in ('put', 'get'):
            with client.open_sftp() as sftp:
                getattr(sftp, request['op'])(request['src'], request['dst'])
            print('RPC_TRANSFER_OK')
        elif request['op'] == 'close':
            break
    except Exception as e:
        print('RPC_ERROR', type(e).__name__, str(e))
client.close()
