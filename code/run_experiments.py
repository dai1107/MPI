#!/usr/bin/env python3
"""Run reproducible correctness and performance experiments on the MPI cluster."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
os.chdir(str(ROOT))
os.environ['PATH'] = '/usr/local/mpich/bin:' + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = '/usr/local/mpich/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

def run(args, log, checked=False):
    args = [str(x) for x in args]
    print('$ ' + ' '.join(shlex.quote(x) for x in args), flush=True)
    start = time.time()
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=1800)
    text = p.stdout
    (RESULTS / log).write_text('$ ' + ' '.join(shlex.quote(x) for x in args) + '\n' + text,
                                encoding='utf-8')
    print(text, flush=True)
    if p.returncode or (checked and (text.count('检查通过') != 3 or '检查失败' in text)):
        raise RuntimeError('failed: ' + log)
    print('DONE {} {:.1f}s'.format(log, time.time()-start), flush=True)
    return text

def generate(name, nodes=200000, degree=20, feat=64, communities=8, extra=()):
    path = name + '.bin'
    run(['python3', 'gen_graph.py', '--nodes', nodes, '--degree', degree,
         '--feat-dim', feat, '--communities', communities, '--seed', 2025,
         '--out', path] + list(extra), 'generate_' + name + '.log')
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for h in ['ecs-hw-0002', 'ecs-hw-0003', 'ecs-hw-0004']:
        subprocess.run(['scp', '-q', path, h + ':' + str(ROOT) + '/'], check=True)
        remote = subprocess.check_output(['ssh', h, 'sha256sum ' + str(ROOT/path)],
                                          universal_newlines=True).split()[0]
        if remote != digest:
            raise RuntimeError('data hash mismatch: ' + h)
    with (RESULTS/'hashes.txt').open('a') as f:
        f.write('{} {} ALL_FOUR_MATCH\n'.format(digest, path))

def test(name, graph, p, host='spread', part='block', layers=2, hidden=64):
    return run(['mpirun', '-f', 'hostfile_' + host, '-np', p, './main',
                '--graph', graph + '.bin', '--part', part, '--layers', layers,
                '--hidden', hidden, '--warmup', 1, '--repeats', 10], name+'.log', True)

def main():
    generate('tiny', nodes=203, degree=1, feat=7, communities=7)
    generate('empty', nodes=3, degree=0, feat=7, communities=1)
    for part in ['block', 'cyclic']:
        for p in [1, 4]:
            for layers in [1, 3]:
                test('check_{}_p{}_l{}'.format(part,p,layers), 'tiny', p,
                     part=part, layers=layers, hidden=11)
        test('check_empty_'+part, 'empty', 4, part=part, layers=3, hidden=11)
    print('CORRECTNESS_ALL_PASS 10 configurations x 3 strategies', flush=True)

    generate('graph')
    for host, p in [('single',1),('single',2),('single',4),('spread',2),
                    ('spread',4),('spread8',8)]:
        test('perf_{}_p{}'.format(host,p), 'graph', p, host=host)
    test('perf_base_cyclic', 'graph', 4, part='cyclic')
    for intra in ['0.10', '0.95']:
        graph = 'intra' + intra.replace('.', '')
        generate(graph, extra=['--intra', intra])
        for part in ['block','cyclic']:
            test('perf_{}_{}'.format(graph,part), graph, 4, part=part)
    for layers, hidden in [(1,64),(4,64),(2,32),(2,128)]:
        test('perf_l{}_h{}'.format(layers,hidden), 'graph', 4,
             layers=layers, hidden=hidden)
    for span in ['0.1','1.0']:
        graph = 'skew' + span.replace('.', '')
        generate(graph, extra=['--skew','0.02','--skew-degree','400','--hub-span',span])
        for part in ['block','cyclic']:
            test('perf_{}_{}'.format(graph,part), graph, 4, part=part)
    print('PERFORMANCE_ALL_PASS 19 configurations x 3 strategies', flush=True)

if __name__ == '__main__':
    main()
