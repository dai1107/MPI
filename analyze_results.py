"""Convert the original experiment logs into auditable tables."""
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'experiment'
if not OUT.exists() and (ROOT/'experiment').exists():
    OUT = ROOT/'experiment'
LOGS = OUT / 'results'

def match(pattern, text, cast=float):
    found = re.search(pattern, text)
    if not found:
        raise ValueError('missing pattern: '+pattern)
    return cast(found.group(1))

def collect():
    records = []
    for path in sorted(LOGS.glob('perf_*.log')):
        text = path.read_text(encoding='utf-8')
        if '节点/边' not in text:
            continue
        graph = match(r'--graph (\S+)\.bin', text, str)
        hidden = match(r'--hidden (\d+)', text, int)
        layers = match(r'--layers (\d+)', text, int)
        serial_path = LOGS / f'serial_{graph}_h{hidden}_l{layers}.log'
        baseline = None
        if serial_path.exists():
            baseline = match(r'SERIAL median=([\d.]+)', serial_path.read_text())
        plans = [list(map(int, row)) for row in re.findall(
            r'PLAN rank=(\d+) local=(\d+) edges=(\d+) halo=(\d+) interior=(\d+) send=(\d+) boundary=(\d+)', text)]
        loads = [row[2] for row in plans]
        base = dict(config=path.stem, graph=graph, processes=match(r'-np (\d+)',text,int),
                    placement=match(r'-f hostfile_(\S+)',text,str),
                    part=match(r'--part (\S+)',text,str), hidden=hidden, layers=layers,
                    halo_sum=match(r'halo 节点\s*: 合计 (\d+)',text,int),
                    halo_max=match(r'单进程最多 (\d+)',text,int),
                    cross_edges=match(r'跨进程边\s*: (\d+)',text,int),
                    cross_percent=match(r'跨进程边\s*:.*?\(([\d.]+)%\)',text),
                    plan_seconds=match(r'划分\+建计划\s*: ([\d.]+)',text),
                    framework_serial=match(r'串行参考\s*: ([\d.]+)',text),
                    serial_median=baseline,
                    interior_percent=100*sum(row[4] for row in plans)/sum(row[1] for row in plans),
                    edge_load_max_mean=max(loads)/(sum(loads)/len(loads)))
        base['halo_receive_mib'] = base['halo_sum'] * (64 + (layers-1)*hidden)*4/2**20
        base['allgather_receive_mib'] = 200000*(base['processes']-1)*(64+(layers-1)*hidden)*4/2**20
        for row in re.findall(r'^(mpi1|mpi2|allgather)\s+中位 ([\d.]+) s\s+最小 ([\d.]+)\s+最大 ([\d.]+).*?(检查通过|检查失败)',text,re.M):
            strategy,median,low,high,status=row
            record = dict(base, strategy=strategy, median_seconds=float(median),
                          min_seconds=float(low), max_seconds=float(high), correct=status=='检查通过')
            record['speedup'] = baseline/float(median) if baseline else None
            record['efficiency'] = record['speedup']/base['processes'] if baseline else None
            records.append(record)
    return records

def main():
    rows = collect()
    (OUT/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    lines=['# 完整性能结果','',
           '每组预热 1 次、正式运行 10 次；时间单位为秒。加速比统一使用相同图、层数和维度的独立串行参考中位数，原框架单次串行计时保留在日志及 CSV 中。','',
           '| 配置 | 策略 | 中位数 | 最小值 | 最大值 | 加速比 | 效率 | 检查 |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        sp='待测' if r['speedup'] is None else f"{r['speedup']:.2f}"
        eff='待测' if r['efficiency'] is None else f"{r['efficiency']:.2f}"
        lines.append(f"| {r['config']} | {r['strategy']} | {r['median_seconds']:.4f} | {r['min_seconds']:.4f} | {r['max_seconds']:.4f} | {sp} | {eff} | {'通过' if r['correct'] else '失败'} |")
    (OUT/'完整数据表.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(rows)//3} performance configurations, {len(rows)} strategy results')
    for r in rows:
        if r['strategy']=='mpi2':
            print(r['config'],r['median_seconds'], 'halo',r['halo_sum'],
                  'interior %',round(r['interior_percent'],2),'load',round(r['edge_load_max_mean'],3))

if __name__=='__main__':
    main()
