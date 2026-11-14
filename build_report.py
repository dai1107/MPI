"""Build the four-page report and a detailed, reproducible experiment guide."""
from pathlib import Path
import json
import re
import unicodedata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'/'experiment'
if not OUT.exists() and (ROOT/'experiment').exists():
    OUT=ROOT/'experiment'
LOGS=OUT/'results'
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
BOLD=FontProperties(fname='C:/Windows/Fonts/msyhbd.ttc')
BLUE='#183a56'
TEAL='#087e8b'
ROWS=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
R={(r['config'],r['strategy']):r for r in ROWS}
def r(config,strategy='mpi2'): return R[('perf_'+config,strategy)]
def t(config,strategy='mpi2'): return f"{r(config,strategy)['median_seconds']:.4f}"
def fmt(value): return f'{value:.2f}'
def log(name): return (LOGS/name).read_text(encoding='utf-8')
def counter(name,event):
    text=log(name)
    return re.search(r'^\s*([\d,.]+)\s+(?:msec\s+)?'+re.escape(event),text,re.M).group(1)
def rss(name): return int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',log(name)).group(1))
def switches(name):
    text=log(name)
    a=re.search(r'Voluntary context switches: (\d+)',text).group(1)
    b=re.search(r'Involuntary context switches: (\d+)',text).group(1)
    return a+'/'+b

def wrap(text,limit=102):
    lines=[]
    for paragraph in text.split('\n'):
        current='';width=0
        tokens=re.findall(r'[A-Za-z0-9_./=:+%<>-]+|.',paragraph)
        for c in tokens:
            w=sum(2 if unicodedata.east_asian_width(ch) in ('W','F') else 1 for ch in c)
            if width+w>limit:
                lines.append(current.rstrip());current='';width=0
            if not current and c.isspace(): continue
            current+=c;width+=w
        lines.append(current)
    return lines

class Page:
    def __init__(self,pdf,number,title,subtitle):
        self.fig=plt.figure(figsize=(8.2677,11.6929),facecolor='white')
        self.pdf=pdf;self.y=754
        self.text_at(48,797,title,19,BLUE,True)
        self.text_at(48,774,subtitle,8.5,'#5e6f7b')
        self.fig.add_artist(Rectangle((48/595,761/842),499/595,2/842,
                                     transform=self.fig.transFigure,color=TEAL))
        self.text_at(48,30,'实验三  |  实测日期：2026-11-14（北京时间）',8,'#6f7b84')
        self.text_at(523,30,f'{number} / 4',8,'#6f7b84')
    def text_at(self,x,y,text,size=10,color='#263d4b',bold=False):
        self.fig.text(x/595,y/842,text,fontproperties=BOLD if bold else FONT,
                      fontsize=size,color=color,va='top')
    def heading(self,text):
        self.y-=10;self.text_at(48,self.y,text,12,TEAL,True);self.y-=25
    def body(self,text,size=9.2):
        for line in wrap(text,108 if size<=9.5 else 98):
            self.text_at(48,self.y,line,size);self.y-=size*1.62
        self.y-=4
    def table(self,headers,rows,widths=None,size=8.5):
        n=len(headers)
        widths=widths or [499/n]*n
        height=24
        for j,row in enumerate([headers]+rows):
            x=48
            bg=BLUE if j==0 else ('#eff5f7' if j%2 else '#ffffff')
            for value,width in zip(row,widths):
                self.fig.add_artist(Rectangle((x/595,(self.y-height)/842),width/595,height/842,
                                               transform=self.fig.transFigure,facecolor=bg,
                                               edgecolor='#d8e2e7',linewidth=.45))
                self.text_at(x+5,self.y-7,str(value),size,'white' if j==0 else '#263d4b',j==0)
                x+=width
            self.y-=height
        self.y-=12
    def finish(self):
        if self.y<47: raise RuntimeError(f'page overflow at {self.y}')
        self.pdf.savefig(self.fig);plt.close(self.fig)

def build_chart():
    configs=['single_p1','single_p2','single_p4','spread_p2','spread_p4','spread8_p8']
    labels=['单机1进程','单机2进程','单机4进程','双机2进程','四机4进程','四机8进程']
    fig,ax=plt.subplots(figsize=(10,3.9),layout='constrained')
    xs=np.arange(6)
    for k,(strategy,color) in enumerate([('mpi1','#087e8b'),('mpi2','#e69931'),('allgather','#537ca0')]):
        ax.bar(xs+(k-1)*.24,[float(t(c,strategy)) for c in configs],.24,label=strategy,color=color)
    ax.set_xticks(xs,labels,fontproperties=FONT,fontsize=10)
    ax.set_ylabel('推理中位时间（秒）',fontproperties=FONT)
    ax.set_yscale('log');ax.set_ylim(.25,8)
    ax.set_yticks([.25,.5,1,2,4,8],['0.25','0.5','1','2','4','8'])
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.legend(frameon=False,ncol=3)
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'scaling.png',dpi=180)
    plt.close(fig)

def build_pdf():
    pdfdir=ROOT/'output'/'pdf';pdfdir.mkdir(exist_ok=True)
    serial=r('spread_p4')['serial_median']
    with PdfPages(pdfdir/'MPI分布式GNN推理实验报告.pdf') as pdf:
        p=Page(pdf,1,'MPI 分布式图神经网络推理','实现、验证与性能分析  |  4 台鲲鹏 ECS，3 种 MPI 策略')
        p.heading('实验结论')
        p.body(f"默认图上，四机四进程 mpi2 为 {t('spread_p4')} 秒，相对独立串行参考 {serial:.4f} 秒，加速比 {r('spread_p4')['speedup']:.2f}，效率 {r('spread_p4')['efficiency']:.2f}。四机八进程降至 {t('spread8_p8')} 秒，但波动更大。单机四进程受两核争用影响，慢于单机双进程。")
        p.body('halo 交换的收益取决于图划分。强社区结构适合 block；高阶节点集中时，cyclic 虽增加通信，却能通过均衡负载缩短耗时。非阻塞实现具备重叠计算路径，但不保证每个配置都更快。')
        p.heading('环境与比较条件')
        p.body('四台 openEuler 20.03 LTS、aarch64；每台 2 vCPU，lscpu 显示 HiSilicon、2.9 GHz；系统可见内存 2844 MiB。MPICH 3.3.2 统一位于 /usr/local/mpich；GCC 7.3.0；Python 3.7.9、NumPy 1.21.6。集群使用私网通信。')
        p.body('编译统一采用 -O3 -std=c++17 -Wall。默认图 N=200000、平均度参数 20、输入维度 64、8 个社区、intra=0.70、seed=2025；实际有向边 3999004。默认网络 2 层、隐层 64，权重种子由框架固定，最后一层不加 ReLU。')
        p.heading('三种实现与通信计划')
        p.body('gnnSetup：扫描局部 CSR，将远程邻居按 owner 分组并去重；Alltoall 交换请求数，Alltoallv 交换全局节点编号；发送方转成本地编号。halo 按发送方分组连续存放，counts/displs 在每层乘输入维度。计划只建一次，每层交换当前表示。')
        p.body('mpi1：打包请求节点，MPI_Alltoallv 仅交换 halo，再调用原有 SpMM 核。mpi2：先 Irecv、再 Isend；本地节点预先重排为内部节点前缀与边界后缀，传输期间计算内部前缀，Waitall 后计算边界。缓冲区在等待完成前保持有效。')
        p.body('allgather：每层用 Allgatherv 收集全部节点表示，再按全局编号放入本地/halo 区。显式保留 rank 顺序到全局编号的映射，适用于 cyclic 和内部节点重排。基线仍需整图表示缓冲区。')
        p.heading('正确性')
        p.body('原有计算核、串行参考、校验逻辑保持不变。10 组边界配置 × 3 策略全部通过：P=1/4、block/cyclic、L=1/3；N=203、d=7、hidden=11，含 77 个孤立点且不能被 4 整除；另用 N=3、0 边、P=4 覆盖空分区。19 组性能配置的 57 项检查也全部通过，容差 atol=rtol=1e-7。')
        p.finish()

        p=Page(pdf,2,'进程分布决定扩展收益','默认图与网络参数固定；时间单位为秒')
        p.heading('计时方法与公平串行基准')
        p.body('每组预热 1 次、正式 10 次。每次计时前 Barrier，使用 MPI_Wtime，取所有进程的最大耗时；计时包含缓冲区分配、打包、通信、等待和计算，排除加载、划分、计划、校验和打印。报告单独列出划分+建计划时间。')
        p.body(f'框架的 rank 0 串行参考会受到其他 rank 忙等的影响，单机四进程尤甚。因此报告使用独立驱动调用同一个 gnnSerial，预热 1 次、重复 10 次；同图/层数/维度统一使用其中位数。默认 T_serial={serial:.6f} 秒。S=T_serial/T_parallel，E=S/P；原始框架结果完整保留。')
        p.y-=4
        ax=p.fig.add_axes([48/595,365/842,499/595,217/842])
        ax.imshow(plt.imread(OUT/'scaling.png'));ax.axis('off')
        p.y=354
        headers=['进程分布','mpi1','mpi2','allgather','mpi2 最小/最大','S / E']
        configs=[('单机 1','single_p1'),('单机 2','single_p2'),('单机 4','single_p4'),
                 ('双机 2','spread_p2'),('四机 4','spread_p4'),('四机 8','spread8_p8')]
        data=[]
        for label,c in configs:
            v=r(c)
            data.append([label,t(c,'mpi1'),t(c),t(c,'allgather'),
                         f"{v['min_seconds']:.4f}/{v['max_seconds']:.4f}",
                         f"{v['speedup']:.2f}/{v['efficiency']:.2f}"])
        p.table(headers,data,[73,58,58,67,139,104],8.3)
        p.body('单机四进程时仅有两核可执行，MPI 忙等和调度竞争使墙钟时间上升；现场 RSS 合计约 1 GiB，未使用交换空间，未观察到内存耗尽。四机八进程用满 8 核，但 halo 总数从 466026 升至 739026，通信与进程调度限制扩展。')
        p.body(f"默认四机四进程的内部节点仅占 {r('spread_p4')['interior_percent']:.2f}%，可供重叠的工作很少。八进程 mpi2 的最小/最大值为 {r('spread8_p8')['min_seconds']:.4f}/{r('spread8_p8')['max_seconds']:.4f} 秒，不能将最小值当稳定性能。")
        p.finish()

        p=Page(pdf,3,'图结构、计算量与负载均衡','四机四进程；除表中变量外采用默认参数')
        p.heading('社区局部性与划分')
        data=[]
        for intra,part,c in [('0.10','block','intra010_block'),('0.10','cyclic','intra010_cyclic'),
                              ('0.70','block','spread_p4'),('0.70','cyclic','base_cyclic'),
                              ('0.95','block','intra095_block'),('0.95','cyclic','intra095_cyclic')]:
            v=r(c)
            data.append([intra,part,f"{v['cross_percent']:.1f}",f"{v['halo_sum']/1000:.1f}",
                         f"{v['interior_percent']:.1f}",t(c,'mpi1'),t(c),t(c,'allgather')])
        p.table(['intra','划分','跨边%','halo/千','内部%','mpi1','mpi2','全量'],data,
                [46,57,54,67,54,73,73,75],8.2)
        p.body(f"intra=0.95 时 block 的 halo 接收量约为 {r('intra095_block')['halo_receive_mib']:.1f} MiB/次（所有 rank 合计），全量基线的逻辑远程接收量约 {r('intra095_block')['allgather_receive_mib']:.1f} MiB/次，不代表物理链路计数。内部节点占 47.4%，但 mpi2 略慢于 mpi1：存在重叠路径不等于一定有收益；计算占比、MPI 进展和额外开销都可能限制效果，本实验未分阶段定量归因。")
        p.heading('层数与隐层维度')
        data=[]
        for label,c in [('L=1 H=64','l1_h64'),('L=2 H=32','l2_h32'),('L=2 H=64','spread_p4'),
                        ('L=2 H=128','l2_h128'),('L=4 H=64','l4_h64')]:
            data.append([label,t(c,'mpi1'),t(c),t(c,'allgather'),f"{r(c)['plan_seconds']:.4f}"])
        p.table(['网络参数','mpi1','mpi2','allgather','划分+计划'],data,[133,91,91,91,93])
        p.body('每层表示都会改变，L 层必须交换 L 次。增加 hidden 不仅增加通信，还提高线性变换的计算量；因此隐层 128 的耗时增幅大于简单的通信字节比例。')
        p.heading('高阶节点集中与分散')
        data=[]
        for span,part,c in [('0.1','block','skew01_block'),('0.1','cyclic','skew01_cyclic'),
                             ('1.0','block','skew10_block'),('1.0','cyclic','skew10_cyclic')]:
            data.append([span,part,f"{r(c)['edge_load_max_mean']:.3f}",
                         f"{r(c)['halo_sum']/1000:.1f}",t(c),t(c,'allgather')])
        p.table(['hub-span','划分','边负载 max/mean','halo/千','mpi2','allgather'],data,
                [69,64,128,75,78,85],8.2)
        p.body('skew=0.02、skew-degree=400：高阶节点集中时 cyclic 用更多 halo 换取均衡。边数负载只是工作量代理，线性变换也按节点付出计算成本，不能把边负载比例直接当作耗时比例。')
        p.finish()

        p=Page(pdf,4,'性能工具、局限与复现','原始日志、完整 min/median/max 与源代码均随压缩包提交')
        p.heading('限定采集范围，避免混入 mpirun')
        p.body('虚拟机 perf list 仅列出软件事件；cycles、instructions 实测为 <not supported>。采用 task-clock、context-switches、page-faults，并用 /usr/bin/time -v 补充 RSS。MPI 在 rank 0 的可执行命令外加采集器，其他 rank 正常运行；不对 mpirun 整体采集。')
        p.body('串行驱动直接调用原 gnnSerial；MPI 使用 --only 策略 --no-check --warmup 1 --repeats 10，避免串行参考和校验污染采集。两者均执行 11 次推理；计数仍包含进程启动、图加载及 MPI 的划分/建计划，不能当作单次纯推理的硬件计数。')
        data=[]
        for label,pf,tf in [('串行','perf_serial.log','time_serial.log'),
                             ('mpi1/rank0','perf_mpi1_rank0.log','time_mpi1_rank0.log'),
                             ('mpi2/rank0','perf_mpi2_rank0.log','time_mpi2_rank0.log'),
                             ('全量/rank0','perf_allgather_rank0.log','time_allgather_rank0.log')]:
            data.append([label,counter(pf,'task-clock'),switches(tf),
                         counter(pf,'page-faults'),f'{rss(tf)/1024:.1f}'])
        p.table(['采集对象','task-clock/ms','切换 V/I','缺页次数','峰值 RSS/MiB'],data,
                [101,105,91,91,111],8.3)
        serial_rss=rss('time_serial.log')/1024
        all_rss=rss('time_allgather_rank0.log')/1024
        p.body(f'全量基线 rank 0 命令树峰值 RSS 为 {all_rss:.1f} MiB；串行为 {serial_rss:.1f} MiB。RSS 是采集范围内最大常驻集，不能相加成精确的集群峰值。当前框架每个 rank 原本就完整加载 CSR 和初始特征，因此 halo 策略也未实现整图存储随 P 缩小。')
        p.body('task-clock 包含 MPI 忙等；缺页以次缺页为主，time 记录 major faults=0、swaps=0，不能据此推断缓存命中率。perf 在当前权限下自动使用 :u，context-switches:u 均为 0，不代表没有调度切换；表中 V/I 改用 time 的自愿/非自愿切换数。MPI 计数只代表 rank 0。')
        p.heading('局限、负优化与可扩展方向')
        p.body('单机超配进程数是明确负优化；四机八进程的长尾波动提示不能只报告最快一次。当前按固定策略顺序运行，云主机噪声、温度和资源共享仍可能影响小幅差异，未做显著性检验。intra=0.95 下两种 halo 策略的微小差异不足以证明谁普遍更优。')
        p.body('扩展可考虑更好的图划分、仅存本地 CSR/特征、按维度复用通信缓冲区，以及在明确 CPU 预算下评估异步进展线程。本实验没有训练、GPU 或工业分布式框架，不能外推到训练任务。')
        p.heading('复现与提交材料')
        p.body('云端目录：/home/dmq/lab3-20261114。执行 make；python3 run_experiments.py；bash run_profiles.sh。脚本含所有图生成命令和参数，生成后自动分发并核对四台 SHA-256。hostfile_single 为单机 4 slots；spread 为每机 1 slot；spread8 为每机 2 slots。')
        p.body('提交包包含 main.cpp/main.h、生成器、Makefile、独立串行驱动、采集脚本、hostfiles、4 页报告、逐步实验记录、完整 CSV 和原始日志；按要求不包含图数据文件。')
        p.finish()

def build_guide():
    serial=r('spread_p4')['serial_median']
    lines=['# 实验三：逐步过程与实测结果','',
           '实验日期：2026-11-14（北京时间）。以下结果来自实际云服务器，不是示例数据。','',
           '## 1. 确认已有集群','',
           '使用 dmq 用户登录主节点；四台均为 openEuler 20.03 LTS / aarch64 / 2 vCPU，系统可见内存 2844 MiB；MPICH 3.3.2，GCC 7.3.0。已验证四机四进程分别落在四台主机，单机四进程均落在 0001。完整证据见 results/environment.log。','',
           '## 2. 理解计算','',
           '每层先按 CSR 聚合邻居特征并除以度数，再乘权重；除最后一层外使用 ReLU。输入特征和每一层表示不同，因此 L 层网络需要 L 次 halo 交换。多个跨分区边可以共享同一个远程节点，halo 必须去重。','',
           '## 3. 建立通信计划','',
           'gnnSetup 扫描远程邻居，按 owner 分组、排序、去重；Alltoall 交换请求数量，Alltoallv 交换请求的全局节点编号。收到请求的 owner 用 uid 转为本地索引，建立发送表。重排本地 CSR 行、特征、全局编号，使内部节点构成前缀，再 remapLocalIndices。每行邻居的顺序保持原样，沿用框架 double 累加的数值语义。','',
           '## 4. 实现三种推理','',
           '- mpi1：打包 → Alltoallv halo → SpMM。','- mpi2：打包 → Irecv/Isend → 计算内部节点 → Waitall → 计算边界节点。','- allgather：Allgatherv 全部表示 → 用显式全局编号映射恢复本地/halo 缓冲区 → SpMM。','',
           'counts/displs 的单位为元素而非字节；节点计数要乘当层输入维度。Isend 完成前发送缓冲区不可修改；Irecv 完成前不能读取 halo。没有修改计算核、串行参考或校验容差。','',
           '## 5. 编译、分发与校验','',
           '```bash','cd /home/dmq/lab3-20261114','make',
           'for h in ecs-hw-0002 ecs-hw-0003 ecs-hw-0004; do',
           '  scp main "$h:/home/dmq/lab3-20261114/"','done','```','',
           '编译选项：-O3 -std=c++17 -Wall。四台可执行文件 SHA-256 一致。每份图生成后自动分发，并核对四台数据 SHA-256，证据保存在 results/hashes.txt。','',
           '## 6. 正确性检查','',
           '```bash','python3 gen_graph.py --nodes 203 --degree 1 --communities 7 --feat-dim 7 --seed 2025 --out tiny.bin',
           'mpirun -f hostfile_spread -np 4 ./main --graph tiny.bin --part cyclic --layers 3 --hidden 11 --warmup 1 --repeats 10','```','',
           'tiny.bin 实际有 203 节点、198 有向边、77 孤立点。对 block/cyclic、P=1/4、L=1/3 的 8 种组合逐一检查；再用 N=3、degree=0、communities=1、P=4、L=3 检查两种划分的空分区。10 组 × 3 策略全部通过，atol=rtol=1e-7。性能实验另外 19 组 × 3 策略也全部通过。','',
           '## 7. 性能实验','',
           '默认图命令：','', '```bash',
           'python3 gen_graph.py --nodes 200000 --degree 20 --feat-dim 64 --communities 8 --intra 0.70 --seed 2025 --out graph.bin',
           'mpirun -f hostfile_spread -np 4 ./main --graph graph.bin --part block --layers 2 --hidden 64 --warmup 1 --repeats 10','```','',
           '默认图 200000 节点、3999004 有向边。正式配置全部预热 1 次、重复 10 次，按最慢 rank 的墙钟时间统计。下表只列中位数（秒），完整最小/最大值见完整数据表.md 和 results.csv。','',
           '| 配置 | mpi1 | mpi2 | allgather | mpi2 加速比 | 效率 |','|---|---:|---:|---:|---:|---:|']
    labels=[('单机1进程','single_p1'),('单机2进程','single_p2'),('单机4进程','single_p4'),
            ('双机2进程','spread_p2'),('四机4进程','spread_p4'),('四机8进程','spread8_p8'),
            ('intra=.70 cyclic','base_cyclic'),('intra=.10 block','intra010_block'),('intra=.10 cyclic','intra010_cyclic'),
            ('intra=.95 block','intra095_block'),('intra=.95 cyclic','intra095_cyclic'),
            ('L=1 H=64','l1_h64'),('L=4 H=64','l4_h64'),('L=2 H=32','l2_h32'),('L=2 H=128','l2_h128'),
            ('skew=.02 span=.1 block','skew01_block'),('skew=.02 span=.1 cyclic','skew01_cyclic'),
            ('skew=.02 span=1 block','skew10_block'),('skew=.02 span=1 cyclic','skew10_cyclic')]
    for label,c in labels:
        v=r(c)
        lines.append(f"| {label} | {t(c,'mpi1')} | {t(c)} | {t(c,'allgather')} | {v['speedup']:.2f} | {v['efficiency']:.2f} |")
    lines+=['',f'默认图加速比的统一串行基准为 {serial:.9f} 秒，来自不启动其他 rank 的独立驱动，调用原有 gnnSerial 并取 10 次中位数。不同图/层数/维度分别有自己的同条件串行基准。原框架中的单次串行时间和原加速比保留在原始日志，报告与整理表采用独立基准以避免 MPI 忙等污染。','',
            '变量设置：intra=0.10/0.70/0.95 均对比 block/cyclic；L=1/2/4、hidden=32/64/128；负载实验 skew=0.02、skew-degree=400、hub-span=0.1/1.0 均对比两种划分。没有把层数或维度不同的结果当作同一任务的加速比。','',
            '## 8. Linux 性能工具','',
            '已安装系统仓库的 perf 和 time。硬件事件 cycles/instructions 不支持，使用 task-clock/context-switches/page-faults。MPI 仅采集 rank 0 的 ./main，其他 rank 正常执行，不把 mpirun 算进去。使用 --no-check 避免包含串行参考和正确性收集。具体命令在 profile_rank.sh 和 run_profiles.sh。','',
            '计数覆盖进程启动、加载、建计划、1 次预热和 10 次正式推理；MPI 不是集群总计数。/usr/bin/time -v 包裹 perf 的命令树，RSS 为该范围的最大常驻集，不能当作所有 rank 同时峰值之和。','',
            '| 对象 | task-clock (ms) | 切换V/I | 缺页 | 峰值RSS (MiB) |','|---|---:|---:|---:|---:|']
    for label,pf,tf in [('串行','perf_serial.log','time_serial.log'),('mpi1/rank0','perf_mpi1_rank0.log','time_mpi1_rank0.log'),
                         ('mpi2/rank0','perf_mpi2_rank0.log','time_mpi2_rank0.log'),('allgather/rank0','perf_allgather_rank0.log','time_allgather_rank0.log')]:
        lines.append(f"| {label} | {counter(pf,'task-clock')} | {switches(tf)} | {counter(pf,'page-faults')} | {rss(tf)/1024:.1f} |")
    lines+=['','CPU时间包含 MPI 忙等；缺页主要与内存分配/首次访问有关，time 显示 major faults=0、swaps=0，不能据此推断缓存命中率。当前 perf 事件自动使用 :u，context-switches:u 均为0，不等于没有调度切换；表中 V/I 为 time 记录的自愿/非自愿切换数。原始日志保留所有明细。','',
            '## 9. 结果理解与局限','',
            '单机四进程在 2 vCPU 上出现明显负优化。默认图中内部节点仅约 1.12%，可重叠工作有限。intra=.95 的 block 内部节点增至 47.43%，但非阻塞实现没有稳定领先集合通信，不能宣称“用了 Isend 就更快”。低社区局部性时 halo 几乎覆盖全部远程节点，Allgather 可能更快。','',
            '高阶节点集中时 cyclic 降低边负载不均衡，可抵消额外通信；高阶节点均匀分散时这种优势会变化。八进程的较大 min/max 范围反映云环境和通信抖动。固定策略顺序及云主机噪声限制小差异的解释；没有做统计显著性检验。','',
            '框架每个 rank 仍完整加载全图，优化的是各层表示通信，不是全图存储分片；性能计数器仅 rank0，硬件事件不可用，不能外推到 GPU 或训练。','',
            '## 10. 一键复现','',
            '```bash','make','python3 run_experiments.py','bash run_profiles.sh','```','',
            '在本机用 analyze_results.py 整理下载日志，再用 build_report.py 生成报告。原始图不放入提交包，所有生成参数均可从 run_experiments.py 与 generate_*.log 还原。云服务器中的实验文件已保留。']
    (OUT/'实验过程与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    assert len(ROWS)==57 and all(x['correct'] and x['serial_median'] for x in ROWS)
    build_chart();build_pdf();build_guide()
    print('Created four-page PDF, chart and detailed guide')
