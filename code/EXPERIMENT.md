# 已完成的 MPI 推理实验

`main.cpp` 已补全四个函数；`main.h`、计算核和串行参考保持原样。

- `mpi1`：预先交换节点需求，每层使用 `MPI_Alltoallv` 仅传输去重后的 halo。
- `mpi2`：`MPI_Irecv`/`MPI_Isend` 发起后计算内部节点，`MPI_Waitall` 后计算边界节点。
- `allgather`：每层收集全图表示，显式保存 rank 顺序到全局编号的映射。

## 云端运行

目录为 `/home/dmq/lab3-20261114`，四台机器的路径相同。需要同版本、同路径的 MPICH，且普通用户可通过内网 SSH 免密访问四台机器。

```bash
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
make
# 首次或修改源码后，将 main 分发到其他三台的相同目录。
python3 run_experiments.py
bash run_profiles.sh
```

图生成器使用 NumPy；本次 Python 3.7.9 环境安装 NumPy 1.21.6。`run_experiments.py` 包含全部生成参数、四机数据校验、10 组正确性检查和 19 组性能实验；每组性能配置预热 1 次、正式重复 10 次。

`run_profiles.sh` 会等待主实验结束，再编译独立串行驱动、测量 9 种图/网络配置的串行中位数，最后采集 perf 软件计数及 GNU time 数据。需要主节点安装系统仓库的 perf、time；采集脚本仅包裹 rank 0，不包裹 mpirun。

同一图、层数和维度下，各并行策略统一使用独立串行驱动中的 `gnnSerial` 中位时间计算加速比，避免其他 rank 忙等干扰框架原有的单次串行计时。原始框架输出保留，整理表中的修正基准另有列记录。

## 文件

- `serial_profile.cpp`：同一串行参考，单独运行，预热 1 次、重复 10 次；参数为图路径、hidden、layers。
- `profile_rank.sh`：限定 rank 0 的采集范围。
- `hostfile_single`：0001 上最多 4 个进程；机器实际只有 2 vCPU，因此 4 进程属于超配。
- `hostfile_spread`：4 台各 1 个进程。
- `hostfile_spread8`：4 台各 2 个进程。

图数据未放入提交包。报告、逐步记录、完整 CSV 和原始日志位于提交包的 report/ 与 experiment/。
