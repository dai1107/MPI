# 实验三 代码说明

实验要求见上一级目录的《实验三：MPI 分布式图神经网络推理》，集群配置见该文档附录 A。

## 文件

| 文件 | 说明 | 是否需要修改 |
|---|---|---|
| `main.cpp` | 四个待实现的函数，编译入口 | 是，只改这个 |
| `main.h` | 数据结构、计算核、串行参考、结果检查、计时、`main()` | 一般不需要 |
| `gen_graph.py` | 图数据生成 | 不需要 |

`main.cpp` 包含 `main.h`，整个程序是一个编译单元，`make` 只编译这一个文件。改动 `main.h` 会触发全量重编译。

增加策略的做法：在 `main.cpp` 里实现同签名的函数，再到 `main.h` 的 `kStrategies` 表中加一行注册。

## 编译与运行

```bash
make
python3 gen_graph.py --nodes 200000 --out graph.bin
mpirun --hostfile hostfile -np 4 ./main --graph graph.bin
```

`./main --help` 和 `python3 gen_graph.py --help` 列出全部选项。

图数据要分发到所有节点的相同路径，内容必须一致。不一致时结果检查会失败，但报出来的是一堆数值偏差，很难看出根因在数据而不在代码，建议先用 `md5sum` 核对。

## 数据结构

`Graph` 是全图，CSR 格式，每个进程都完整读入一份。

`Partition` 是划分结果，关键字段：

| 字段 | 含义 |
|---|---|
| `localN` / `haloN` | 本进程拥有的节点数 / halo 节点数 |
| `owner[v]` | 全局节点 `v` 属于哪个进程 |
| `globalId[i]` | 本地下标 `i` 对应的全局编号 |
| `uid[v]` | 全局节点 `v` 的统一本地编号，不相关的节点为 `-1` |
| `locIndptr` / `locIndices` | 局部 CSR |
| `invDeg[i]` | `1 / deg`，度数为 0 时是 0 |
| `locFeat` | 本进程节点的输入特征，`localN * featDim` |

## 接口说明

`gnnSetup` 在划分之后、计时开始之前调用一次，用来建立通信计划。计划的数据结构和存放位置不作限定，`main.h` 不会访问它。

`buildPartition` 之后 `pt.locIndices` 里存的是**全局**节点编号，`pt.haloN` 为 0。

`remapLocalIndices(pt, haloGlobalIds)` 按传入列表的顺序给 halo 节点分配本地编号 `[localN, localN + haloN)`，并把 `pt.locIndices` 改写成本地编号。列表不能有重复、不能含本进程拥有的节点，排列顺序不作限定。若 `locIndices` 中出现既不属于本进程又不在列表中的节点，程序会报 "halo 集合不完整" 并退出。

`spmmLayer` / `spmmLayerRange` 要求 `H` 的前 `localN` 行是本地节点、之后 `haloN` 行是 halo 区，调用前 halo 区的数据必须是当前层的。`spmmLayerRange` 多两个参数，只计算本地下标落在 `[rowBegin, rowEnd)` 内的节点。

未实现的函数会输出全零并被结果检查判为失败。

## 调试提示

| 现象 | 常见原因 |
|---|---|
| 报 "halo 集合不完整" | 扫描 `locIndices` 时漏了节点，或忘了去重 |
| 报 "halo 列表中出现重复节点" | 同一个远程节点被加入列表多次 |
| 检查失败，偏差杂乱 | 多层推理只在第一层交换，后续层用了旧数据 |
| 单进程通过，多进程失败 | 本地计算正确，问题在通信 |
| 进程数越多偏差越大 | counts / displs 的单位算错，或漏算了部分节点 |
| 程序卡住不动 | 有进程提前返回，没进入集合通信 |

调试通信问题时用很小的图（`--nodes 200 --layers 1`）比用默认图快得多。多进程的输出是交错的，打印时记得带上 rank。
