#ifndef MAIN_H_
#define MAIN_H_

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using std::vector;

static const int32_t GRAPH_MAGIC = 0x474E4E31;

enum PartMode { PART_BLOCK = 0, PART_CYCLIC = 1 };

struct Graph {
    int numNodes = 0;
    int numEdges = 0;
    int featDim = 0;
    vector<int> indptr;
    vector<int> indices;
    vector<float> feats;
};

struct Partition {
    int rank = 0, worldSize = 1;
    int localN = 0;
    int haloN = 0;

    vector<int> owner;
    vector<int> globalId;
    vector<int> uid;

    vector<int> locIndptr;
    vector<int> locIndices;
    vector<float> invDeg;
    vector<float> locFeat;
};

void loadGraph(const std::string &path, Graph &g);

void buildPartition(const Graph &g, Partition &pt, PartMode mode);

void remapLocalIndices(Partition &pt, const vector<int> &haloGlobalIds);

void spmmLayer(const Partition &pt, const vector<float> &H, int dimIn,
               const vector<float> &W, int dimOut, vector<float> &Hout, bool relu);

void spmmLayerRange(const Partition &pt, const vector<float> &H, int dimIn,
                    const vector<float> &W, int dimOut, vector<float> &Hout,
                    bool relu, int rowBegin, int rowEnd);

void makeWeights(const vector<int> &dims, uint32_t seed, vector<vector<float>> &weights);

vector<int> makeDims(int featDim, int hidden, int numLayers);

void gnnSerial(const Graph &g, const vector<vector<float>> &weights,
               const vector<int> &dims, vector<float> &out);

bool checkResult(const Partition &pt, const vector<float> &localOut, int dim,
                 const vector<float> &serialRef, double atol = 1e-7,
                 double rtol = 1e-7);

struct TimeStat {
    double median = 0, vmin = 0, vmax = 0;
};

TimeStat summarize(vector<double> samples);

double maxAcrossRanks(double v);

static void die(const char *msg) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "[rank %d] 错误: %s\n", rank, msg);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

void loadGraph(const std::string &path, Graph &g) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) die(("无法打开图文件 " + path + "，请先运行 gen_graph.py").c_str());

    int32_t hdr[4];
    if (fread(hdr, sizeof(int32_t), 4, f) != 4) die("图文件头读取失败");
    if (hdr[0] != GRAPH_MAGIC) die("图文件 magic 不匹配");
    g.numNodes = hdr[1];
    g.numEdges = hdr[2];
    g.featDim = hdr[3];

    g.indptr.resize(g.numNodes + 1);
    g.indices.resize(g.numEdges);
    g.feats.resize((size_t)g.numNodes * g.featDim);

    int ok = 0;
    ok += fread(g.indptr.data(), sizeof(int), g.numNodes + 1, f) == (size_t)g.numNodes + 1;
    ok += fread(g.indices.data(), sizeof(int), g.numEdges, f) == (size_t)g.numEdges;
    ok += fread(g.feats.data(), sizeof(float), g.feats.size(), f) == g.feats.size();
    fclose(f);
    if (ok != 3) die("图文件内容不完整");
}

static inline int ownerOf(int v, int n, int P, PartMode mode) {
    if (mode == PART_BLOCK) return (int)((long long)v * P / n);
    return v % P;
}

void buildPartition(const Graph &g, Partition &pt, PartMode mode) {
    const int n = g.numNodes, P = pt.worldSize, me = pt.rank;

    pt.owner.resize(n);
    for (int v = 0; v < n; v++) pt.owner[v] = ownerOf(v, n, P, mode);

    pt.uid.assign(n, -1);
    pt.globalId.clear();
    for (int v = 0; v < n; v++) {
        if (pt.owner[v] == me) {
            pt.uid[v] = (int)pt.globalId.size();
            pt.globalId.push_back(v);
        }
    }
    pt.localN = (int)pt.globalId.size();
    pt.haloN = 0;

    pt.locIndptr.assign(pt.localN + 1, 0);
    for (int i = 0; i < pt.localN; i++) {
        int v = pt.globalId[i];
        pt.locIndptr[i + 1] = pt.locIndptr[i] + (g.indptr[v + 1] - g.indptr[v]);
    }
    pt.locIndices.resize(pt.locIndptr[pt.localN]);
    pt.invDeg.resize(pt.localN);
    pt.locFeat.resize((size_t)pt.localN * g.featDim);

    for (int i = 0; i < pt.localN; i++) {
        int v = pt.globalId[i];
        int deg = g.indptr[v + 1] - g.indptr[v];
        pt.invDeg[i] = deg > 0 ? 1.0f / (float)deg : 0.0f;
        int w = pt.locIndptr[i];
        for (int e = g.indptr[v]; e < g.indptr[v + 1]; e++) pt.locIndices[w++] = g.indices[e];
        memcpy(&pt.locFeat[(size_t)i * g.featDim], &g.feats[(size_t)v * g.featDim],
               sizeof(float) * g.featDim);
    }
}

void remapLocalIndices(Partition &pt, const vector<int> &haloGlobalIds) {
    pt.haloN = (int)haloGlobalIds.size();
    for (int k = 0; k < pt.haloN; k++) {
        int u = haloGlobalIds[k];
        if (u < 0 || u >= (int)pt.owner.size())
            die("remapLocalIndices: halo 列表中出现非法的全局节点编号");
        if (pt.owner[u] == pt.rank) die("remapLocalIndices: halo 列表中混入了本进程拥有的节点");
        if (pt.uid[u] >= pt.localN) die("remapLocalIndices: halo 列表中出现重复节点");
        pt.uid[u] = pt.localN + k;
    }
    for (size_t e = 0; e < pt.locIndices.size(); e++) {
        int gid = pt.locIndices[e];
        int id = pt.uid[gid];
        if (id < 0) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "remapLocalIndices: 全局节点 %d 既不属于本进程，也不在 halo 列表中"
                     "（halo 集合不完整）", gid);
            die(buf);
        }
        pt.locIndices[e] = id;
    }
}

void spmmLayerRange(const Partition &pt, const vector<float> &H, int dimIn,
                    const vector<float> &W, int dimOut, vector<float> &Hout,
                    bool relu, int rowBegin, int rowEnd) {
    vector<double> acc(dimIn);
    for (int i = rowBegin; i < rowEnd; i++) {
        std::fill(acc.begin(), acc.end(), 0.0);

        for (int e = pt.locIndptr[i]; e < pt.locIndptr[i + 1]; e++) {
            const float *hu = &H[(size_t)pt.locIndices[e] * dimIn];
            for (int j = 0; j < dimIn; j++) acc[j] += (double)hu[j];
        }
        double s = (double)pt.invDeg[i];
        for (int j = 0; j < dimIn; j++) acc[j] *= s;

        float *o = &Hout[(size_t)i * dimOut];
        for (int c = 0; c < dimOut; c++) {
            double sum = 0.0;
            for (int j = 0; j < dimIn; j++) sum += acc[j] * (double)W[(size_t)j * dimOut + c];
            if (relu && sum < 0.0) sum = 0.0;
            o[c] = (float)sum;
        }
    }
}

void spmmLayer(const Partition &pt, const vector<float> &H, int dimIn,
               const vector<float> &W, int dimOut, vector<float> &Hout, bool relu) {
    spmmLayerRange(pt, H, dimIn, W, dimOut, Hout, relu, 0, pt.localN);
}

vector<int> makeDims(int featDim, int hidden, int numLayers) {
    vector<int> dims;
    dims.push_back(featDim);
    for (int l = 1; l < numLayers; l++) dims.push_back(hidden);
    dims.push_back(hidden);
    return dims;
}

void makeWeights(const vector<int> &dims, uint32_t seed, vector<vector<float>> &weights) {
    uint32_t s = seed;
    auto nextf = [&s]() {
        s = s * 1664525u + 1013904223u;
        return (float)((s >> 8) * (1.0 / 16777216.0)) * 2.0f - 1.0f;
    };
    weights.clear();
    for (size_t l = 0; l + 1 < dims.size(); l++) {
        int din = dims[l], dout = dims[l + 1];
        float limit = std::sqrt(6.0f / (float)(din + dout));
        vector<float> W((size_t)din * dout);
        for (size_t i = 0; i < W.size(); i++) W[i] = nextf() * limit;
        weights.push_back(std::move(W));
    }
}

void gnnSerial(const Graph &g, const vector<vector<float>> &weights,
               const vector<int> &dims, vector<float> &out) {
    const int n = g.numNodes;
    vector<float> cur = g.feats;
    vector<double> acc;

    for (size_t l = 0; l + 1 < dims.size(); l++) {
        const int din = dims[l], dout = dims[l + 1];
        const bool relu = (l + 2 < dims.size());
        vector<float> nxt((size_t)n * dout);
        acc.assign(din, 0.0);

        for (int v = 0; v < n; v++) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int e = g.indptr[v]; e < g.indptr[v + 1]; e++) {
                const float *hu = &cur[(size_t)g.indices[e] * din];
                for (int j = 0; j < din; j++) acc[j] += (double)hu[j];
            }
            int deg = g.indptr[v + 1] - g.indptr[v];
            double s = deg > 0 ? (double)(1.0f / (float)deg) : 0.0;
            for (int j = 0; j < din; j++) acc[j] *= s;

            float *o = &nxt[(size_t)v * dout];
            for (int c = 0; c < dout; c++) {
                double sum = 0.0;
                for (int j = 0; j < din; j++)
                    sum += acc[j] * (double)weights[l][(size_t)j * dout + c];
                if (relu && sum < 0.0) sum = 0.0;
                o[c] = (float)sum;
            }
        }
        cur.swap(nxt);
    }
    out.swap(cur);
}

bool checkResult(const Partition &pt, const vector<float> &localOut, int dim,
                 const vector<float> &serialRef, double atol, double rtol) {
    const int P = pt.worldSize;
    vector<int> cnts(P), dsps(P);
    int myc = pt.localN;
    MPI_Gather(&myc, 1, MPI_INT, cnts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> allIds;
    vector<float> allVals;
    int total = 0;
    if (pt.rank == 0) {
        for (int p = 0; p < P; p++) {
            dsps[p] = total;
            total += cnts[p];
        }
        allIds.resize(total);
        allVals.resize((size_t)total * dim);
    }
    MPI_Gatherv(pt.globalId.data(), myc, MPI_INT, allIds.data(), cnts.data(), dsps.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> cntsD(P), dspsD(P);
    for (int p = 0; p < P; p++) {
        cntsD[p] = cnts[p] * dim;
        dspsD[p] = dsps[p] * dim;
    }
    MPI_Gatherv(localOut.data(), myc * dim, MPI_FLOAT, allVals.data(), cntsD.data(),
                dspsD.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    int pass = 1;
    if (pt.rank == 0) {
        int bad = 0;
        vector<uint8_t> seen(total, 0);
        for (int k = 0; k < total; k++) {
            int v = allIds[k];
            if (v < 0 || v >= total) {
                printf("  [检查失败] 非法节点编号 %d\n", v);
                pass = 0;
                break;
            }
            if (seen[v]) {
                printf("  [检查失败] 节点 %d 被多个进程重复输出\n", v);
                pass = 0;
                break;
            }
            seen[v] = 1;
            for (int c = 0; c < dim; c++) {
                double got = allVals[(size_t)k * dim + c];
                double ref = serialRef[(size_t)v * dim + c];
                double diff = std::fabs(got - ref);
                if (diff > atol + rtol * std::fabs(ref)) {
                    if (bad < 5)
                        printf("  [检查失败] 节点 %d 第 %d 维: 得到 %.17g, 参考 %.17g,"
                               " 差 %.3g\n", v, c, got, ref, diff);
                    bad++;
                }
            }
        }
        if (pass) {
            for (int v = 0; v < total; v++) {
                if (!seen[v]) {
                    printf("  [检查失败] 节点 %d 没有任何进程输出\n", v);
                    pass = 0;
                    break;
                }
            }
        }
        if (bad > 0) {
            printf("  [检查失败] 共 %d 个元素超出容差\n", bad);
            pass = 0;
        }
    }
    MPI_Bcast(&pass, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return pass != 0;
}

TimeStat summarize(vector<double> samples) {
    TimeStat t;
    if (samples.empty()) return t;
    std::sort(samples.begin(), samples.end());
    t.vmin = samples.front();
    t.vmax = samples.back();
    size_t m = samples.size() / 2;
    t.median = (samples.size() % 2) ? samples[m] : 0.5 * (samples[m - 1] + samples[m]);
    return t;
}

double maxAcrossRanks(double v) {
    double out = v;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return out;
}

void gnnSetup(const Graph &g, Partition &pt);
void gnn_mpi_student_1(const Partition &, const vector<vector<float>> &,
                       const vector<int> &, vector<float> &);
void gnn_mpi_student_2(const Partition &, const vector<vector<float>> &,
                       const vector<int> &, vector<float> &);
void gnn_mpi_allgather(const Partition &, const vector<vector<float>> &,
                       const vector<int> &, vector<float> &);

typedef void (*GnnFn)(const Partition &, const vector<vector<float>> &,
                      const vector<int> &, vector<float> &);

struct Strategy {
    const char *name;
    GnnFn fn;
};

static const Strategy kStrategies[] = {
    {"mpi1", gnn_mpi_student_1},
    {"mpi2", gnn_mpi_student_2},
    {"allgather", gnn_mpi_allgather},
};
static const int kNumStrategies = (int)(sizeof(kStrategies) / sizeof(kStrategies[0]));

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    Partition pt;
    MPI_Comm_rank(MPI_COMM_WORLD, &pt.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &pt.worldSize);

    std::string graphPath = "graph.bin";
    std::string only;
    int hidden = 64, layers = 2, repeats = 10, warmup = 1;
    PartMode part = PART_BLOCK;
    bool doCheck = true;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *opt) {
            if (i + 1 >= argc) {
                if (pt.rank == 0) fprintf(stderr, "选项 %s 缺少参数\n", opt);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            return std::string(argv[++i]);
        };
        if (a == "--graph") graphPath = need("--graph");
        else if (a == "--hidden") hidden = atoi(need("--hidden").c_str());
        else if (a == "--layers") layers = atoi(need("--layers").c_str());
        else if (a == "--repeats") repeats = atoi(need("--repeats").c_str());
        else if (a == "--warmup") warmup = atoi(need("--warmup").c_str());
        else if (a == "--only") only = need("--only");
        else if (a == "--no-check") doCheck = false;
        else if (a == "--part") part = (need("--part") == "cyclic") ? PART_CYCLIC : PART_BLOCK;
        else if (a == "--help") {
            if (pt.rank == 0) {
                printf("用法: mpirun -np <P> %s [选项]\n", argv[0]);
                printf("  --graph <文件>        图数据文件 (默认 graph.bin)\n");
                printf("  --part block|cyclic   划分方式 (默认 block)\n");
                printf("  --hidden <d>          隐层维度 (默认 64)\n");
                printf("  --layers <L>          层数 (默认 2)\n");
                printf("  --repeats <n>         正式重复次数 (默认 10)\n");
                printf("  --warmup <n>          预热次数 (默认 1)\n");
                printf("  --only <策略>         只跑一个策略: ");
                for (int s = 0; s < kNumStrategies; s++) printf("%s ", kStrategies[s].name);
                printf("\n");
                printf("  --no-check            跳过正确性检查与串行参考（仅用于 perf 采集，\n");
                printf("                        此时无 T_serial，加速比一栏显示 -）\n");
            }
            MPI_Finalize();
            return 0;
        }
    }

    Graph g;
    loadGraph(graphPath, g);

    vector<int> dims = makeDims(g.featDim, hidden, layers);
    vector<vector<float>> weights;
    makeWeights(dims, 20251018u, weights);
    const int outDim = dims.back();

    double tPlan0 = MPI_Wtime();
    buildPartition(g, pt, part);
    gnnSetup(g, pt);
    MPI_Barrier(MPI_COMM_WORLD);
    double tPlan = MPI_Wtime() - tPlan0;

    long long lh = pt.haloN, haloSum = 0, haloMax = 0;
    long long crossEdges = 0;
    for (int i = 0; i < pt.localN; i++) {
        int v = pt.globalId[i];
        for (int e = g.indptr[v]; e < g.indptr[v + 1]; e++)
            if (pt.owner[g.indices[e]] != pt.rank) crossEdges++;
    }
    long long crossSum = 0;
    MPI_Allreduce(&lh, &haloSum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&lh, &haloMax, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&crossEdges, &crossSum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    vector<float> serialRef;
    double tSerial = 0.0;
    if (doCheck && pt.rank == 0) {
        double t0 = MPI_Wtime();
        gnnSerial(g, weights, dims, serialRef);
        tSerial = MPI_Wtime() - t0;
    }
    MPI_Bcast(&tSerial, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (pt.rank == 0) {
        printf("============ MPI 分布式 GNN 推理 ============\n");
        printf("图          : %s\n", graphPath.c_str());
        printf("节点/边     : %d / %d    特征维度: %d\n", g.numNodes, g.numEdges, g.featDim);
        printf("进程数      : %d    划分: %s    层数: %d    隐层: %d\n", pt.worldSize,
               part == PART_BLOCK ? "block" : "cyclic", layers, hidden);
        printf("跨进程边    : %lld / %d  (%.1f%%)\n", crossSum, g.numEdges,
               g.numEdges ? 100.0 * crossSum / g.numEdges : 0.0);
        printf("halo 节点   : 合计 %lld, 单进程最多 %lld (本地约 %d)\n", haloSum, haloMax,
               g.numNodes / pt.worldSize);
        printf("划分+建计划 : %.4f s\n", tPlan);
        if (doCheck)
            printf("串行参考    : %.4f s (单进程整图 %d 层)\n", tSerial, layers);
        else
            printf("串行参考    : 已跳过 (--no-check，加速比一栏无意义)\n");
        printf("--------------------------------------------\n");
    }

    for (int s = 0; s < kNumStrategies; s++) {
        if (!only.empty() && only != kStrategies[s].name) continue;

        vector<float> out;

        for (int w = 0; w < warmup; w++) {
            out.assign((size_t)pt.localN * outDim, 0.0f);
            kStrategies[s].fn(pt, weights, dims, out);
        }

        bool ok = true;
        if (doCheck) {
            out.assign((size_t)pt.localN * outDim, 0.0f);
            kStrategies[s].fn(pt, weights, dims, out);
            ok = checkResult(pt, out, outDim, serialRef);
        }

        vector<double> samples;
        for (int r = 0; r < repeats; r++) {
            out.assign((size_t)pt.localN * outDim, 0.0f);
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            kStrategies[s].fn(pt, weights, dims, out);
            double dt = MPI_Wtime() - t0;
            samples.push_back(maxAcrossRanks(dt));
        }
        TimeStat ts = summarize(samples);
        double sp = ts.median > 0 ? tSerial / ts.median : 0.0;

        if (pt.rank == 0) {
            if (doCheck)
                printf("%-10s  中位 %.4f s  最小 %.4f  最大 %.4f  加速比 %.2f  效率 %.2f"
                       "  %s\n", kStrategies[s].name, ts.median, ts.vmin, ts.vmax, sp,
                       sp / pt.worldSize, ok ? "检查通过" : "检查失败");
            else
                printf("%-10s  中位 %.4f s  最小 %.4f  最大 %.4f  加速比 -     效率 -"
                       "     (未检查)\n", kStrategies[s].name, ts.median, ts.vmin, ts.vmax);
        }
    }

    MPI_Finalize();
    return 0;
}

#endif
