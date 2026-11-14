#include "main.h"
#include <climits>

namespace {
struct CommPlan {
    vector<int> recvCounts, recvDispls, sendCounts, sendDispls;
    vector<int> sendLocalIds, haloIds;
    vector<int> allCounts, allDispls, allIds;
    int interiorN = 0;
} plan;

int offsets(const vector<int> &counts, vector<int> &displs) {
    displs.resize(counts.size());
    long long total = 0;
    for (size_t p = 0; p < counts.size(); ++p) {
        if (counts[p] < 0 || total + counts[p] > INT_MAX)
            die("MPI count/displacement exceeds INT_MAX");
        displs[p] = (int)total;
        total += counts[p];
    }
    return (int)total;
}

void scaled(const vector<int> &counts, int dim, vector<int> &out,
            vector<int> &displs) {
    out.resize(counts.size());
    for (size_t p = 0; p < counts.size(); ++p) {
        if ((long long)counts[p] * dim > INT_MAX)
            die("MPI feature count exceeds INT_MAX");
        out[p] = counts[p] * dim;
    }
    offsets(out, displs);
}

void pack(const vector<float> &cur, int dim, vector<float> &send) {
    send.resize(plan.sendLocalIds.size() * dim);
    for (size_t k = 0; k < plan.sendLocalIds.size(); ++k)
        std::copy_n(cur.data() + (size_t)plan.sendLocalIds[k] * dim, dim,
                    send.data() + k * dim);
}

// Group owned rows into an interior prefix and boundary suffix. The supplied
// range kernel then computes each group without changing arithmetic order.
void reorderOwned(const Graph &g, Partition &pt) {
    vector<int> order;
    order.reserve(pt.localN);
    vector<unsigned char> interior(pt.localN, 1);
    for (int i = 0; i < pt.localN; ++i)
        for (int e = pt.locIndptr[i]; e < pt.locIndptr[i + 1]; ++e)
            if (pt.owner[pt.locIndices[e]] != pt.rank) interior[i] = 0;
    for (int i = 0; i < pt.localN; ++i) if (interior[i]) order.push_back(i);
    plan.interiorN = (int)order.size();
    for (int i = 0; i < pt.localN; ++i) if (!interior[i]) order.push_back(i);
    vector<int> ids(pt.localN), ptr(pt.localN + 1, 0), edges;
    vector<float> inv(pt.localN), feat(pt.locFeat.size());
    edges.reserve(pt.locIndices.size());
    for (int i = 0; i < pt.localN; ++i) {
        int old = order[i];
        ids[i] = pt.globalId[old];
        pt.uid[ids[i]] = i;
        inv[i] = pt.invDeg[old];
        std::copy_n(pt.locFeat.data() + (size_t)old * g.featDim, g.featDim,
                    feat.data() + (size_t)i * g.featDim);
        edges.insert(edges.end(), pt.locIndices.begin() + pt.locIndptr[old],
                     pt.locIndices.begin() + pt.locIndptr[old + 1]);
        ptr[i + 1] = (int)edges.size();
    }
    pt.globalId.swap(ids);
    pt.locIndptr.swap(ptr);
    pt.locIndices.swap(edges);
    pt.invDeg.swap(inv);
    pt.locFeat.swap(feat);
}
} // namespace

void gnnSetup(const Graph &g, Partition &pt) {
    plan = CommPlan{};
    reorderOwned(g, pt);
    const int P = pt.worldSize;
    vector<vector<int>> needed(P);
    vector<unsigned char> seen(g.numNodes, 0);
    for (int gid : pt.locIndices) {
        int owner = pt.owner[gid];
        if (owner != pt.rank && !seen[gid]) {
            seen[gid] = 1;
            needed[owner].push_back(gid);
        }
    }
    plan.recvCounts.resize(P);
    for (int p = 0; p < P; ++p) {
        std::sort(needed[p].begin(), needed[p].end());
        plan.recvCounts[p] = (int)needed[p].size();
        plan.haloIds.insert(plan.haloIds.end(), needed[p].begin(), needed[p].end());
    }
    offsets(plan.recvCounts, plan.recvDispls);
    plan.sendCounts.resize(P);
    MPI_Alltoall(plan.recvCounts.data(), 1, MPI_INT, plan.sendCounts.data(), 1,
                 MPI_INT, MPI_COMM_WORLD);
    int sendN = offsets(plan.sendCounts, plan.sendDispls);
    vector<int> requested(sendN);
    MPI_Alltoallv(plan.haloIds.data(), plan.recvCounts.data(), plan.recvDispls.data(),
                  MPI_INT, requested.data(), plan.sendCounts.data(),
                  plan.sendDispls.data(), MPI_INT, MPI_COMM_WORLD);
    plan.sendLocalIds.resize(sendN);
    for (int k = 0; k < sendN; ++k) {
        int gid = requested[k];
        if (gid < 0 || gid >= g.numNodes || pt.owner[gid] != pt.rank)
            die("communication plan requested a node from the wrong owner");
        plan.sendLocalIds[k] = pt.uid[gid];
    }
    remapLocalIndices(pt, plan.haloIds);
    // Rank order is not global-ID order with cyclic partition or reordering.
    plan.allCounts.resize(P);
    MPI_Allgather(&pt.localN, 1, MPI_INT, plan.allCounts.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);
    int allN = offsets(plan.allCounts, plan.allDispls);
    plan.allIds.resize(allN);
    MPI_Allgatherv(pt.globalId.data(), pt.localN, MPI_INT, plan.allIds.data(),
                   plan.allCounts.data(), plan.allDispls.data(), MPI_INT,
                   MPI_COMM_WORLD);
    long long local[6] = {pt.localN, (long long)pt.locIndices.size(), pt.haloN,
                           plan.interiorN, sendN, pt.localN - plan.interiorN};
    vector<long long> stats((size_t)P * 6);
    MPI_Gather(local, 6, MPI_LONG_LONG, stats.data(), 6, MPI_LONG_LONG, 0,
               MPI_COMM_WORLD);
    if (pt.rank == 0) {
        for (int p = 0; p < P; ++p)
            printf("PLAN rank=%d local=%lld edges=%lld halo=%lld interior=%lld send=%lld boundary=%lld\n",
                   p, stats[p*6], stats[p*6+1], stats[p*6+2], stats[p*6+3],
                   stats[p*6+4], stats[p*6+5]);
    }
}

void gnn_mpi_student_1(const Partition &pt, const vector<vector<float>> &weights,
                       const vector<int> &dims, vector<float> &out) {
    vector<float> cur = pt.locFeat, send, H, next;
    vector<int> sc, sd, rc, rd;
    for (size_t l = 0; l < weights.size(); ++l) {
        int din = dims[l], dout = dims[l+1];
        scaled(plan.sendCounts, din, sc, sd);
        scaled(plan.recvCounts, din, rc, rd);
        pack(cur, din, send);
        H.resize((size_t)(pt.localN + pt.haloN) * din);
        std::copy(cur.begin(), cur.end(), H.begin());
        float *halo = pt.haloN ? H.data() + (size_t)pt.localN * din : H.data();
        MPI_Alltoallv(send.data(), sc.data(), sd.data(), MPI_FLOAT,
                      halo, rc.data(), rd.data(), MPI_FLOAT, MPI_COMM_WORLD);
        next.resize((size_t)pt.localN * dout);
        spmmLayer(pt, H, din, weights[l], dout, next, l + 1 < weights.size());
        cur.swap(next);
    }
    out.swap(cur);
}

void gnn_mpi_student_2(const Partition &pt, const vector<vector<float>> &weights,
                       const vector<int> &dims, vector<float> &out) {
    vector<float> cur = pt.locFeat, send, H, next;
    vector<int> sc, sd, rc, rd;
    for (size_t l = 0; l < weights.size(); ++l) {
        int din = dims[l], dout = dims[l+1];
        scaled(plan.sendCounts, din, sc, sd);
        scaled(plan.recvCounts, din, rc, rd);
        pack(cur, din, send);
        H.resize((size_t)(pt.localN + pt.haloN) * din);
        std::copy(cur.begin(), cur.end(), H.begin());
        vector<MPI_Request> reqs;
        reqs.reserve(2 * pt.worldSize);
        for (int p = 0; p < pt.worldSize; ++p) if (rc[p]) {
            MPI_Request req;
            MPI_Irecv(H.data() + (size_t)pt.localN * din + rd[p], rc[p],
                       MPI_FLOAT, p, 37, MPI_COMM_WORLD, &req);
            reqs.push_back(req);
        }
        for (int p = 0; p < pt.worldSize; ++p) if (sc[p]) {
            MPI_Request req;
            MPI_Isend(send.data() + sd[p], sc[p], MPI_FLOAT, p, 37,
                       MPI_COMM_WORLD, &req);
            reqs.push_back(req);
        }
        next.resize((size_t)pt.localN * dout);
        // Send/receive buffers remain valid and untouched during this work.
        spmmLayerRange(pt, H, din, weights[l], dout, next,
                        l + 1 < weights.size(), 0, plan.interiorN);
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        spmmLayerRange(pt, H, din, weights[l], dout, next,
                        l + 1 < weights.size(), plan.interiorN, pt.localN);
        cur.swap(next);
    }
    out.swap(cur);
}

void gnn_mpi_allgather(const Partition &pt, const vector<vector<float>> &weights,
                       const vector<int> &dims, vector<float> &out) {
    vector<float> cur = pt.locFeat, gathered, H, next;
    vector<int> counts, displs;
    for (size_t l = 0; l < weights.size(); ++l) {
        int din = dims[l], dout = dims[l+1];
        scaled(plan.allCounts, din, counts, displs);
        gathered.resize(plan.allIds.size() * din);
        MPI_Allgatherv(cur.data(), pt.localN * din, MPI_FLOAT, gathered.data(),
                       counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
        H.resize((size_t)(pt.localN + pt.haloN) * din);
        for (size_t k = 0; k < plan.allIds.size(); ++k) {
            int uid = pt.uid[plan.allIds[k]];
            if (uid >= 0)
                std::copy_n(gathered.data() + k * din, din,
                            H.data() + (size_t)uid * din);
        }
        next.resize((size_t)pt.localN * dout);
        spmmLayer(pt, H, din, weights[l], dout, next, l + 1 < weights.size());
        cur.swap(next);
    }
    out.swap(cur);
}
