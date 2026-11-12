#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import struct

import numpy as np

MAGIC = 0x474E4E31


def build_edges(rng, num_nodes, degree, num_comm, intra_ratio):
    comm_size = num_nodes // num_comm
    num_undirected = num_nodes * degree // 2
    num_intra = int(num_undirected * intra_ratio)
    num_inter = num_undirected - num_intra

    comm = rng.integers(0, num_comm, num_intra)
    lo = comm * comm_size
    hi = np.where(comm == num_comm - 1, num_nodes, (comm + 1) * comm_size)
    span = (hi - lo).astype(np.int64)
    src_intra = lo + (rng.random(num_intra) * span).astype(np.int64)
    dst_intra = lo + (rng.random(num_intra) * span).astype(np.int64)

    src_inter = rng.integers(0, num_nodes, num_inter)
    dst_inter = rng.integers(0, num_nodes, num_inter)

    src = np.concatenate([src_intra, src_inter])
    dst = np.concatenate([dst_intra, dst_inter])

    keep = src != dst
    src, dst = src[keep], dst[keep]

    u = np.minimum(src, dst).astype(np.int64)
    v = np.maximum(src, dst).astype(np.int64)
    key = np.unique(u * num_nodes + v)
    u, v = key // num_nodes, key % num_nodes

    return np.concatenate([u, v]), np.concatenate([v, u])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=200000, help="节点数")
    ap.add_argument("--degree", type=int, default=20, help="平均度数")
    ap.add_argument("--communities", type=int, default=8, help="社区数")
    ap.add_argument("--feat-dim", type=int, default=64, help="节点特征维度")
    ap.add_argument("--intra", type=float, default=0.70,
                    help="社区内边占比，越大社区结构越明显 (0~1)")
    ap.add_argument("--skew", type=float, default=0.0,
                    help="度数偏斜：该比例的节点额外连很多边，制造负载不均衡")
    ap.add_argument("--skew-degree", type=int, default=400,
                    help="偏斜节点的额外度数")
    ap.add_argument("--hub-span", type=float, default=0.1,
                    help="偏斜节点集中在前多少比例的编号区间内（1.0 = 全图均匀散布）")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default="graph.bin")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.nodes

    src, dst = build_edges(rng, n, args.degree, args.communities, args.intra)

    if args.skew > 0:
        num_hub = max(1, int(n * args.skew))
        span = max(num_hub, int(n * args.hub_span))
        hubs = rng.choice(span, num_hub, replace=False)
        hs = np.repeat(hubs, args.skew_degree)
        hd = rng.integers(0, n, len(hs))
        keep = hs != hd
        src = np.concatenate([src, hs[keep], hd[keep]])
        dst = np.concatenate([dst, hd[keep], hs[keep]])

    key = np.unique(src.astype(np.int64) * n + dst.astype(np.int64))
    src, dst = key // n, key % n
    indices = dst.astype(np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(np.bincount(src, minlength=n))

    feats = rng.standard_normal((n, args.feat_dim), dtype=np.float32)

    with open(args.out, "wb") as f:
        f.write(struct.pack("<4i", MAGIC, n, len(indices), args.feat_dim))
        f.write(indptr.astype("<i4").tobytes())
        f.write(indices.astype("<i4").tobytes())
        f.write(feats.astype("<f4").tobytes())

    deg = np.diff(indptr)
    comm_size = n // args.communities
    comm_id = np.minimum(np.arange(n) // comm_size, args.communities - 1)
    cross = (comm_id[src] != comm_id[dst]).mean()
    print(f"已写出 {args.out}")
    print(f"  节点数      : {n}")
    print(f"  有向边数    : {len(indices)}")
    print(f"  平均度数    : {deg.mean():.2f} (最大 {deg.max()}, 中位 {int(np.median(deg))},"
          f" 孤立点 {(deg == 0).sum()})")
    print(f"  度数偏斜    : max/mean = {deg.max() / deg.mean():.1f}")
    print(f"  特征维度    : {args.feat_dim}")
    print(f"  跨社区边    : {cross * 100:.1f}%")
    loads = []
    for p_count in (4, 8):
        w = np.array([float(deg[p * n // p_count:(p + 1) * n // p_count].sum())
                      for p in range(p_count)])
        loads.append(f"P={p_count}: {w.max() / w.mean():.2f}")
    print(f"  block 负载  : max/mean  {'  '.join(loads)}")
    size_mb = (16 + 4 * (n + 1) + 4 * len(indices) + 4 * n * args.feat_dim) / 2**20
    print(f"  文件大小    : {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
