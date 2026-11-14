// Separate entry point to measure the supplied serial reference without MPI
// strategies or correctness gathers in the hardware-counter collection scope.
#define main framework_main
#include "main.cpp"
#undef main

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    Graph g;
    loadGraph(argc > 1 ? argv[1] : "graph.bin", g);
    int hidden = argc > 2 ? atoi(argv[2]) : 64;
    int layers = argc > 3 ? atoi(argv[3]) : 2;
    vector<int> dims = makeDims(g.featDim, hidden, layers);
    vector<vector<float>> weights;
    makeWeights(dims, 20251018u, weights);
    vector<float> out;
    gnnSerial(g, weights, dims, out);
    vector<double> times;
    for (int i = 0; i < 10; ++i) {
        double t0 = MPI_Wtime();
        gnnSerial(g, weights, dims, out);
        times.push_back(MPI_Wtime() - t0);
    }
    TimeStat t = summarize(times);
    printf("SERIAL median=%.9f min=%.9f max=%.9f checksum=%.9g\n",
           t.median, t.vmin, t.vmax, out.empty() ? 0.0 : out[0]);
    MPI_Finalize();
}
