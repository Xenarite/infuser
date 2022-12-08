#include "common.h"
#include "graph.h"
#include "hyperfuser.h"
#include "infuser.cuh"

using namespace std;

vector<tuple<size_t, float>> read_cin() {
	vector<tuple<size_t, float>> results;
	for (unsigned long long i; cin >> i;)
		results.push_back(std::make_tuple(i, 0.f));
	return results;
}
int PROC_SIZE = 1, PROC_RANK = 0;
int main(int argc, char* argv[]) {
#ifndef NOMPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &PROC_SIZE);
	MPI_Comm_rank(MPI_COMM_WORLD, &PROC_RANK);
	cerr << "PROCESS " << PROC_RANK << " of " << PROC_SIZE << " initialized!"
		<< endl;
#endif
	int K = 50, R = 64, c, blocksize = 32, NH = 1;
	bool directed = false, sorted = true, oracle = true, harmonic = true;
	float p = 0.01, eps = 0.3, tr = 0.01, trc = 0.02;
	int J = 16;
	string method = "mega", filename;
	ofstream out;
	if (argc < 2) {
		cerr << "Usage: " << argv[0]
			<< " -M [Method=[~MixGreedy/HyperFuser]] -R [#MC=" << R
			<< "] -e [threshold=" << tr << "] -o [output file(optional)]\n";
		exit(-1);
	}

	for (int i = 1; i < argc; i++) {
		string s(argv[i]);
		if (s == "-K") K = atoi(argv[++i]);
		else if (s == "-R") R = atoi(argv[++i]);
		else if (s == "-M") method = string(argv[++i]);
		else if (s == "-e") eps = atof(argv[++i]);
		else if (s == "-t") tr = atof(argv[++i]);
		else if (s == "-s") sorted = atoi(argv[++i]);
		else if (s == "-h") harmonic = atoi(argv[++i]);
		else if (s == "-J") J = atoi(argv[++i]);
		else if (s == "-S") oracle = false;
		else if (s == "-H") NH = atoi(argv[++i]);
		else if (s == "-c") trc = atof(argv[++i]);
		else if (s == "-o") {
			out.open(argv[++i]);
			std::cout.rdbuf(out.rdbuf());
		}

		else
			filename = s;
	}

	graph_t g;
	g.read_txt(filename, [](const char* str, size_t& i, size_t& j) -> wedge_t {
		float w;
		sscanf(str, "%zu %zu %f", &i, &j, &w);
		return wedge_t{ unsigned(j), signed(w * INT_MAX),
					   __hash((((uint64_t)i) << 32) | j), 0 };
		});
	R = R / PROC_SIZE;
	std::cerr << "Batch size of the process is " << R << std::endl;
	std::cerr << "Graph size is " << g.n << "\t" << g.m << std::endl;

	auto rands = get_rands(R * PROC_SIZE);
	auto X = rands.get() + (R * PROC_RANK);


	auto func = [&](wedge_t e) -> bool {
		// return true;
		for (int i = 0; i < R; i++)
			if ((e.hash ^ X[i]) <= e.w)
				return true;
		return false;
	};
	Graph _g = g.filter(func, 1.0f / PROC_SIZE);
	std::swap(g, _g);
	if (!oracle && PROC_RANK==0){
		delete[] _g.adj;
		delete[] _g.xadj;
	}
	std::cerr << "Graph size after pre-sampling is " << g.n << "\t" << g.m << std::endl;

	std::for_each(method.begin(), method.end(),
		[](char& c) { c = ::tolower(c); });
	std::cout << std::fixed << std::setprecision(2);

	std::vector<tuple<size_t, float>> result;
	if (method == "oracle")
		result = read_cin();
	else
		result = infuser_gpu(g, K, R, J, X, NH, harmonic);

	if (oracle && PROC_RANK == 0) {
		R = 64;
		auto cache_ptr = std::make_unique<char[]>(R * g.n);
		auto X = get_rands(R);
		for (auto [s, t] : result) {
			float score = run_ic(_g, s, R, X.get(), cache_ptr.get());
			std::cout << s << "\t" << score << "\t" << t << std::endl;
		}
	}
	else if (PROC_RANK == 0)
		for (auto [s, t] : result) std::cout << s << "\t" << t << std::endl;
#ifndef NOMPI
	MPI_Finalize();
#endif
	return 0;
}
