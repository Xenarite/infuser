#pragma once
#include <string>
#include <vector>
#include <fstream>

template <typename vert_t, typename edge_t>
struct Graph {
  size_t n, m;
  vert_t* xadj;
  edge_t* adj;
  vert_t* begin() { return &xadj[0]; }
  vert_t* end() { return &xadj[n + 1]; }
  edge_t* begin(vert_t vertex) { return &adj[xadj[vertex]]; }
  edge_t* end(vert_t vertex) { return &adj[xadj[vertex + 1]]; }
  // auto vertices() { return std::ranges::subrange(this->begin(), this->end()); }
  // auto edges(vert_t v) {
  //   return std::ranges::subrange(this->begin(v), this->end(v));
  // }

  static edge_t snapline(const char* s, size_t& i, size_t& j) {
    sscanf(s, "%zu %zu", &i, &j);
    return edge_t(j);
  }
  Graph& read_txt(std::string filename,
                  edge_t (*func)(const char*, size_t&, size_t&) = snapline) {
    std::ifstream rf(filename);
    if (!rf)
      throw std::runtime_error("Error: Cannot open the file " + filename);
    rf >> n >> m;
    xadj = new vert_t[n + 1];
    adj = new edge_t[m];
    size_t s, t;
    float w = 0;
    vert_t i = 0;
    vert_t j = 0;
    size_t pos = 0;
    vert_t last_seen = 0;
    std::string str;
    getline(rf, str);
    while (getline(rf, str)) {
      edge_t e = func(str.c_str(), i, j);
      while (last_seen <= i) xadj[last_seen++] = pos;
      adj[pos++] = e;
    }
    xadj[last_seen] = pos;
    return *this;
  }
  template<typename F>
  Graph filter(F func, float limit = 1.0f) {
    Graph g;
    g.n = this->n;
    // g.m = limit * this->m;
    g.xadj = new vert_t[n + 1];
    g.adj = new edge_t[size_t(m * limit)];
    size_t pos = 0;
    for (size_t i = 0; i < n; i++) {
      g.xadj[i] = pos;
      for (size_t j = xadj[i];
           pos < m * limit && j < xadj[i + 1]; 
           j++) {
        edge_t e = adj[j];
        if (func(e)) {
          g.adj[pos++] = e;
        }}}
    g.xadj[n+1] = pos;
    g.m=pos;
    return g;
  }
};


struct wedge_t {
  unsigned v;
  signed w;
  unsigned hash;
  signed __align__;
};

typedef Graph<size_t, wedge_t> graph_t;
