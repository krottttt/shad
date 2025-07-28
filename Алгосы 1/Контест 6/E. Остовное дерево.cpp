#include <algorithm>
#include <iostream>
#include <vector>

struct Edge {
    int from;
    int to;
    int weight;

    Edge(int from_vertex, int to_vertex, int edge_weight)
        : from(from_vertex), to(to_vertex), weight(edge_weight) {}
};

class Graph {
private:
    std::vector<Edge> edges_;
    int num_vertices_;

    int FindSet(std::vector<int>& parent, int vertex) {
        if (parent[vertex] != vertex) {
            parent[vertex] = FindSet(parent, parent[vertex]);
        }
        return parent[vertex];
    }

    void UnionSets(std::vector<int>& parent, std::vector<int>& rank,
                   int vertex_x, int vertex_y) {
        int root_x = FindSet(parent, vertex_x);
        int root_y = FindSet(parent, vertex_y);

        if (root_x == root_y) {
            return;
        }

        if (rank[root_x] < rank[root_y]) {
            parent[root_x] = root_y;
        } else if (rank[root_x] > rank[root_y]) {
            parent[root_y] = root_x;
        } else {
            parent[root_y] = root_x;
            rank[root_x]++;
        }
    }

public:
    explicit Graph(int num_vertices) : num_vertices_(num_vertices) {}

    void AddEdge(int from, int to, int weight) {
        edges_.emplace_back(from - 1, to - 1, weight);
    }

    int FindMaxEdgeInMST() {
        std::sort(edges_.begin(), edges_.end(),
                  [](const Edge& edge_a, const Edge& edge_b) {
                      return edge_a.weight < edge_b.weight;
                  });

        std::vector<int> parent(num_vertices_);
        std::vector<int> rank(num_vertices_, 0);

        for (int i = 0; i < num_vertices_; ++i) {
            parent[i] = i;
        }

        int max_edge_weight = 0;
        int edges_added = 0;

        for (const Edge& edge : edges_) {
            if (FindSet(parent, edge.from) != FindSet(parent, edge.to)) {
                UnionSets(parent, rank, edge.from, edge.to);
                max_edge_weight = edge.weight;
                edges_added++;

                if (edges_added == num_vertices_ - 1) {
                    break;
                }
            }
        }

        return max_edge_weight;
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int num_vertices;
    int num_edges;
    std::cin >> num_vertices >> num_edges;

    Graph graph(num_vertices);
    for (int edge_index = 0; edge_index < num_edges; ++edge_index) {
        int from;
        int to;
        int weight;
        std::cin >> from >> to >> weight;
        graph.AddEdge(from, to, weight);
    }

    int result = graph.FindMaxEdgeInMST();
    std::cout << result << '\n';

    return 0;
}
