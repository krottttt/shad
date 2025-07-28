#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

struct Edge {
    int to;
    int id;
    int destruction_cost;
};

namespace {
const int kMaxCost = std::numeric_limits<int>::max();

std::vector<std::vector<Edge>> adjacency_list;
std::vector<int> entry_time;
std::vector<int> low_link;
std::vector<bool> visited;
int current_time = 0;
int minimum_bridge_cost = kMaxCost;

void DepthFirstSearch(int current_node, int parent_edge_id) {
    visited[current_node] = true;
    entry_time[current_node] = low_link[current_node] = ++current_time;

    for (const Edge& edge : adjacency_list[current_node]) {
        int neighbor = edge.to;
        if (edge.id == parent_edge_id) {
            continue;
        }

        if (visited[neighbor]) {
            low_link[current_node] =
                std::min(low_link[current_node], entry_time[neighbor]);
        } else {
            DepthFirstSearch(neighbor, edge.id);
            low_link[current_node] =
                std::min(low_link[current_node], low_link[neighbor]);
            if (low_link[neighbor] > entry_time[current_node]) {
                minimum_bridge_cost =
                    std::min(minimum_bridge_cost, edge.destruction_cost);
            }
        }
    }
}
}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int city_count;
    std::cin >> city_count;
    int road_count;
    std::cin >> road_count;

    adjacency_list.resize(city_count + 1);
    entry_time.assign(city_count + 1, -1);
    low_link.assign(city_count + 1, -1);
    visited.assign(city_count + 1, false);

    for (int edge_index = 0; edge_index < road_count; ++edge_index) {
        int from_city;
        std::cin >> from_city;
        int to_city;
        std::cin >> to_city;
        int cost;
        std::cin >> cost;

        adjacency_list[from_city].push_back({to_city, edge_index, cost});
        adjacency_list[to_city].push_back({from_city, edge_index, cost});
    }

    DepthFirstSearch(1, -1);

    if (minimum_bridge_cost == kMaxCost) {
        std::cout << -1 << '\n';
    } else {
        std::cout << minimum_bridge_cost << '\n';
    }

    return 0;
}
