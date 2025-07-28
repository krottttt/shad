#include <bits/stdc++.h>
using namespace std;

constexpr int kMaxNodes = 100000;
constexpr int kMaxEdges = 500000;
constexpr int kMaxLog = 20;

int num_nodes, num_edges, capital_city;
vector<pair<int, int>> adjacency[kMaxNodes + 1];
int edge_from[kMaxEdges];
int edge_to[kMaxEdges];
bool is_bridge[kMaxEdges];

int dfs_timer;
int entry_time[kMaxNodes + 1];
int low_link[kMaxNodes + 1];
bool visited[kMaxNodes + 1];

int component_id[kMaxNodes + 1];
int component_count;

vector<int> component_tree[kMaxNodes + 1];
int depth[kMaxNodes + 1];
int parent[kMaxLog][kMaxNodes + 1];

// === Алгоритмы ===

void FindBridges(int current_node, int parent_edge_id) {
    visited[current_node] = true;
    entry_time[current_node] = low_link[current_node] = ++dfs_timer;

    for (const auto& [neighbor, edge_id] : adjacency[current_node]) {
        if (edge_id == parent_edge_id) {
            continue;
        }
        if (visited[neighbor]) {
            low_link[current_node] =
                min(low_link[current_node], entry_time[neighbor]);
        } else {
            FindBridges(neighbor, edge_id);
            low_link[current_node] =
                min(low_link[current_node], low_link[neighbor]);
            if (low_link[neighbor] > entry_time[current_node]) {
                is_bridge[edge_id] = true;
            }
        }
    }
}

void AssignComponent(int current_node, int current_component_id) {
    component_id[current_node] = current_component_id;
    for (const auto& [neighbor, edge_id] : adjacency[current_node]) {
        if (component_id[neighbor] == 0 && !is_bridge[edge_id]) {
            AssignComponent(neighbor, current_component_id);
        }
    }
}

void BuildComponentTree(int current_component, int parent_component) {
    parent[0][current_component] = parent_component;
    for (int neighbor : component_tree[current_component]) {
        if (neighbor == parent_component) {
            continue;
        }
        depth[neighbor] = depth[current_component] + 1;
        BuildComponentTree(neighbor, current_component);
    }
}

int GetLowestCommonAncestor(int component_a, int component_b) {
    if (depth[component_a] < depth[component_b]) {
        swap(component_a, component_b);
    }

    int depth_diff = depth[component_a] - depth[component_b];
    for (int i = 0; i < kMaxLog; ++i) {
        if ((depth_diff & (1 << i)) != 0) {
            component_a = parent[i][component_a];
        }
    }

    if (component_a == component_b) {
        return component_a;
    }

    for (int i = kMaxLog - 1; i >= 0; --i) {
        if (parent[i][component_a] != parent[i][component_b]) {
            component_a = parent[i][component_a];
            component_b = parent[i][component_b];
        }
    }

    return parent[0][component_a];
}

void ReadGraph() {
    cin >> num_nodes >> num_edges;
    cin >> capital_city;

    for (int i = 0; i < num_edges; ++i) {
        int from_node;
        cin >> from_node;
        int to_node;
        cin >> to_node;

        edge_from[i] = from_node;
        edge_to[i] = to_node;
        adjacency[from_node].emplace_back(to_node, i);
        adjacency[to_node].emplace_back(from_node, i);
    }
}

void PrepareBridgesAndComponents() {
    fill(begin(visited), end(visited), false);
    dfs_timer = 0;
    for (int node = 1; node <= num_nodes; ++node) {
        if (!visited[node]) {
            FindBridges(node, -1);
        }
    }

    fill(begin(component_id), end(component_id), 0);
    component_count = 0;
    for (int node = 1; node <= num_nodes; ++node) {
        if (component_id[node] == 0) {
            ++component_count;
            AssignComponent(node, component_count);
        }
    }

    for (int i = 0; i < num_edges; ++i) {
        if (is_bridge[i]) {
            int comp_a = component_id[edge_from[i]];
            int comp_b = component_id[edge_to[i]];
            component_tree[comp_a].push_back(comp_b);
            component_tree[comp_b].push_back(comp_a);
        }
    }
}

void PreprocessLCA(int root_component) {
    depth[root_component] = 0;
    BuildComponentTree(root_component, root_component);

    for (int i = 1; i < kMaxLog; ++i) {
        for (int j = 1; j <= component_count; ++j) {
            parent[i][j] = parent[i - 1][parent[i - 1][j]];
        }
    }
}

void ProcessQueries() {
    int num_queries;
    cin >> num_queries;

    while (num_queries > 0) {
        --num_queries;

        int yellow_start;
        cin >> yellow_start;
        int blue_start;
        cin >> blue_start;

        int yellow_component = component_id[yellow_start];
        int blue_component = component_id[blue_start];
        int ancestor =
            GetLowestCommonAncestor(yellow_component, blue_component);

        cout << depth[ancestor] << '\n';
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    ReadGraph();
    PrepareBridgesAndComponents();

    int root_component = component_id[capital_city];
    PreprocessLCA(root_component);

    ProcessQueries();
    return 0;
}
