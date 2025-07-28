#include <bits/stdc++.h>

using namespace std;

const int kMaxNodes = 200005;

vector<int> depth;
vector<vector<int>> ancestors;
vector<int> alive_ancestor;

int GetAliveAncestor(int node_index) {
    if (alive_ancestor[node_index] == node_index) {
        return node_index;
    }
    return alive_ancestor[node_index] =
               GetAliveAncestor(alive_ancestor[node_index]);
}

int FindLowestCommonAncestor(int first_node, int second_node, int max_steps) {
    if (depth[first_node] < depth[second_node]) {
        swap(first_node, second_node);
    }
    for (int i = max_steps; i >= 0; --i) {
        if (depth[first_node] - (1 << i) >= depth[second_node]) {
            first_node = ancestors[first_node][i];
        }
    }
    if (first_node == second_node) {
        return first_node;
    }
    for (int i = max_steps; i >= 0; --i) {
        if (ancestors[first_node][i] != ancestors[second_node][i]) {
            first_node = ancestors[first_node][i];
            second_node = ancestors[second_node][i];
        }
    }
    return ancestors[first_node][0];
}

void HandleAddition(int parent_index, int& current_node_count, int max_steps) {
    --parent_index;

    depth.push_back(depth[parent_index] + 1);
    vector<int> current_ancestors(max_steps + 1);
    current_ancestors[0] = parent_index;

    for (int i = 1; i <= max_steps; ++i) {
        current_ancestors[i] = ancestors[current_ancestors[i - 1]][i - 1];
    }

    ancestors.push_back(current_ancestors);
    alive_ancestor.push_back(current_node_count);
    ++current_node_count;
}

void HandleDeletion(int removed_node_index) {
    --removed_node_index;
    alive_ancestor[removed_node_index] = ancestors[removed_node_index][0];
}

void HandleQuery(int first_query_node, int second_query_node, int max_steps) {
    --first_query_node;
    --second_query_node;

    int common_ancestor = FindLowestCommonAncestor(
        first_query_node, second_query_node, max_steps);
    cout << GetAliveAncestor(common_ancestor) + 1 << '\n';
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int query_count;
    cin >> query_count;

    int max_steps = 0;
    while ((1 << max_steps) <= query_count) {
        ++max_steps;
    }

    depth.reserve(kMaxNodes);
    ancestors.reserve(kMaxNodes);
    alive_ancestor.reserve(kMaxNodes);

    depth.push_back(0);
    ancestors.push_back(vector<int>(max_steps + 1, 0));
    alive_ancestor.push_back(0);

    int current_node_count = 1;

    for (int query_index = 0; query_index < query_count; ++query_index) {
        char query_type;
        cin >> query_type;

        if (query_type == '+') {
            int parent_index;
            cin >> parent_index;
            HandleAddition(parent_index, current_node_count, max_steps);
        } else if (query_type == '-') {
            int removed_node_index;
            cin >> removed_node_index;
            HandleDeletion(removed_node_index);
        } else if (query_type == '?') {
            int first_query_node;
            int second_query_node;
            cin >> first_query_node >> second_query_node;
            HandleQuery(first_query_node, second_query_node, max_steps);
        }
    }

    return 0;
}
