#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

const int kMod = 123456789;

vector<vector<long long>> memoization_matrix;
vector<int> key_values;

long long CountBinarySearchTrees(int left_index, int right_index) {
    if (left_index > right_index) {
        return 1;
    }
    if (memoization_matrix[left_index][right_index] != -1) {
        return memoization_matrix[left_index][right_index];
    }
    long long result = 0;
    for (int root_index = left_index; root_index <= right_index; ++root_index) {
        if (root_index > left_index &&
            key_values[root_index] == key_values[root_index - 1]) {
            continue;
        }
        long long left_count =
            CountBinarySearchTrees(left_index, root_index - 1);
        long long right_count =
            CountBinarySearchTrees(root_index + 1, right_index);
        result = (result + left_count * right_count) % kMod;
    }
    memoization_matrix[left_index][right_index] = result;
    return result;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int key_count;
    cin >> key_count;
    key_values.resize(key_count);
    for (int key_index = 0; key_index < key_count; ++key_index) {
        cin >> key_values[key_index];
    }
    sort(key_values.begin(), key_values.end());
    memoization_matrix.assign(key_count, vector<long long>(key_count, -1));
    cout << CountBinarySearchTrees(0, key_count - 1) << endl;
    return 0;
}