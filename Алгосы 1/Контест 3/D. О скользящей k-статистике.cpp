#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::string;
using std::vector;

const int kMaxSize = 100005;
int binary_indexed_tree[2 * kMaxSize];
int elements_count;
int operations_count;
int k_threshold;
vector<int> elements;
vector<int> compressed_values;

void UpdateBinaryIndexedTree(int index, int delta) {
    while (index < 2 * kMaxSize) {
        binary_indexed_tree[index] += delta;
        index += (index & -index);
    }
}

int QueryPrefixSum(int index) {
    int result = 0;
    while (index > 0) {
        result += binary_indexed_tree[index];
        index -= (index & -index);
    }
    return result;
}

int FindKthElement(int k_value) {
    int left_bound = 1;
    int right_bound = compressed_values.size();
    int answer = -1;

    while (left_bound <= right_bound) {
        int middle = (left_bound + right_bound) / 2;
        if (QueryPrefixSum(middle) >= k_value) {
            answer = middle;
            right_bound = middle - 1;
        } else {
            left_bound = middle + 1;
        }
    }
    return answer == -1 ? -1 : compressed_values[answer - 1];
}

int GetCompressedIndex(int value) {
    return std::lower_bound(compressed_values.begin(), compressed_values.end(),
                            value) -
           compressed_values.begin() + 1;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> elements_count >> operations_count >> k_threshold;
    elements.resize(elements_count);
    compressed_values.resize(elements_count);

    for (int index = 0; index < elements_count; ++index) {
        cin >> elements[index];
        compressed_values[index] = elements[index];
    }

    std::sort(compressed_values.begin(), compressed_values.end());
    compressed_values.erase(
        std::unique(compressed_values.begin(), compressed_values.end()),
        compressed_values.end());

    string operations;
    cin >> operations;

    int left_window = 0;
    int right_window = 0;
    UpdateBinaryIndexedTree(GetCompressedIndex(elements[0]), 1);

    for (char operation : operations) {
        if (operation == 'R') {
            ++right_window;
            if (right_window < elements_count) {
                UpdateBinaryIndexedTree(
                    GetCompressedIndex(elements[right_window]), 1);
            }
        } else if (operation == 'L') {
            UpdateBinaryIndexedTree(GetCompressedIndex(elements[left_window]),
                                    -1);
            ++left_window;
        }

        int window_size = right_window - left_window + 1;
        if (window_size < k_threshold) {
            cout << -1 << '\n';
        } else {
            cout << FindKthElement(k_threshold) << '\n';
        }
    }

    return 0;
}
