#include <iostream>
#include <queue>
#include <vector>

using namespace std;

struct HeapElement {
    int value;
    int sequence_index;
    int element_index;
    bool operator>(const HeapElement& other) const {
        return value > other.value;
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int sequence_count;
    int sequence_length;
    cin >> sequence_count >> sequence_length;

    vector<vector<int>> sequences(sequence_count, vector<int>(sequence_length));
    for (int sequence_index = 0; sequence_index < sequence_count;
         ++sequence_index) {
        for (int element_index = 0; element_index < sequence_length;
             ++element_index) {
            cin >> sequences[sequence_index][element_index];
        }
    }

    priority_queue<HeapElement, vector<HeapElement>, greater<HeapElement>>
        min_heap;
    for (int sequence_index = 0; sequence_index < sequence_count;
         ++sequence_index) {
        min_heap.push({sequences[sequence_index][0], sequence_index, 0});
    }

    vector<int> merged_result;
    merged_result.reserve(sequence_count * sequence_length);

    while (!min_heap.empty()) {
        HeapElement current = min_heap.top();
        min_heap.pop();
        merged_result.push_back(current.value);

        int next_element_index = current.element_index + 1;
        if (next_element_index < sequence_length) {
            min_heap.push(
                {sequences[current.sequence_index][next_element_index],
                 current.sequence_index, next_element_index});
        }
    }

    for (int i = 0; i < static_cast<int>(merged_result.size()); ++i) {
        if (i > 0) {
            cout << " ";
        }
        cout << merged_result[i];
    }
    cout << "\n";
    return 0;
}