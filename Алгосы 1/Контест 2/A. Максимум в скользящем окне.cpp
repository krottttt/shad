#include <deque>
#include <iostream>
#include <vector>

std::vector<int> SlidingWindowMaximum(const std::vector<int>& array,
                                      const std::vector<char>& moves) {
    int left = 0;
    int right = 0;
    std::deque<int> max_indices;
    std::vector<int> result;

    max_indices.push_back(0);

    for (char move : moves) {
        if (move == 'L') {
            left++;
            while (!max_indices.empty() && max_indices.front() < left) {
                max_indices.pop_front();
            }
        } else {
            right++;
            while (!max_indices.empty() &&
                   array[max_indices.back()] <= array[right]) {
                max_indices.pop_back();
            }
            max_indices.push_back(right);
        }

        result.push_back(array[max_indices.front()]);
    }

    return result;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int array_size;
    std::cin >> array_size;

    std::vector<int> array(array_size);
    for (int i = 0; i < array_size; ++i) {
        std::cin >> array[i];
    }

    int moves_count;
    std::cin >> moves_count;

    std::vector<char> moves(moves_count);
    for (int i = 0; i < moves_count; ++i) {
        std::cin >> moves[i];
    }

    std::vector<int> result = SlidingWindowMaximum(array, moves);

    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i];
        if (i < result.size() - 1) {
            std::cout << " ";
        }
    }
    std::cout << std::endl;

    return 0;
}
