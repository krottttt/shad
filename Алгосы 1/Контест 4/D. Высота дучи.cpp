#include <iomanip>
#include <iostream>
#include <vector>

namespace {
constexpr int kMaxSize = 100;
constexpr int kPrecision = 6;

double probability_cache[kMaxSize + 1][kMaxSize + 1];
bool computed[kMaxSize + 1][kMaxSize + 1];

double ComputeProbability(int node_count, int target_height) {
    if (node_count == 0) {
        return (target_height == -1) ? 1.0 : 0.0;
    }
    if (target_height < 0 || target_height > node_count - 1) {
        return 0.0;
    }
    if (computed[node_count][target_height]) {
        return probability_cache[node_count][target_height];
    }

    double total_probability = 0.0;

    for (int left_count = 0; left_count < node_count; ++left_count) {
        int right_count = node_count - 1 - left_count;
        double inverse_total = 1.0 / node_count;

        for (int right_height = -1; right_height < target_height - 1;
             ++right_height) {
            total_probability +=
                ComputeProbability(left_count, target_height - 1) *
                ComputeProbability(right_count, right_height) * inverse_total;
        }

        for (int left_height = -1; left_height < target_height - 1;
             ++left_height) {
            total_probability +=
                ComputeProbability(left_count, left_height) *
                ComputeProbability(right_count, target_height - 1) *
                inverse_total;
        }

        total_probability +=
            ComputeProbability(left_count, target_height - 1) *
            ComputeProbability(right_count, target_height - 1) * inverse_total;
    }

    computed[node_count][target_height] = true;
    probability_cache[node_count][target_height] = total_probability;
    return total_probability;
}
}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int node_count;
    long long target_height;
    std::cin >> node_count >> target_height;

    double result =
        (target_height <= node_count - 1
             ? ComputeProbability(node_count, static_cast<int>(target_height))
             : 0.0);

    std::cout << std::fixed << std::setprecision(kPrecision) << result << "\n";
    return 0;
}