#include <algorithm>
#include <iostream>
#include <vector>

const int kInfinity = 1e9 + 7;

struct Coin {
    int position;
    int deadline;
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    int coin_count;
    std::cin >> coin_count;
    std::vector<Coin> coins(coin_count);
    for (int coin_index = 0; coin_index < coin_count; ++coin_index) {
        std::cin >> coins[coin_index].position >> coins[coin_index].deadline;
    }
    std::sort(coins.begin(), coins.end(),
              [](const Coin& first, const Coin& second) {
                  return first.position < second.position;
              });
    std::vector<std::vector<std::vector<int>>> dp(
        coin_count, std::vector<std::vector<int>>(
                        coin_count, std::vector<int>(2, kInfinity)));
    for (int coin_index = 0; coin_index < coin_count; ++coin_index) {
        if (0 <= coins[coin_index].deadline) {
            dp[coin_index][coin_index][0] = dp[coin_index][coin_index][1] = 0;
        }
    }
    for (int length = 1; length < coin_count; ++length) {
        for (int left = 0; left + length < coin_count; ++left) {
            int right = left + length;
            int time = dp[left + 1][right][0] +
                       abs(coins[left + 1].position - coins[left].position);
            if (time <= coins[left].deadline) {
                dp[left][right][0] = std::min(dp[left][right][0], time);
            }
            time = dp[left + 1][right][1] +
                   abs(coins[right].position - coins[left].position);
            if (time <= coins[left].deadline) {
                dp[left][right][0] = std::min(dp[left][right][0], time);
            }
            time = dp[left][right - 1][0] +
                   abs(coins[left].position - coins[right].position);
            if (time <= coins[right].deadline) {
                dp[left][right][1] = std::min(dp[left][right][1], time);
            }
            time = dp[left][right - 1][1] +
                   abs(coins[right - 1].position - coins[right].position);
            if (time <= coins[right].deadline) {
                dp[left][right][1] = std::min(dp[left][right][1], time);
            }
        }
    }
    int result = std::min(dp[0][coin_count - 1][0], dp[0][coin_count - 1][1]);
    if (result == kInfinity) {
        std::cout << "No solution\n";
    } else {
        std::cout << result << '\n';
    }
    return 0;
}
