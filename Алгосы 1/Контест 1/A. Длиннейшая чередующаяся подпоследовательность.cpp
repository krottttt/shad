#include <iostream>
#include <vector>
#include <algorithm>

struct State {
    int length = 1;
    std::vector<int> indices; // храним индексы для лекс сравнения

    State() = default;

    State(int idx) : length(1), indices(1, idx) {}

    // Лексикографическое сравнение по индексам
    bool IsLexSmallerThan(const State& other) const {
        for (size_t i = 0; i < std::min(indices.size(), other.indices.size()); ++i) {
            if (indices[i] < other.indices[i]) return true;
            if (indices[i] > other.indices[i]) return false;
        }
        return indices.size() < other.indices.size();
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    std::cin >> n;
    std::vector<int> a(n);
    for (int& x : a) std::cin >> x;

    // dp[i][0] - LCAS с последним ростом, заканчивающаяся в i
    // dp[i][1] - LCAS с последним спадом, заканчивающаяся в i
    std::vector<std::vector<State>> dp(n, std::vector<State>(2));

    for (int i = 0; i < n; ++i) {
        dp[i][0] = State(i);
        dp[i][1] = State(i);
    }

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (a[j] < a[i]) {
                // рост, обновляем dp[i][0] из dp[j][1]
                int candidate_len = dp[j][1].length + 1;
                if (candidate_len > dp[i][0].length) {
                    dp[i][0].length = candidate_len;
                    dp[i][0].indices = dp[j][1].indices;
                    dp[i][0].indices.push_back(i);
                } else if (candidate_len == dp[i][0].length) {
                    // лекс сравнение
                    std::vector<int> candidate_indices = dp[j][1].indices;
                    candidate_indices.push_back(i);
                    State candidate_state;
                    candidate_state.length = candidate_len;
                    candidate_state.indices = candidate_indices;

                    if (candidate_state.IsLexSmallerThan(dp[i][0])) {
                        dp[i][0] = candidate_state;
                    }
                }
            } else if (a[j] > a[i]) {
                // спад, обновляем dp[i][1] из dp[j][0]
                int candidate_len = dp[j][0].length + 1;
                if (candidate_len > dp[i][1].length) {
                    dp[i][1].length = candidate_len;
                    dp[i][1].indices = dp[j][0].indices;
                    dp[i][1].indices.push_back(i);
                } else if (candidate_len == dp[i][1].length) {
                    std::vector<int> candidate_indices = dp[j][0].indices;
                    candidate_indices.push_back(i);
                    State candidate_state;
                    candidate_state.length = candidate_len;
                    candidate_state.indices = candidate_indices;

                    if (candidate_state.IsLexSmallerThan(dp[i][1])) {
                        dp[i][1] = candidate_state;
                    }
                }
            }
        }
    }

    // Поиск максимального результата
    State answer;
    answer.length = 0;

    for (int i = 0; i < n; ++i) {
        for (int s = 0; s < 2; ++s) {
            if (dp[i][s].length > answer.length) {
                answer = dp[i][s];
            } else if (dp[i][s].length == answer.length) {
                if (dp[i][s].IsLexSmallerThan(answer)) {
                    answer = dp[i][s];
                }
            }
        }
    }

    // Вывод результата
    for (size_t i = 0; i < answer.indices.size(); ++i) {
        if (i > 0) std::cout << " ";
        std::cout << a[answer.indices[i]];
    }
    std::cout << "\n";

    return 0;
}