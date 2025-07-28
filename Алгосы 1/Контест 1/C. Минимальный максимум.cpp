#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, l;
    cin >> n >> m >> l;

    vector<vector<int>> A(n, vector<int>(l));
    vector<vector<int>> B(m, vector<int>(l));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < l; ++j)
            cin >> A[i][j];

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < l; ++j)
            cin >> B[i][j];

    int q;
    cin >> q;

    while (q--) {
        int i, j;
        cin >> i >> j;
        --i, --j;

        int left = 0, right = l - 1;
        int k0 = l;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (A[i][mid] >= B[j][mid]) {
                k0 = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        vector<int> candidates;
        if (k0 < l) {
            candidates.push_back(k0);
        }
        if (k0 > 0) {
            candidates.push_back(k0 - 1);
        }

        int best_k = -1;
        int best_val = numeric_limits<int>::max();
        for (int k : candidates) {
            int current_val = max(A[i][k], B[j][k]);
            if (current_val < best_val) {
                best_val = current_val;
                best_k = k;
            }
        }

        cout << best_k + 1 << '\n';
    }

    return 0;
}
