#include <algorithm>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

using std::cin;
using std::cout;
using std::endl;
using std::gcd;
using std::ios_base;
using std::make_tuple;
using std::sort;
using std::swap;
using std::tuple;
using std::vector;

tuple<int, int, int> NormalizeTriangle(int side_a, int side_b, int side_c) {
    int gcd_ab = gcd(side_a, side_b);
    int gcd_abc = gcd(gcd_ab, side_c);

    side_a /= gcd_abc;
    side_b /= gcd_abc;
    side_c /= gcd_abc;

    if (side_a > side_b) {
        swap(side_a, side_b);
    }
    if (side_b > side_c) {
        swap(side_b, side_c);
    }
    if (side_a > side_b) {
        swap(side_a, side_b);
    }

    return make_tuple(side_a, side_b, side_c);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int triangle_count;
    cin >> triangle_count;

    vector<tuple<int, int, int>> normalized_triangles;
    normalized_triangles.reserve(triangle_count);

    for (int i = 0; i < triangle_count; ++i) {
        int side_a;
        int side_b;
        int side_c;
        cin >> side_a >> side_b >> side_c;

        auto normalized = NormalizeTriangle(side_a, side_b, side_c);
        normalized_triangles.push_back(normalized);
    }

    sort(normalized_triangles.begin(), normalized_triangles.end());

    int unique_count = 0;
    if (!normalized_triangles.empty()) {
        unique_count = 1;
        for (size_t i = 1; i < normalized_triangles.size(); ++i) {
            if (normalized_triangles[i] != normalized_triangles[i - 1]) {
                ++unique_count;
            }
        }
    }
    cout << unique_count << endl;
    return 0;
}