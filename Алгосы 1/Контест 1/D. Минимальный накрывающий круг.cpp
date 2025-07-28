#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

struct Point {
    double x, y;
};

bool canCover(const vector<Point>& points, int k, double r) {
    vector<pair<double, double>> intervals;

    for (const auto& p : points) {
        if (r * r < p.y * p.y) continue; // точка вне круга при любом c

        double diff = sqrt(r * r - p.y * p.y);
        intervals.emplace_back(p.x - diff, p.x + diff);
    }

    if ((int)intervals.size() < k) return false;

    // Собираем события для "максимального пересечения интервалов"
    vector<pair<double, int>> events;
    for (auto& iv : intervals) {
        events.emplace_back(iv.first, +1);
        events.emplace_back(iv.second, -1);
    }

    sort(events.begin(), events.end());

    int current = 0;
    int maxCover = 0;
    for (auto& e : events) {
        current += e.second;
        if (current > maxCover) maxCover = current;
    }

    return maxCover >= k;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, k;
    cin >> n >> k;

    vector<Point> points(n);
    for (int i = 0; i < n; i++) {
        cin >> points[i].x >> points[i].y;
    }

    double left = 0.0;
    double right = 2000.0; // достаточно большой радиус
    for (int iter = 0; iter < 60; ++iter) { // ~60 итераций для хорошей точности
        double mid = (left + right) / 2;
        if (canCover(points, k, mid)) {
            right = mid;
        } else {
            left = mid;
        }
    }

    cout << fixed << setprecision(6) << right << "\n";

    return 0;
}