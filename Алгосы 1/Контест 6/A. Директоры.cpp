#include <iostream>
#include <vector>

class DisjointSetUnion {
public:
    explicit DisjointSetUnion(int num_employees) : parent_(num_employees + 1) {
        for (int employee_id = 1; employee_id <= num_employees; ++employee_id) {
            parent_[employee_id] = employee_id;
        }
    }

    int FindDirector(int employee_id) {
        if (parent_[employee_id] != employee_id) {
            parent_[employee_id] = FindDirector(parent_[employee_id]);
        }
        return parent_[employee_id];
    }

    // Назначить boss_id начальником subordinate_id
    bool SetBoss(int boss_id, int subordinate_id) {
        if (parent_[subordinate_id] != subordinate_id) {
            return false;  // Уже есть начальник
        }
        int boss_director = FindDirector(boss_id);
        int subordinate_director = FindDirector(subordinate_id);
        if (boss_director == subordinate_director) {
            return false;  // Цикл
        }
        parent_[subordinate_id] = boss_id;
        return true;
    }

private:
    std::vector<int> parent_;
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int num_employees;
    int num_queries;
    std::cin >> num_employees >> num_queries;
    DisjointSetUnion dsu(num_employees);

    for (int query_index = 0; query_index < num_queries; ++query_index) {
        int first_value;
        if (std::cin >> first_value) {
            if (std::cin.peek() == '\n' || std::cin.eof()) {
                // Запрос директора
                std::cout << dsu.FindDirector(first_value) << '\n';
            } else {
                int second_value;
                std::cin >> second_value;
                std::cout << dsu.SetBoss(first_value, second_value) << '\n';
            }
        }
    }
    return 0;
}