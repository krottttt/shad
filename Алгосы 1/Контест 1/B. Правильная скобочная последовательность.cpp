#include <iostream>
#include <string>
#include <stack>
#include <unordered_map>

bool IsOpeningBracket(char ch) {
    return ch == '(' || ch == '[' || ch == '{';
}

bool IsMatchingPair(char open, char close) {
    static const std::unordered_map<char, char> pairs = {
        {'(', ')'},
        {'[', ']'},
        {'{', '}'}
    };
    auto it = pairs.find(open);
    return it != pairs.end() && it->second == close;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string input;
    std::cin >> input;

    std::stack<char> bracket_stack;
    int max_prefix = 0; // максимальная длина префикса, который корректен или может быть продолжен

    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        char ch = input[i];

        if (IsOpeningBracket(ch)) {
            bracket_stack.push(ch);
        } else {
            // Закрывающая скобка
            if (bracket_stack.empty() || !IsMatchingPair(bracket_stack.top(), ch)) {
                // Нарушение — префикс с этим символом уже нельзя продолжить
                std::cout << max_prefix << "\n";
                return 0;
            }
            bracket_stack.pop();
        }

        // После обработки символа:
        // Если стек пуст — это правильный префикс
        // Если стек не пуст — это префикс, который можно продолжить
        max_prefix = i + 1;
    }

    // После прохода по всей строке
    if (bracket_stack.empty()) {
        // Вся строка корректна
        std::cout << "CORRECT\n";
    } else {
        // Не вся, но префикс может быть продолжен
        std::cout << max_prefix << "\n";
    }

    return 0;
}