#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using std::cin;
using std::cout;
using std::string;
using std::tie;
using std::tuple;
using std::vector;

struct TreapNode {
    char value;
    int priority;
    int subtree_size;
    TreapNode* left;
    TreapNode* right;

    explicit TreapNode(char character)
        : value(character),
          priority(rand()),
          subtree_size(1),
          left(nullptr),
          right(nullptr) {}
};

int GetSize(TreapNode* node) {
    return node != nullptr ? node->subtree_size : 0;
}

void UpdateSize(TreapNode* node) {
    if (node != nullptr) {
        node->subtree_size = 1 + GetSize(node->left) + GetSize(node->right);
    }
}

void Split(TreapNode* current, int count, TreapNode** left_tree,
           TreapNode** right_tree) {
    if (current == nullptr) {
        *left_tree = nullptr;
        *right_tree = nullptr;
        return;
    }
    if (GetSize(current->left) >= count) {
        Split(current->left, count, left_tree, &current->left);
        *right_tree = current;
    } else {
        Split(current->right, count - GetSize(current->left) - 1,
              &current->right, right_tree);
        *left_tree = current;
    }
    UpdateSize(current);
}

TreapNode* Merge(TreapNode* left_tree, TreapNode* right_tree) {
    if (left_tree == nullptr) {
        return right_tree;
    }
    if (right_tree == nullptr) {
        return left_tree;
    }
    if (left_tree->priority < right_tree->priority) {
        left_tree->right = Merge(left_tree->right, right_tree);
        UpdateSize(left_tree);
        return left_tree;
    }
    right_tree->left = Merge(left_tree, right_tree->left);
    UpdateSize(right_tree);
    return right_tree;
}

void TraverseInOrder(TreapNode* node, string* output) {
    if (node == nullptr) {
        return;
    }
    TraverseInOrder(node->left, output);
    output->push_back(node->value);
    TraverseInOrder(node->right, output);
}

TreapNode* BuildInitialTreap(const string& encrypted) {
    TreapNode* root = nullptr;
    for (char character : encrypted) {
        TreapNode* node = new TreapNode(character);
        root = Merge(root, node);
    }
    return root;
}

void ApplyReverseShift(TreapNode** root, int left_index, int right_index,
                       int shift) {
    int segment_length = right_index - left_index + 1;
    TreapNode* left = nullptr;
    TreapNode* middle_with_right = nullptr;
    TreapNode* middle = nullptr;
    TreapNode* right = nullptr;
    TreapNode* first_part = nullptr;
    TreapNode* second_part = nullptr;

    Split(*root, left_index - 1, &left, &middle_with_right);
    Split(middle_with_right, segment_length, &middle, &right);
    Split(middle, shift, &first_part, &second_part);
    TreapNode* rotated_middle = Merge(second_part, first_part);
    *root = Merge(Merge(left, rotated_middle), right);
}

void FreeTreap(TreapNode* node) {
    if (node == nullptr) {
        return;
    }
    FreeTreap(node->left);
    FreeTreap(node->right);
    delete node;
}

int main() {
    std::ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string encrypted;
    cin >> encrypted;

    int shift_count;
    cin >> shift_count;

    vector<tuple<int, int, int>> shifts;
    shifts.reserve(shift_count);

    for (int index = 0; index < shift_count; ++index) {
        int start_index;
        int end_index;
        int shift_amount;
        cin >> start_index >> end_index >> shift_amount;
        shifts.emplace_back(start_index, end_index, shift_amount);
    }

    TreapNode* root = BuildInitialTreap(encrypted);

    for (int index = shift_count - 1; index >= 0; --index) {
        int left_index;
        int right_index;
        int shift;
        tie(left_index, right_index, shift) = shifts[index];
        ApplyReverseShift(&root, left_index, right_index, shift);
    }

    string original;
    original.reserve(encrypted.size());
    TraverseInOrder(root, &original);
    cout << original << '\n';

    FreeTreap(root);
    return 0;
}
