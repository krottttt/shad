#include <iostream>
#include <limits>
#include <vector>

struct Node {
    int key;
    Node* left;
    Node* right;
    explicit Node(int key_value)
        : key(key_value), left(nullptr), right(nullptr) {}
};

int current_index = 0;

Node* BuildBST(const std::vector<int>& preorder, int min_val, int max_val) {
    if (current_index == static_cast<int>(preorder.size())) {
        return nullptr;
    }
    int val = preorder[current_index];
    if (val < min_val || val > max_val) {
        return nullptr;
    }

    Node* root = new Node(val);
    ++current_index;

    root->left = BuildBST(preorder, min_val, val - 1);
    root->right = BuildBST(preorder, val, max_val);

    return root;
}

void PostorderTraversal(Node* root, std::vector<int>& result) {
    if (root == nullptr) {
        return;
    }
    PostorderTraversal(root->left, result);
    PostorderTraversal(root->right, result);
    result.push_back(root->key);
}

void InorderTraversal(Node* root, std::vector<int>& result) {
    if (root == nullptr) {
        return;
    }
    InorderTraversal(root->left, result);
    result.push_back(root->key);
    InorderTraversal(root->right, result);
}

void DeleteTree(Node* root) {
    if (root == nullptr) {
        return;
    }
    DeleteTree(root->left);
    DeleteTree(root->right);
    delete root;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int node_count;
    std::cin >> node_count;
    std::vector<int> preorder(node_count);
    for (int i = 0; i < node_count; ++i) {
        std::cin >> preorder[i];
    }

    current_index = 0;
    Node* root = BuildBST(preorder, std::numeric_limits<int>::min(),
                          std::numeric_limits<int>::max());

    std::vector<int> postorder_result;
    std::vector<int> inorder_result;
    PostorderTraversal(root, postorder_result);
    InorderTraversal(root, inorder_result);

    for (int i = 0; i < static_cast<int>(postorder_result.size()); ++i) {
        if (i > 0) {
            std::cout << " ";
        }
        std::cout << postorder_result[i];
    }
    std::cout << "\n";

    for (int i = 0; i < static_cast<int>(inorder_result.size()); ++i) {
        if (i > 0) {
            std::cout << " ";
        }
        std::cout << inorder_result[i];
    }
    std::cout << "\n";

    DeleteTree(root);

    return 0;
}