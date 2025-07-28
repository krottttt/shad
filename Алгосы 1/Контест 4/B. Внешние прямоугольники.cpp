#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

struct Event {
    int x_start;
    int x_end;
    int y_coord;
    int rect_id;
    bool is_start;
};

bool CompareEvents(const Event& left_event, const Event& right_event) {
    if (left_event.y_coord != right_event.y_coord) {
        return left_event.y_coord < right_event.y_coord;
    }
    if (left_event.is_start != right_event.is_start) {
        return left_event.is_start;
    }
    return left_event.x_start < right_event.x_start;
}

struct TreapNode {
    int x_start;
    int x_end;
    int rect_id;
    int priority;
    TreapNode* left;
    TreapNode* right;
    TreapNode(int x_start_value, int x_end_value, int rect_id_value,
              int priority_value)
        : x_start(x_start_value),
          x_end(x_end_value),
          rect_id(rect_id_value),
          priority(priority_value),
          left(nullptr),
          right(nullptr) {}
};

const int kRngSeed = 123456;
std::mt19937 rng(kRngSeed);

void Split(TreapNode* node, int x_split, TreapNode*& left, TreapNode*& right) {
    if (node == nullptr) {
        left = nullptr;
        right = nullptr;
    } else if (node->x_start < x_split) {
        Split(node->right, x_split, node->right, right);
        left = node;
    } else {
        Split(node->left, x_split, left, node->left);
        right = node;
    }
}

TreapNode* Merge(TreapNode* left, TreapNode* right) {
    if (left == nullptr || right == nullptr) {
        if (left != nullptr) {
            return left;
        }
        return right;
    }
    if (left->priority > right->priority) {
        left->right = Merge(left->right, right);
        return left;
    }
    right->left = Merge(left, right->left);
    return right;
}

void Insert(TreapNode*& root, TreapNode* node) {
    if (root == nullptr) {
        root = node;
    } else if (node->priority > root->priority) {
        Split(root, node->x_start, node->left, node->right);
        root = node;
    } else if (node->x_start < root->x_start) {
        Insert(root->left, node);
    } else {
        Insert(root->right, node);
    }
}

void Erase(TreapNode*& root, int x_start_value, int rect_id_value) {
    if (root == nullptr) {
        return;
    }
    if (root->x_start == x_start_value && root->rect_id == rect_id_value) {
        TreapNode* old_node = root;
        root = Merge(root->left, root->right);
        delete old_node;
    } else if (x_start_value < root->x_start) {
        Erase(root->left, x_start_value, rect_id_value);
    } else {
        Erase(root->right, x_start_value, rect_id_value);
    }
}

TreapNode* FindPrev(TreapNode* root, int x_start_value) {
    TreapNode* result = nullptr;
    while (root != nullptr) {
        if (root->x_start < x_start_value) {
            result = root;
            root = root->right;
        } else {
            root = root->left;
        }
    }
    return result;
}

bool IsContained(const TreapNode* outer, int x_start_value, int x_end_value) {
    return (outer != nullptr) && (outer->x_start <= x_start_value) &&
           (x_end_value <= outer->x_end);
}

void DeleteTreap(TreapNode* root) {
    if (root == nullptr) {
        return;
    }
    DeleteTreap(root->left);
    DeleteTreap(root->right);
    delete root;
}

int ProcessEvents(const vector<Event>& events, int rectangle_count) {
    TreapNode* treap = nullptr;
    vector<bool> is_external(rectangle_count, false);

    for (const Event& event : events) {
        if (event.is_start) {
            TreapNode* prev = FindPrev(treap, event.x_start);
            bool contained = false;
            if (IsContained(prev, event.x_start, event.x_end)) {
                contained = true;
            }
            if (!contained) {
                int priority = rng();
                TreapNode* node = new TreapNode(event.x_start, event.x_end,
                                                event.rect_id, priority);
                Insert(treap, node);
            }
        } else {
            TreapNode* node = treap;
            while (node != nullptr) {
                if (node->x_start == event.x_start &&
                    node->rect_id == event.rect_id) {
                    is_external[event.rect_id] = true;
                    break;
                }
                if (event.x_start < node->x_start) {
                    node = node->left;
                } else {
                    node = node->right;
                }
            }
            Erase(treap, event.x_start, event.rect_id);
        }
    }

    int result =
        static_cast<int>(count(is_external.begin(), is_external.end(), true));
    DeleteTreap(treap);
    return result;
}

void ReadRectangles(int count, vector<Event>& events, istream& in) {
    for (int index = 0; index < count; ++index) {
        int x_start = 0;
        int y_start = 0;
        int x_end = 0;
        int y_end = 0;

        in >> x_start >> y_start >> x_end >> y_end;

        if (x_start > x_end) {
            swap(x_start, x_end);
        }

        if (y_start > y_end) {
            swap(y_start, y_end);
        }

        events.push_back({x_start, x_end, y_start, index, true});
        events.push_back({x_start, x_end, y_end, index, false});
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    ifstream fin("input.txt");
    int rectangle_count = 0;
    fin >> rectangle_count;

    vector<Event> events;
    events.reserve(rectangle_count * 2);

    ReadRectangles(rectangle_count, events, fin);

    sort(events.begin(), events.end(), CompareEvents);

    int result = ProcessEvents(events, rectangle_count);

    cout << result << "\n";

    return 0;
}