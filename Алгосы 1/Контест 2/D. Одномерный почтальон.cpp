#include <cstdint>
#include <iostream>
#include <vector>

using namespace std;

const int kRandShift = 8;
const int kRadix = 8;
const int kBucketCount = 1 << kRadix;
const int kPasses = 32 / kRadix;

unsigned int current_value;

unsigned int GenerateNextRandom24(unsigned int random_multiplier,
                                  unsigned int random_increment) {
    current_value = current_value * random_multiplier + random_increment;
    return current_value >> kRandShift;
}

unsigned int GenerateNextRandom32(unsigned int random_multiplier,
                                  unsigned int random_increment) {
    unsigned int first_part =
        GenerateNextRandom24(random_multiplier, random_increment);
    unsigned int second_part =
        GenerateNextRandom24(random_multiplier, random_increment);
    return (first_part << kRandShift) ^ second_part;
}

void RadixSort(vector<unsigned int>& values) {
    int value_count = values.size();
    vector<unsigned int> buffer(value_count);
    for (int pass = 0; pass < kPasses; ++pass) {
        int shift = pass * kRadix;
        vector<int> count(kBucketCount, 0);
        for (int i = 0; i < value_count; ++i) {
            int bucket = (values[i] >> shift) & (kBucketCount - 1);
            ++count[bucket];
        }
        for (int i = 1; i < kBucketCount; ++i) {
            count[i] += count[i - 1];
        }
        for (int i = value_count - 1; i >= 0; --i) {
            int bucket = (values[i] >> shift) & (kBucketCount - 1);
            buffer[--count[bucket]] = values[i];
        }
        values.swap(buffer);
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int house_count;
    cin >> house_count;
    unsigned int random_multiplier;
    unsigned int random_increment;
    cin >> random_multiplier >> random_increment;

    vector<unsigned int> house_positions(house_count);
    current_value = 0;
    for (int house_index = 0; house_index < house_count; ++house_index) {
        house_positions[house_index] =
            GenerateNextRandom32(random_multiplier, random_increment);
    }

    RadixSort(house_positions);

    int median_index = house_count / 2;
    unsigned int optimal_position = house_positions[median_index];

    long long total_distance = 0;
    for (int house_index = 0; house_index < house_count; ++house_index) {
        long long difference =
            static_cast<long long>(house_positions[house_index]) -
            static_cast<long long>(optimal_position);
        if (difference < 0) {
            difference = -difference;
        }
        total_distance += difference;
    }

    cout << total_distance << "\n";
    return 0;
}