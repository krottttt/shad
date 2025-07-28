#ifndef FIXED_SET_H
#define FIXED_SET_H

#include <cstdint>
#include <optional>
#include <vector>

class FixedSet {
public:
    FixedSet() : capacity_(0), size_(0) {}

    void Initialize(const std::vector<int>& numbers) {
        size_ = static_cast<int>(numbers.size());
        capacity_ = 1;
        while (capacity_ < size_ * 2) {
            capacity_ <<= 1;  // capacity is power of two >= 2*size
        }

        table_.assign(capacity_, std::nullopt);

        for (int number : numbers) {
            Insert(number);
        }
    }

    bool Contains(int number) const {
        int hash1 = Hash1(number);
        int hash2 = Hash2(number);
        int index = hash1;
        int probes = 0;

        while (probes < capacity_) {
            if (!table_[index].has_value()) {
                return false;
            }
            if (table_[index].value() == number) {
                return true;
            }
            index = (index + hash2) & (capacity_ - 1);
            probes++;
        }
        return false;
    }

private:
    int capacity_;
    int size_;
    std::vector<std::optional<int>> table_;

    static constexpr int kShiftBits = 16;
    static constexpr uint32_t kHashMul = 0x45d9f3b;
    static constexpr uint32_t kHashXor1 = 0x9e3779b9;
    static constexpr uint32_t kHashXor2 = 0x7f4a7c15;

    static uint32_t HashUInt(uint32_t value) {
        value = ((value >> kShiftBits) ^ value) * kHashMul;
        value = ((value >> kShiftBits) ^ value) * kHashMul;
        value = (value >> kShiftBits) ^ value;
        return value;
    }

    int Hash1(int number) const {
        uint32_t uint_num = static_cast<uint32_t>(number) ^ kHashXor1;
        return HashUInt(uint_num) & (capacity_ - 1);
    }

    int Hash2(int number) const {
        uint32_t uint_num = static_cast<uint32_t>(number) ^ kHashXor2;
        int step = HashUInt(uint_num) & (capacity_ - 2);
        return step + 1;
    }

    void Insert(int number) {
        int hash1 = Hash1(number);
        int hash2 = Hash2(number);
        int index = hash1;
        while (table_[index].has_value()) {
            index = (index + hash2) & (capacity_ - 1);
        }
        table_[index] = number;
    }
};
