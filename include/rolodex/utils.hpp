#pragma once

#include <algorithm>
#include <queue>
#include <string>
#include <utility>
#include <vector>

class Dataset;

namespace utils {
namespace cache {

bool ensure_dir_recursive(const std::string &path, int mode);
std::string dataset_basename(const Dataset *dataset);

} // namespace cache

namespace path {

std::string scratch_dir();
std::string dataset_path(const std::string &dataset_filename);

} // namespace path

namespace knn {

class TopKAccumulator {
  public:
    using Entry = std::pair<float, std::size_t>;

    TopKAccumulator() = default;
    explicit TopKAccumulator(std::size_t k) : k_(k) {}

    void reset(std::size_t k) {
        k_ = k;
        heap_ = Heap();
    }

    std::size_t size() const {
        return heap_.size();
    }

    std::size_t capacity() const {
        return k_;
    }

    bool would_accept(float dist, std::size_t idx) const {
        if (k_ == 0) {
            return false;
        }
        if (heap_.size() < k_) {
            return true;
        }
        return is_better(Entry(dist, idx), heap_.top());
    }

    void push_accepted(float dist, std::size_t idx) {
        if (k_ == 0) {
            return;
        }
        Entry candidate(dist, idx);
        if (heap_.size() < k_) {
            heap_.push(candidate);
            return;
        }
        heap_.pop();
        heap_.push(candidate);
    }

    void maybe_push(float dist, std::size_t idx) {
        if (!would_accept(dist, idx)) {
            return;
        }
        push_accepted(dist, idx);
    }

    std::vector<Entry> extract_sorted() const {
        std::vector<Entry> entries = entries_unsorted();
        std::sort(entries.begin(), entries.end(), is_better);
        return entries;
    }

    std::vector<Entry> entries_unsorted() const {
        Heap copy = heap_;
        std::vector<Entry> out;
        out.reserve(copy.size());
        while (!copy.empty()) {
            out.push_back(copy.top());
            copy.pop();
        }
        return out;
    }

  private:
    struct Worse {
        bool operator()(const Entry &a, const Entry &b) const {
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second < b.second;
        }
    };

    static bool is_better(const Entry &a, const Entry &b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    }

    using Heap = std::priority_queue<Entry, std::vector<Entry>, Worse>;
    std::size_t k_ = 0;
    Heap heap_;
};

} // namespace knn
} // namespace utils
