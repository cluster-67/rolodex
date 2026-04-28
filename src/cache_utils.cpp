#include "rolodex/cache_utils.hpp"

#include "rolodex/dataset.hpp"

#include <cerrno>
#include <cstdint>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace {

uint64_t fnv1a64(const std::string &s) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : s) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool stat_path(const std::string &path, struct stat &out) {
    return stat(path.c_str(), &out) == 0;
}

} // namespace

namespace cache_utils {

uint64_t compute_dataset_signature(const Dataset *dataset) {
    if (dataset == nullptr) {
        return 0;
    }

    // Use stable, deterministic components:
    // - dataset filename string
    // - file size + mtime (seconds)
    //
    // This avoids false-positive cache hits across different datasets with same (n,d).
    const std::string &fname = dataset->filename();
    struct stat st;
    uint64_t sig = fnv1a64(fname);
    if (stat_path(fname, st)) {
        sig ^= static_cast<uint64_t>(st.st_size) + 0x9e3779b97f4a7c15ULL + (sig << 6) + (sig >> 2);
        sig ^= static_cast<uint64_t>(st.st_mtime) + 0x9e3779b97f4a7c15ULL + (sig << 6) + (sig >> 2);
    }
    return sig;
}

bool ensure_dir_recursive(const std::string &path, int mode) {
    if (path.empty()) {
        return false;
    }

    struct stat st;
    if (stat_path(path, st)) {
        return S_ISDIR(st.st_mode);
    }

    std::size_t sep = path.find_last_of('/');
    if (sep != std::string::npos) {
        const std::string parent = path.substr(0, sep);
        if (!parent.empty() && !ensure_dir_recursive(parent, mode)) {
            return false;
        }
    }

    if (mkdir(path.c_str(), static_cast<mode_t>(mode)) != 0) {
        if (errno == EEXIST && stat_path(path, st)) {
            return S_ISDIR(st.st_mode);
        }
        return false;
    }
    return true;
}

std::string to_hex(uint64_t value) {
    std::ostringstream out;
    out << std::hex << value;
    return out.str();
}

} // namespace cache_utils
