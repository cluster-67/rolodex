#include "rolodex/cache_utils.hpp"

#include "rolodex/dataset.hpp"

#include <cerrno>
#include <string>
#include <sys/stat.h>

namespace {

bool stat_path(const std::string &path, struct stat &out) {
    return stat(path.c_str(), &out) == 0;
}

} // namespace

namespace cache_utils {

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

std::string dataset_basename(const Dataset *dataset) {
    if (dataset == nullptr) {
        return std::string();
    }
    const std::string &full = dataset->filename();
    const std::size_t slash = full.find_last_of("/\\");
    if (slash == std::string::npos) {
        return full;
    }
    return full.substr(slash + 1);
}

} // namespace cache_utils
