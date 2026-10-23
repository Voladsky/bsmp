#include <filesystem>
#include "core/matrix_formats.hpp"

namespace bsmp {

namespace io {

class DataReader {
   public:
    virtual ~DataReader() = default;
    template <typename MatrixFormat, typename Profiler>
    MatrixFormat::Buffers read(const std::filesystem::path& filename);
};
template <typename MatrixFormat, typename Profiler>
inline MatrixFormat::Buffers DataReader::read(
    const std::filesystem::path& filename) {
    return MatrixFormat::Buffers();
}

template <typename Profiler>
inline core::formats::COOFormat::Buffers DataReader::read(
    const std::filesystem::path& filename) {
}

}  // namespace io
}  // namespace bsmp
