#ifndef BSMP_DATA_READER_HPP
#define BSMP_DATA_READER_HPP

#include <filesystem>
#include "core/matrix_formats.hpp"

namespace bsmp {

namespace io {

class DataReader {
   public:
    virtual ~DataReader() = default;
    template <typename MatrixFormat, typename Profiler>
    MatrixFormat::Data read(const std::filesystem::path& filename);
};
template <typename MatrixFormat, typename Profiler>
inline MatrixFormat::Data DataReader::read(
    const std::filesystem::path& filename) {
    return MatrixFormat::Data();
}

template <typename Profiler>
inline core::formats::COOFormat::Data DataReader::read(
    const std::filesystem::path& filename) {
}

}  // namespace io
}  // namespace bsmp

#endif // BSMP_DATA_READER_HPP
