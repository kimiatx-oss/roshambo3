// Format-3 codec: the bounded read native callers use to fetch block bytes
// (FORMAT.md section 17 item 12, Appendix B S7). POSIX pread; no mmap, so a
// file truncated after it was opened is a kShortRead, never a SIGBUS. Kept out
// of the standalone fuzzing core (r3g_codec.cpp has no I/O).
#include "r3g_codec.hpp"

#include <cerrno>
#include <cstdint>
#include <limits>

#include <sys/types.h>
#include <unistd.h>

namespace r3 {
namespace r3g {

Status read_exact(int fd, std::uint64_t offset, std::uint8_t* out, std::size_t length) noexcept {
    if (out == nullptr && length != 0) return Status::kBadArgument;
    if (fd < 0) return Status::kBadArgument;
    std::uint64_t end = 0;
    constexpr std::uint64_t kMaxOff = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
    if (__builtin_add_overflow(offset, static_cast<std::uint64_t>(length), &end) || end > kMaxOff)
        return Status::kSizeOverflow;
    std::size_t done = 0;
    while (done < length) {
        std::size_t want = length - done;
        // A single pread is capped (Linux transfers at most ~2 GiB per call).
        if (want > (std::size_t(1) << 30)) want = std::size_t(1) << 30;
        const ssize_t got = ::pread(fd, out + done, want, static_cast<off_t>(offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            return Status::kIoError;
        }
        if (got == 0) return Status::kShortRead;
        done += static_cast<std::size_t>(got);
    }
    return Status::kOk;
}

}  // namespace r3g
}  // namespace r3
