#ifndef CRC32C_H
#define CRC32C_H

#include <cstddef>
#include <cstdint>
#include <span>

#include <util/aliases.h>

namespace vdb {

// CRC-32C (Castagnoli, reflected polynomial 0x82F63B78) — the checksum of
// every WAL record and file trailer. Use the hardware instruction when
// available (SSE4.2 crc32 on x86, __crc32c* on ARMv8).
//
// Chaining: crc32c(b, crc32c(a)) == crc32c(a ++ b).
uint32_t crc32c(s::span<const s::byte> data, uint32_t crc = 0);
uint32_t crc32c(const void* data, size_t size, uint32_t crc = 0);

}  // namespace vdb

#endif  // CRC32C_H
