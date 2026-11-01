#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <expected>
#include <functional>

#include <util/aliases.h>

namespace vdb {

// Dense per-segment row number. Indexes, bitmaps and vector stores are all
// addressed by it; users never see it.
using InternalId = uint32_t;

// WAL sequence number. Doubles as the version of every point it touched, so
// replaying an already-applied operation is a no-op.
using OpNum = uint64_t;

inline constexpr InternalId INVALID_ID = UINT32_MAX;

enum struct Metric : uint8_t {
  L2,            // squared euclidean distance, no sqrt
  InnerProduct,  // distance = -dot(a, b)
  Cosine,        // distance = 1 - cos(a, b); vectors normalized on insert
};

enum struct ErrorCode : uint8_t {
  InvalidArgument,
  NotFound,
  AlreadyExists,
  DimensionMismatch,
  ParseError,
  Corruption,
  VersionMismatch,
  IOError,
  Unsupported,
  Internal,
};

struct Error {
  ErrorCode code;
  str message;
};

template <typename T>
using Result = s::expected<T, Error>;
using Status = Result<void>;

inline s::unexpected<Error> make_error(ErrorCode code, str message) {
  return s::unexpected<Error>{Error{code, s::move(message)}};
}

// Internal search result. `distance` follows the metric: smaller is closer.
struct Hit {
  InternalId id;
  float distance;

  bool operator==(const Hit&) const = default;
};

// Predicate pushed into an index search: return false to exclude a point.
// An empty function accepts everything.
using IdFilter = s::function<bool(InternalId)>;

}  // namespace vdb

#endif  // TYPES_H
