#ifndef PAYLOAD_H
#define PAYLOAD_H

#include <map>
#include <span>
#include <variant>

#include <vdb/types.h>

namespace vdb {

using StrList = vec<str>;
using IntList = vec<int64_t>;

// A payload value. Nested JSON objects are flattened into dotted keys
// ("user.age"), so a Payload is always one level deep.
using Value =
    s::variant<s::monostate, bool, int64_t, double, str, StrList, IntList>;

enum struct ValueType : uint8_t {
  Null,
  Bool,
  Int,
  Double,
  String,
  StrList,
  IntList,
};

ValueType type_of(const Value& v);

// Ordered so serialization is deterministic.
using Payload = s::map<str, Value, s::less<>>;

// JSON object -> Payload.
//  - integers that fit int64 become Int, other numbers Double
//  - arrays must be homogeneous strings or integers (empty -> IntList{})
//  - nested objects flatten with '.'; arrays of objects are rejected
//  - top level must be an object
Result<Payload> parse_payload(strv json);

// Payload -> compact JSON object with dotted keys left flat.
str to_json(const Payload& p);

// Compact binary encoding used inside segment files.
vec<uint8_t> encode_payload(const Payload& p);
Result<Payload> decode_payload(s::span<const uint8_t> bytes);

}  // namespace vdb

#endif  // PAYLOAD_H
