#ifndef PAYLOAD_INDEX_H
#define PAYLOAD_INDEX_H

#include <roaring/roaring.hh>

#include <vdb/filter.h>

namespace vdb {

enum struct FieldType : uint8_t {
  Keyword,  // String / StrList: hash map value -> bitmap
  Integer,  // Int / IntList: sorted (value, id) runs for range queries
  Float,    // Int or Double, stored as double
  Bool,
};

struct FieldIndexSpec {
  str name;
  FieldType type;
  // Marks the namespace / tenant field. The planner treats an Eq on it as
  // a partition selector, and segments keep its bitmaps hot.
  bool is_tenant = false;

  bool operator==(const FieldIndexSpec&) const = default;
};

// Estimated number of matching points as an interval. Always
// min <= exp <= max <= total, and the true count lies in [min, max].
struct Cardinality {
  size_t min;
  size_t exp;
  size_t max;

  bool operator==(const Cardinality&) const = default;
};

// Per-segment secondary indexes over declared payload fields. Values whose
// type does not match the field type are ignored by the index (the point
// is still found by a payload scan).
struct PayloadIndex {
  explicit PayloadIndex(vec<FieldIndexSpec> specs);
  ~PayloadIndex();
  PayloadIndex(PayloadIndex&&) noexcept;
  PayloadIndex& operator=(PayloadIndex&&) noexcept;

  const vec<FieldIndexSpec>& specs() const;
  bool is_indexed(strv field) const;

  void add(InternalId id, const Payload& p);
  void remove(InternalId id, const Payload& p);

  // Exact matching ids when every leaf of `f` touches an indexed field;
  // nullopt otherwise (caller falls back to scanning payloads). `universe`
  // is the set of live ids, needed to evaluate Not / Ne / Nin / Exists.
  opt<roaring::Roaring> evaluate(
      const Filter& f,
      const roaring::Roaring& universe) const;

  // Cheap cardinality estimate over `total` live points.
  //  - indexed leaf: exact count (min == exp == max)
  //  - unindexed leaf: {0, total / 2, total}
  //  - And: min = max(0, sum(min) - (n-1)*total), exp = total * prod(p),
  //         max = min(max_i)
  //  - Or:  min = max(min_i), exp = total * (1 - prod(1 - p)),
  //         max = min(total, sum(max))
  //  - Not: {total - max, total - exp, total - min}
  // where p = exp / total of each child.
  Cardinality estimate(const Filter& f, size_t total) const;

private:
  struct Impl;
  uptr<Impl> mImpl;
};

}  // namespace vdb

#endif  // PAYLOAD_INDEX_H
