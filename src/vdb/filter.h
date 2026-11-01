#ifndef FILTER_H
#define FILTER_H

#include <vdb/payload.h>

namespace vdb {

enum struct FilterOp : uint8_t {
  And,
  Or,
  Not,
  Eq,
  Ne,
  Gt,
  Gte,
  Lt,
  Lte,
  In,
  Nin,
  Exists,
};

// Filter AST. Leaf ops use `field` + `value` (Eq..Lte, Exists with a Bool)
// or `field` + `values` (In, Nin). And/Or/Not use `children`.
struct Filter {
  FilterOp op;
  str field;
  Value value;
  vec<Value> values;
  vec<Filter> children;
};

// Mongo-style JSON filter grammar:
//
//   filter   := { clause, ... }                  implicit AND of clauses
//   clause   := "$and": [filter, ...] | "$or": [filter, ...]
//             | "$not": filter
//             | field: scalar                    shorthand for $eq
//             | field: { opexpr, ... }           implicit AND of ops
//   opexpr   := "$eq"|"$ne"|"$gt"|"$gte"|"$lt"|"$lte": scalar
//             | "$in"|"$nin": [scalar, ...]
//             | "$exists": bool
//   scalar   := string | number | bool
//
// An empty object {} matches everything (And with no children).
// Unknown $operators, empty $and/$or arrays and nesting deeper than 32
// levels are ParseError.
Result<Filter> parse_filter(strv json);

// Evaluation semantics (shared by every evaluator, indexed or not):
//  - a missing field fails every leaf except Ne, Nin and Exists:false
//  - Int and Double compare numerically; other cross-type compares fail
//  - list fields: Eq/In match when ANY element matches; Ne/Nin match when
//    NO element matches; range ops match when any element is in range
//  - string range ops compare lexicographically by bytes
bool matches(const Filter& f, const Payload& p);

// Convenience constructors for building filters in code.
Filter f_and(vec<Filter> children);
Filter f_or(vec<Filter> children);
Filter f_not(Filter child);
Filter f_cmp(FilterOp op, strv field, Value v);
Filter f_in(strv field, vec<Value> vs);
Filter f_exists(strv field, bool exists = true);

}  // namespace vdb

#endif  // FILTER_H
