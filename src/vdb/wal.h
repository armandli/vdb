#ifndef WAL_H
#define WAL_H

#include <cstddef>
#include <span>

#include <vdb/types.h>

namespace vdb {

struct WalOptions {
  // Roll to a new file once the current one exceeds this size.
  size_t segment_bytes = 32ull << 20;
  // fsync after every append. Off: callers batch with sync() (group commit).
  bool sync_on_append = false;
};

// Append-only write-ahead log of opaque records in a directory of files
// named wal-<first op, 20 digits>.log.
//
// Record framing: u32 length, u32 crc32c(op ++ body), u64 op, body.
//
// Recovery contract: on open, the last file is scanned; the first record
// with a short read or bad checksum and everything after it is treated as
// a torn tail and truncated away. Corruption in any earlier (closed) file
// is reported as ErrorCode::Corruption, never silently skipped.
//
// The file name carries the first op, so numbering survives reopen even
// when truncate_before() removed every record: the log then starts a new
// empty file named after the next op.
struct Wal {
  using Visitor = s::function<Status(OpNum op, s::span<const s::byte> body)>;

  static Result<Wal> open(const sf::path& dir, WalOptions options = {});
  ~Wal();
  Wal(Wal&&) noexcept;
  Wal& operator=(Wal&&) noexcept;

  // Appends one record and returns its op number. Ops start at 1 in an
  // empty log and are strictly increasing across reopen and truncation.
  Result<OpNum> append(s::span<const s::byte> body);

  // Durably persists every record appended so far.
  Status sync();

  // Calls `visit` for each record with op >= from, in order. Stops and
  // returns the first non-ok Status the visitor returns.
  Status replay(OpNum from, const Visitor& visit) const;

  // Drops whole files whose records are all < op. Records >= op remain
  // replayable (a file straddling `op` is kept).
  Status truncate_before(OpNum op);

  // Highest op appended (0 when the log has never held a record).
  OpNum last_op() const;
  // Lowest op still replayable (last_op() + 1 when empty).
  OpNum first_op() const;

private:
  Wal();
  struct Impl;
  uptr<Impl> mImpl;
};

}  // namespace vdb

#endif  // WAL_H
