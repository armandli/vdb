#ifndef DATABASE_H
#define DATABASE_H

#include <vdb/collection.h>
#include <vdb/wal.h>

namespace vdb {

enum struct Durability : uint8_t {
  None,      // no WAL; data is persisted only by flush() / close()
  Buffered,  // WAL written per call, fsync left to the OS / flush()
  Sync,      // WAL fsynced before every mutating call returns
};

struct DatabaseOptions {
  bool create_if_missing = true;
  Durability durability = Durability::Sync;
  // Run Collection::optimize() on a low-priority background thread. Off for
  // embedded use ("nothing runs between your calls"); on for `vdb serve`.
  bool background_optimizer = false;
  WalOptions wal{};
};

// A directory holding collections:
//
//   <dir>/LOCK                     flock(): one process (one Database) at a time
//   <dir>/MANIFEST                 collection list, atomically replaced
//   <dir>/<collection>/config.json
//   <dir>/<collection>/wal/        WAL files
//   <dir>/<collection>/segments/<id>/{vectors,ids,payload,graph,...}
//
// Collection names match [A-Za-z0-9_-]{1,64}.
struct Database {
  static Result<Database> open(const sf::path& dir,
                               DatabaseOptions options = {});
  // Nothing touches disk; flush() is a no-op.
  static Result<Database> open_in_memory(DatabaseOptions options = {});

  ~Database();
  Database(Database&&) noexcept;
  Database& operator=(Database&&) noexcept;

  Result<sptr<Collection>> create_collection(strv name,
                                             CollectionConfig config);
  Result<sptr<Collection>> get_collection(strv name) const;
  // Fails with InvalidArgument when the existing collection's dim, metric
  // or embedding_model differ from `config`.
  Result<sptr<Collection>> get_or_create_collection(strv name,
                                                    CollectionConfig config);
  // Handles to a dropped collection fail every call with NotFound.
  Status drop_collection(strv name);
  vec<str> list_collections() const;

  Status flush();
  // Flushes, stops background work and releases the directory lock.
  // Idempotent; also run by the destructor.
  Status close();

  const sf::path& path() const;

private:
  Database();
  struct Impl;
  uptr<Impl> mImpl;
};

}  // namespace vdb

#endif  // DATABASE_H
