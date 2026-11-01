#ifndef MCP_TOOLS_H
#define MCP_TOOLS_H

#include <vdb/database.h>
#include <vdb/embedder.h>

namespace vdb {

struct ToolSpec {
  str name;
  str description;
  str input_schema;  // JSON Schema object, as JSON text
};

// Transport-free MCP tool layer: JSON arguments in, JSON result out. The
// `vdb serve` binary registers these with cxxmcp for stdio / Streamable
// HTTP; tests drive them directly.
//
// Collection tools (vectors may be replaced by text when an embedder is set):
//   vdb_list_collections  {}
//   vdb_create_collection {name, dim?, metric?, indexes?, text_index?}
//   vdb_upsert            {collection, records: [{id, vector?, document?,
//                          payload?}]}                    -> {"upserted": n}
//   vdb_search            {collection, vector?, query?, k?, filter?}
//                                                        -> {"hits": [...]}
//   vdb_get               {collection, ids: [...]}       -> {"records": [...]}
//   vdb_delete            {collection, ids?, filter?}    -> {"deleted": n}
//
// Agent-memory tools (need an embedder; one shared collection, partitioned
// by the tenant field "namespace"):
//   memory_remember {namespace, content, kind?, importance?, tags?}
//                                                        -> {"id": "..."}
//   memory_recall   {namespace, query, k?, kinds?}       -> {"hits": [...]}
//   memory_forget   {namespace, id}                      -> {"deleted": n}
//
// A memory is stored with payload {namespace, kind (default "episodic"),
// importance (default 5), ts (unix seconds), tags?} and an id generated
// by the tool. memory_recall ranks with MemoryScoring (similarity +
// recency + importance) inside the namespace, optionally restricted to
// `kinds`.
//
// Every hit object has "id", "score", "document" and "payload".
struct McpTools {
  static constexpr const char* MEMORY_COLLECTION = "agent_memory";

  explicit McpTools(Database& db, sptr<Embedder> embedder = nullptr);

  vec<ToolSpec> list_tools() const;

  // Unknown tool, schema violations and engine errors come back as Error;
  // the server maps them to a tool result with isError: true.
  Result<str> call(strv tool, strv arguments_json);

private:
  Database& mDb;
  sptr<Embedder> mEmbedder;
};

}  // namespace vdb

#endif  // MCP_TOOLS_H
