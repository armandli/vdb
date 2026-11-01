#include <cstdio>

#include <CLI/CLI.hpp>

#include <vdb/version.h>

// Entry point for the `vdb` tool. The subcommand surface (create, list,
// import, search, get, delete, optimize, stats, serve, bench) is specified
// in TODO.md, milestone M11.
int main(int argc, char** argv) {
  CLI::App app{"vdb: local vector database for RAG and agent memory"};
  app.set_version_flag("--version", vdb::VERSION);
  app.require_subcommand(0, 1);

  CLI11_PARSE(app, argc, argv);

  if (app.get_subcommands().empty()) {
    std::fputs(app.help().c_str(), stdout);
  }
  return 0;
}
