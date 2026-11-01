# Shortcuts around CMake. The real build lives in CMakeLists.txt.
#
#   make                 configure (if needed) + build, Release
#   make debug           rebuild as Debug
#   make test            build + run every enabled unit test
#   make test T=hnsw     run only build/test_hnsw
#   make asan|tsan|ubsan sanitizer builds in their own build directory
#
# Variables: BUILD_DIR, BUILD_TYPE, JOBS, CMAKE_ARGS, T, GTEST_FILTER.

BUILD_DIR  ?= build
BUILD_TYPE ?= Release
JOBS       ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
CMAKE_ARGS ?=
GENERATOR  := $(if $(shell command -v ninja 2>/dev/null),-G Ninja,)

CONFIGURE = cmake -S . -B $(BUILD_DIR) $(GENERATOR) \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_ARGS)

.PHONY: all build configure reconfigure release debug relwithdebinfo \
	asan tsan ubsan test test-list mcp clean distclean help

all: build

$(BUILD_DIR)/CMakeCache.txt:
	$(CONFIGURE)

configure: $(BUILD_DIR)/CMakeCache.txt

reconfigure:
	$(CONFIGURE)

build: configure
	cmake --build $(BUILD_DIR) -j $(JOBS)
	@ln -sf $(BUILD_DIR)/compile_commands.json compile_commands.json

release:
	$(MAKE) reconfigure build BUILD_TYPE=Release

debug:
	$(MAKE) reconfigure build BUILD_TYPE=Debug

relwithdebinfo:
	$(MAKE) reconfigure build BUILD_TYPE=RelWithDebInfo

asan:
	$(MAKE) reconfigure build BUILD_DIR=build-asan BUILD_TYPE=Debug \
		CMAKE_ARGS="-DVDB_SANITIZE=address $(CMAKE_ARGS)"

tsan:
	$(MAKE) reconfigure build BUILD_DIR=build-tsan BUILD_TYPE=RelWithDebInfo \
		CMAKE_ARGS="-DVDB_SANITIZE=thread $(CMAKE_ARGS)"

ubsan:
	$(MAKE) reconfigure build BUILD_DIR=build-ubsan BUILD_TYPE=Debug \
		CMAKE_ARGS="-DVDB_SANITIZE=undefined $(CMAKE_ARGS)"

ifdef T
test: build
	./$(BUILD_DIR)/test_$(T) $(if $(GTEST_FILTER),--gtest_filter=$(GTEST_FILTER),)
else
test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure -j $(JOBS)
endif

test-list: configure
	@found=$$(ls $(BUILD_DIR)/test_* 2>/dev/null); \
	if [ -n "$$found" ]; then for t in $$found; do basename $$t; done; \
	else echo "no tests built yet (implement src/vdb/*.cpp first)"; fi

# Run the MCP server over stdio against ./data.
mcp: build
	./$(BUILD_DIR)/vdb serve --path ./data --transport stdio

clean:
	@if [ -d $(BUILD_DIR) ]; then cmake --build $(BUILD_DIR) --target clean; fi

distclean:
	rm -rf build build-asan build-tsan build-ubsan compile_commands.json

help:
	@sed -n '1,12p' Makefile
