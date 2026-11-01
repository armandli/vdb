#ifndef EMBEDDER_H
#define EMBEDDER_H

#include <span>

#include <vdb/types.h>

namespace vdb {

// Text -> vector. Collections record the model_id they were created with and
// refuse vectors from a different model, so embedding spaces never mix.
//
// Planned implementations (TODO.md M10/M12): HttpEmbedder for any
// OpenAI-compatible /v1/embeddings endpoint (Ollama, LM Studio, llama.cpp
// server) and LlamaEmbedder running a GGUF model in-process via llama.cpp.
struct Embedder {
  virtual ~Embedder() = default;

  virtual strv model_id() const = 0;
  virtual uint32_t dim() const = 0;

  // One vector of dim() floats per input text, same order.
  virtual Result<vec<vec<float>>> embed(s::span<const str> texts) = 0;
};

}  // namespace vdb

#endif  // EMBEDDER_H
