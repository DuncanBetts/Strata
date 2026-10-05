// FORK (Rank-1, Issue 09), checkpoint 1: proves the vendored slice
// (src/prefill/mmq_ptr/mmq.cuh, llama.cpp pin 3cf03257, MIT) compiles and
// links beside the legacy MMQ path. Instantiates ONE pack type; ptr_list is
// NOT wired yet, so nothing in the engine references these symbols.
#include "mmq.cuh"

namespace strata {
namespace mmq_ptr {
DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS);
}  // namespace mmq_ptr
}  // namespace strata
