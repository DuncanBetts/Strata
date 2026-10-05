// ptr-path instance, IQ3_XXS gate/up (see mmq.cuh for the slice; one TU per
// type so only this TU resolves "mmq.cuh" to the wrapped copy).
#include "mmq.cuh"

namespace strata {
namespace mmq_ptr {
DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS);
}  // namespace mmq_ptr
}  // namespace strata
