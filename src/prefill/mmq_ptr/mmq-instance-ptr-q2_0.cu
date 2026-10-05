// FORK (Rank-1, Issue 09): ptr-path instance, Q2_0 down (see
// mmq-instance-ptr-iq3_xxs.cu for the shape; one TU per type).
#include "mmq.cuh"

namespace strata {
namespace mmq_ptr {
DECL_MMQ_CASE(GGML_TYPE_Q2_0);
}  // namespace mmq_ptr
}  // namespace strata
