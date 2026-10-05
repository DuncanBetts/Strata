// ptr-list dispatch TU. Includes ONLY the vendored
// namespaced slice ("mmq_ptr/mmq.cuh") — never the legacy "mmq.cuh", whose
// file-scope defs would collide (it is compiled separately in moe_mmq.cu).
// One ptr-path template instance per covered type (see mmq-instance-ptr-*.cu);
// ptr_list with any other type is a caller bug and aborts.
#include "strata/prefill/moe_mmq.hpp"

#include "common.cuh"
#include "mmq_ptr/mmq.cuh"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {
namespace {

void ck_ptr(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq ptr: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void run_ptr(const Product& p, void* backend_ctx, void* d_ptrs, void* stream) {
    auto& ctx = *(ggml_backend_cuda_context*) backend_ctx;
    const cudaStream_t s = (cudaStream_t) stream;
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = ggml_blck_size(t), bpr = p.w_cols / qk;
    const char* host_ptrs[16];
    for (int e = 0; e < p.n; ++e) host_ptrs[e] = (const char*) p.blobs[e] + p.w_off;
    ck_ptr(cudaMemcpyAsync(d_ptrs, host_ptrs, (size_t) p.n * sizeof(void*), cudaMemcpyHostToDevice, s), "ptr list");
    const mmq_ptr::mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, p.dst, nullptr,
                    p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                    p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                    1, 1, 0, 0, 0,
                    p.max_rows, p.max_rows, (const char* const*) d_ptrs, 1};
    switch (t) {
    case GGML_TYPE_IQ3_XXS: mmq_ptr::mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, a, s); break;
    case GGML_TYPE_Q2_0: mmq_ptr::mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, a, s); break;
    case GGML_TYPE_IQ4_NL: mmq_ptr::mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, a, s); break;
    default:
        std::fprintf(stderr, "prefill mmq ptr: type %d has no ptr-path instance\n", (int) t);
        std::exit(1);
    }
    ck_ptr(cudaGetLastError(), "ptr mul_mat_q");
}

}  // namespace strata::prefill::mmq
