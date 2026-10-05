// prefill_mmq_ptr_test - stride-vs-pointer parity for
// the vendored MMQ slice. Same weights/activations/routing through (a) the
// legacy stride path (contiguous group buffer, ptr_list unset) and (b) the
// ptr path (scattered per-expert blobs + w_off, no gather) must give
// BITWISE-identical dst. Gu (IQ3_XXS, 1280 rows) and down (Q2_0, 2560 rows)
// with ragged bounds (5,0,9,3: zero-row expert, non-tile counts) + TAIL
// memset, exactly the engine's launch sequence. Exit 77 without sm_80+.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

constexpr int N = 2560, FF = 640, GUN = 1280;
constexpr size_t TAIL = 4096;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
static int g_mode = 0; // 0 = both (parity), 1 = stride only, 2 = ptr only (sanitizer attribution)

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n = 0) { if (n) ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(o.p) { o.p = nullptr; }
};

int run_case(ggml_type gt, ggml_type dt, int64_t wrows_gu, int64_t wrows_d, const char* tag) {
    std::mt19937 rng(1234);
    const size_t gu_row = ggml_row_size(gt, N);
    const size_t d_row = ggml_row_size(dt, FF);
    const size_t up_off = gu_row * FF;
    const size_t down_off = 2 * up_off;
    const size_t blob_bytes = down_off + d_row * N;
    const size_t gub = 2 * up_off, gdb = d_row * N;
    const int nex = 4, rows_per[4] = {5, 0, 9, 3};
    int32_t bounds_h[5] = {0};
    for (int e = 0; e < nex; ++e) bounds_h[e + 1] = bounds_h[e] + rows_per[e];
    const int nr = bounds_h[nex], maxr = 9;

    // scattered blobs (separate allocations, 16B-aligned, unrelated addresses)
    std::vector<std::vector<uint8_t>> host_blob(nex, std::vector<uint8_t>(blob_bytes));
    for (int e = 0; e < nex; ++e)
        for (size_t i = 0; i < blob_bytes; ++i) host_blob[e][i] = (uint8_t) rng();
    std::vector<Dev> blob;
    blob.reserve(nex);
    for (int e = 0; e < nex; ++e) blob.emplace_back(blob_bytes + 16);
    for (int e = 0; e < nex; ++e)
        ck(cudaMemcpy((uint8_t*) blob[e].p, host_blob[e].data(), blob_bytes, cudaMemcpyHostToDevice), "blob H2D");
    // contiguous group copies (what gather_native would produce)
    Dev grp_gu(nex * gub + TAIL), grp_d(nex * gdb + TAIL);
    for (int e = 0; e < nex; ++e) {
        ck(cudaMemcpy((uint8_t*) grp_gu.p + e * gub, host_blob[e].data(), gub, cudaMemcpyHostToDevice), "gu copy");
        ck(cudaMemcpy((uint8_t*) grp_d.p + e * gdb, host_blob[e].data() + down_off, gdb, cudaMemcpyHostToDevice),
           "d copy");
    }
    ck(cudaMemset((uint8_t*) grp_gu.p + nex * gub, 0, TAIL), "tail");
    ck(cudaMemset((uint8_t*) grp_d.p + nex * gdb, 0, TAIL), "tail");

    // activations + routing on device
    std::vector<float> fx(nr * N), fh(nr * FF);
    Dev dxq(mmq::q8_bytes(nr, N)), dhq(mmq::q8_bytes(nr, FF));
    for (auto& v : fh) v = (float) (rng() % 2000 - 1000) / 1000.0f;
    {
        Dev hfx(fx.size() * 4), hfh(fh.size() * 4);
        ck(cudaMemcpy(hfx.p, fx.data(), fx.size() * 4, cudaMemcpyHostToDevice), "fx");
        ck(cudaMemcpy(hfh.p, fh.data(), fh.size() * 4, cudaMemcpyHostToDevice), "fh");
        mmq::quantize((const float*) hfx.p, nullptr, dxq.p, (int) gt, N, N, nr, nullptr);
        mmq::quantize((const float*) hfh.p, nullptr, dhq.p, (int) dt, FF, FF, nr, nullptr);
    }
    Dev dbounds(5 * sizeof(int32_t)), dids(nr * sizeof(int32_t));
    std::vector<int32_t> ids(nr);
    for (int i = 0; i < nr; ++i) ids[i] = i;
    ck(cudaMemcpy(dbounds.p, bounds_h, sizeof(bounds_h), cudaMemcpyHostToDevice), "bounds");
    ck(cudaMemcpy(dids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");

    mmq::Context ctx;
    auto run = [&](bool ptr, int64_t wrows, int wcols, size_t ebytes, const void* xq, float* dst, int64_t ld,
                   const void* const* blobs, size_t woff, int type) {
        if ((g_mode == 1 && ptr) || (g_mode == 2 && !ptr)) return; // sanitizer attribution runs
        mmq::Product p;
        p.w = ptr ? blob[0].p : (wrows == wrows_gu ? grp_gu.p : grp_d.p); // x unused on ptr path (any valid base)
        p.type = type;
        p.w_rows = wrows;
        p.w_cols = wcols;
        p.expert_bytes = ebytes;
        p.n = nex;
        p.xq = xq;
        p.bounds = (const int32_t*) dbounds.p;
        p.ids = (const int32_t*) dids.p;
        p.total_rows = nr;
        p.max_rows = maxr;
        p.dst = dst;
        p.ld_dst = ld;
        p.ptr_list = ptr;
        p.w_off = woff;
        if (ptr) for (int e = 0; e < nex; ++e) p.blobs[e] = blobs[e];
        ctx.run(p, nullptr);
    };
    const void* gu_blobs[4] = {blob[0].p, blob[1].p, blob[2].p, blob[3].p};
    Dev dgu_a(nr * GUN * 4), dgu_b(nr * GUN * 4), dd_a(nr * N * 4), dd_b(nr * N * 4);
    run(false, wrows_gu, N, gub, dxq.p, (float*) dgu_a.p, GUN, nullptr, 0, (int) gt);
    run(true, wrows_gu, N, gub, dxq.p, (float*) dgu_b.p, GUN, gu_blobs, 0, (int) gt);
    run(false, wrows_d, FF, gdb, dhq.p, (float*) dd_a.p, N, nullptr, 0, (int) dt);
    run(true, wrows_d, FF, gdb, dhq.p, (float*) dd_b.p, N, gu_blobs, down_off, (int) dt);
    ck(cudaDeviceSynchronize(), "sync");
    int fails = 0;
    auto cmp = [&](Dev& a, Dev& b, size_t nfloats, const char* what) {
        std::vector<float> ha(nfloats), hb(nfloats);
        ck(cudaMemcpy(ha.data(), a.p, nfloats * 4, cudaMemcpyDeviceToHost), "readback");
        ck(cudaMemcpy(hb.data(), b.p, nfloats * 4, cudaMemcpyDeviceToHost), "readback");
        for (size_t i = 0; i < nfloats; ++i)
            if (memcmp(&ha[i], &hb[i], 4)) {
                std::printf("%s %s: first divergence at %zu: stride=%a ptr=%a\n", tag, what, i, ha[i], hb[i]);
                ++fails;
                return;
            }
        std::printf("%s %s: bitwise identical (%zu floats)\n", tag, what, nfloats);
    };
    cmp(dgu_a, dgu_b, (size_t) nr * GUN, "gu");
    cmp(dd_a, dd_b, (size_t) nr * N, "down");
    return fails;
}
// production-dn mirror: n=16, maxr=15, nr=45, relative bounds, down_off blob
// layout, non-default stream (like m.cs). Isolates the production dn failure.
int run_dn16_maxr(int big) {
    std::mt19937 rng(999);
    const ggml_type dt = GGML_TYPE_Q2_0;
    const size_t d_row = ggml_row_size(dt, FF);
    const size_t down_off = 1254400, gdb = d_row * N, blob_bytes = down_off + gdb;
    const int nex = 16;
    int32_t bounds_h[17] = {0};
    for (int e = 0; e < nex; ++e) bounds_h[e + 1] = bounds_h[e] + (big == 2 ? 42 : big ? 3 : (e == 5 ? 15 : 2));
    const int nr = bounds_h[nex];
    int maxr = 0;
    for (int e = 0; e < nex; ++e) maxr = std::max(maxr, bounds_h[e + 1] - bounds_h[e]);
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreate(&cs), "stream");
    std::vector<std::vector<uint8_t>> host_blob(nex, std::vector<uint8_t>(blob_bytes));
    for (int e = 0; e < nex; ++e)
        for (size_t i = 0; i < blob_bytes; ++i) host_blob[e][i] = (uint8_t) (rng() % 16); // sane scales: raw bytes overflow fp16 math
    std::vector<Dev> blob;
    blob.reserve(nex);
    const bool poison = std::getenv("STRATA_DN16_POISON") != nullptr; // TEMP: 0xFF tail proves K-tail over-read reads past blob end
    for (int e = 0; e < nex; ++e) blob.emplace_back(blob_bytes + (poison ? 4096 : 16));
    for (int e = 0; e < nex; ++e)
        ck(cudaMemcpyAsync(blob[e].p, host_blob[e].data(), blob_bytes, cudaMemcpyHostToDevice, cs), "blob");
    if (poison) for (int e = 0; e < nex; ++e) ck(cudaMemsetAsync((uint8_t*) blob[e].p + blob_bytes, 0xFF, 4096, cs), "poison");
    Dev grp(nex * gdb + TAIL);
    for (int e = 0; e < nex; ++e)
        ck(cudaMemcpyAsync((uint8_t*) grp.p + e * gdb, host_blob[e].data() + down_off, gdb, cudaMemcpyHostToDevice, cs), "grp");
    ck(cudaMemsetAsync((uint8_t*) grp.p + nex * gdb, 0, TAIL, cs), "tail");
    std::vector<float> fh(nr * FF);
    for (auto& v : fh) v = (float) (rng() % 2000 - 1000) / 1000.0f;
    Dev hfh(fh.size() * 4), hq(mmq::q8_bytes(nr, FF));
    ck(cudaMemcpyAsync(hfh.p, fh.data(), fh.size() * 4, cudaMemcpyHostToDevice, cs), "fh");
    mmq::quantize((const float*) hfh.p, nullptr, hq.p, (int) dt, FF, FF, nr, cs);
    Dev dbounds(17 * sizeof(int32_t)), dids(nr * sizeof(int32_t));
    std::vector<int32_t> ids(nr);
    for (int i = 0; i < nr; ++i) ids[i] = i;
    ck(cudaMemcpyAsync(dbounds.p, bounds_h, sizeof(bounds_h), cudaMemcpyHostToDevice, cs), "bounds");
    ck(cudaMemcpyAsync(dids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice, cs), "ids");
    mmq::Context ctx;
    const void* blobs[16];
    for (int e = 0; e < nex; ++e) blobs[e] = blob[e].p;
    auto run = [&](bool ptr, float* dst) {
        mmq::Product p;
        p.w = ptr ? blob[0].p : grp.p;
        p.type = (int) dt;
        p.w_rows = N;
        p.w_cols = FF;
        p.expert_bytes = gdb;
        p.n = nex;
        p.xq = hq.p;
        p.bounds = (const int32_t*) dbounds.p;
        p.ids = (const int32_t*) dids.p;
        p.total_rows = nr;
        p.max_rows = maxr;
        p.dst = dst;
        p.ld_dst = N;
        p.ptr_list = ptr;
        p.w_off = down_off;
        if (ptr) for (int e = 0; e < nex; ++e) p.blobs[e] = blobs[e];
        ctx.run(p, cs);
    };
    Dev da(nr * N * 4), db(nr * N * 4);
    run(false, (float*) da.p);
    run(true, (float*) db.p);
    ck(cudaStreamSynchronize(cs), "sync");
    std::vector<float> ha(nr * N), hb(nr * N);
    ck(cudaMemcpy(ha.data(), da.p, ha.size() * 4, cudaMemcpyDeviceToHost), "rb");
    ck(cudaMemcpy(hb.data(), db.p, hb.size() * 4, cudaMemcpyDeviceToHost), "rb");
    int fails = 0;
    for (size_t i = 0; i < ha.size(); ++i)
        if (memcmp(&ha[i], &hb[i], 4)) {
            std::printf("dn16: divergence at %zu: stride=%a ptr=%a\n", i, ha[i], hb[i]);
            fails = 1;
            break;
        }
    if (!fails) std::printf("dn16: bitwise identical (%zu floats)\n", ha.size());
    ck(cudaStreamDestroy(cs), "sdestroy");
    return fails;
}
// production-dn convention: pre-offset per-expert blobs + w_off=0 (as
// prefill.cpp fills dn.blobs), n=10, nr=420, maxr=42, relative bounds.
int run_dn_prod() {
    std::mt19937 rng(4242);
    const ggml_type dt = GGML_TYPE_IQ4_NL;
    const size_t d_row = ggml_row_size(dt, FF);
    const size_t down_off = 1254400, gdb = d_row * N, blob_bytes = down_off + gdb;
    const int nex = 10;
    int32_t bounds_h[11] = {0};
    for (int e = 0; e < nex; ++e) bounds_h[e + 1] = bounds_h[e] + 42;
    const int nr = bounds_h[nex], maxr = 42;
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreate(&cs), "stream");
    std::vector<std::vector<uint8_t>> host_blob(nex, std::vector<uint8_t>(blob_bytes));
    for (int e = 0; e < nex; ++e)
        for (size_t i = 0; i < blob_bytes; ++i) host_blob[e][i] = (uint8_t) rng();
    std::vector<Dev> blob;
    blob.reserve(nex);
    for (int e = 0; e < nex; ++e) blob.emplace_back(blob_bytes + 16);
    for (int e = 0; e < nex; ++e)
        ck(cudaMemcpyAsync(blob[e].p, host_blob[e].data(), blob_bytes, cudaMemcpyHostToDevice, cs), "blob");
    Dev grp(nex * gdb + TAIL);
    for (int e = 0; e < nex; ++e)
        ck(cudaMemcpyAsync((uint8_t*) grp.p + e * gdb, host_blob[e].data() + down_off, gdb, cudaMemcpyHostToDevice, cs), "grp");
    ck(cudaMemsetAsync((uint8_t*) grp.p + nex * gdb, 0, TAIL, cs), "tail");
    std::vector<float> fh(nr * FF);
    for (auto& v : fh) v = (float) (rng() % 2000 - 1000) / 1000.0f;
    Dev hfh(fh.size() * 4), hq(mmq::q8_bytes(nr, FF));
    ck(cudaMemcpyAsync(hfh.p, fh.data(), fh.size() * 4, cudaMemcpyHostToDevice, cs), "fh");
    mmq::quantize((const float*) hfh.p, nullptr, hq.p, (int) dt, FF, FF, nr, cs);
    Dev dbounds(11 * sizeof(int32_t)), dids(nr * sizeof(int32_t));
    std::vector<int32_t> ids(nr);
    for (int i = 0; i < nr; ++i) ids[i] = i;
    ck(cudaMemcpyAsync(dbounds.p, bounds_h, sizeof(bounds_h), cudaMemcpyHostToDevice, cs), "bounds");
    ck(cudaMemcpyAsync(dids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice, cs), "ids");
    mmq::Context ctx;
    const void* blobs[16];
    for (int e = 0; e < nex; ++e) blobs[e] = (const uint8_t*) blob[e].p + down_off; // pre-offset, like prefill
    auto run = [&](bool ptr, float* dst) {
        mmq::Product p;
        p.w = ptr ? blob[0].p : grp.p;
        p.type = (int) dt;
        p.w_rows = N;
        p.w_cols = FF;
        p.expert_bytes = gdb;
        p.n = nex;
        p.xq = hq.p;
        p.bounds = (const int32_t*) dbounds.p;
        p.ids = (const int32_t*) dids.p;
        p.total_rows = nr;
        p.max_rows = maxr;
        p.dst = dst;
        p.ld_dst = N;
        p.ptr_list = ptr;
        p.w_off = 0;
        if (ptr) for (int e = 0; e < nex; ++e) p.blobs[e] = blobs[e];
        ctx.run(p, cs);
    };
    Dev da(nr * N * 4), db(nr * N * 4);
    run(false, (float*) da.p);
    run(true, (float*) db.p);
    ck(cudaStreamSynchronize(cs), "sync");
    std::vector<float> ha(nr * N), hb(nr * N);
    ck(cudaMemcpy(ha.data(), da.p, ha.size() * 4, cudaMemcpyDeviceToHost), "rb");
    ck(cudaMemcpy(hb.data(), db.p, hb.size() * 4, cudaMemcpyDeviceToHost), "rb");
    int fails = 0;
    for (size_t i = 0; i < ha.size(); ++i)
        if (memcmp(&ha[i], &hb[i], 4)) {
            std::printf("dn_prod: divergence at %zu: stride=%a ptr=%a\n", i, ha[i], hb[i]);
            fails = 1;
            break;
        }
    if (!fails) std::printf("dn_prod: bitwise identical (%zu floats)\n", ha.size());
    ck(cudaStreamDestroy(cs), "sdestroy");
    return fails;
}
// production-gu shape: IQ3_XXS, n=10, nr=420, maxr=42, absolute bounds,
// w_off=0 with base blobs (gu lives at blob offset 0). Dsts zeroed so a
// kernel that writes nothing shows as zeros instead of malloc garbage.
int run_gu_prod() {
    std::mt19937 rng(3535);
    const ggml_type gt = GGML_TYPE_IQ3_XXS;
    const size_t gu_row = ggml_row_size(gt, N);
    const size_t up_off = gu_row * FF;
    const size_t gub = 2 * up_off, blob_bytes = gub + 921600;
    const int nex = 10;
    int32_t bounds_h[11] = {0};
    for (int e = 0; e < nex; ++e) bounds_h[e + 1] = bounds_h[e] + 42;
    const int nr = bounds_h[nex], maxr = 42;
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreate(&cs), "stream");
    std::vector<std::vector<uint8_t>> host_blob(nex, std::vector<uint8_t>(blob_bytes));
    for (int e = 0; e < nex; ++e)
        for (size_t i = 0; i < gub; ++i) host_blob[e][i] = (uint8_t) rng();
    std::vector<Dev> blob;
    blob.reserve(nex);
    for (int e = 0; e < nex; ++e) blob.emplace_back(blob_bytes + 16);
    for (int e = 0; e < nex; ++e)
        ck(cudaMemcpyAsync(blob[e].p, host_blob[e].data(), blob_bytes, cudaMemcpyHostToDevice, cs), "blob");
    Dev grp(nex * gub + TAIL);
    for (int e = 0; e < nex; ++e)
        ck(cudaMemcpyAsync((uint8_t*) grp.p + e * gub, host_blob[e].data(), gub, cudaMemcpyHostToDevice, cs), "grp");
    ck(cudaMemsetAsync((uint8_t*) grp.p + nex * gub, 0, TAIL, cs), "tail");
    std::vector<float> fx(nr * N);
    for (auto& v : fx) v = (float) (rng() % 2000 - 1000) / 1000.0f;
    Dev hfx(fx.size() * 4), dxq(mmq::q8_bytes(nr, N));
    ck(cudaMemcpyAsync(hfx.p, fx.data(), fx.size() * 4, cudaMemcpyHostToDevice, cs), "fx");
    mmq::quantize((const float*) hfx.p, nullptr, dxq.p, (int) gt, N, N, nr, cs);
    Dev dbounds(11 * sizeof(int32_t)), dids(nr * sizeof(int32_t));
    std::vector<int32_t> ids(nr);
    for (int i = 0; i < nr; ++i) ids[i] = i;
    ck(cudaMemcpyAsync(dbounds.p, bounds_h, sizeof(bounds_h), cudaMemcpyHostToDevice, cs), "bounds");
    ck(cudaMemcpyAsync(dids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice, cs), "ids");
    mmq::Context ctx;
    const void* blobs[16];
    for (int e = 0; e < nex; ++e) blobs[e] = blob[e].p;
    auto run = [&](bool ptr, float* dst) {
        mmq::Product p;
        p.w = ptr ? blob[0].p : grp.p;
        p.type = (int) gt;
        p.w_rows = GUN;
        p.w_cols = N;
        p.expert_bytes = gub;
        p.n = nex;
        p.xq = dxq.p;
        p.bounds = (const int32_t*) dbounds.p;
        p.ids = (const int32_t*) dids.p;
        p.total_rows = nr;
        p.max_rows = maxr;
        p.dst = dst;
        p.ld_dst = GUN;
        p.ptr_list = ptr;
        p.w_off = 0;
        if (ptr) for (int e = 0; e < nex; ++e) p.blobs[e] = blobs[e];
        ctx.run(p, cs);
    };
    Dev da(nr * GUN * 4), db(nr * GUN * 4);
    ck(cudaMemsetAsync(da.p, 0, nr * GUN * 4, cs), "zero");
    ck(cudaMemsetAsync(db.p, 0, nr * GUN * 4, cs), "zero");
    run(false, (float*) da.p);
    run(true, (float*) db.p);
    ck(cudaStreamSynchronize(cs), "sync");
    std::vector<float> ha(nr * GUN), hb(nr * GUN);
    ck(cudaMemcpy(ha.data(), da.p, ha.size() * 4, cudaMemcpyDeviceToHost), "rb");
    ck(cudaMemcpy(hb.data(), db.p, hb.size() * 4, cudaMemcpyDeviceToHost), "rb");
    size_t nzero_a = 0, nzero_b = 0;
    for (size_t i = 0; i < ha.size(); ++i) { if (ha[i] == 0.0f) ++nzero_a; if (hb[i] == 0.0f) ++nzero_b; }
    int fails = 0;
    for (size_t i = 0; i < ha.size(); ++i)
        if (memcmp(&ha[i], &hb[i], 4)) {
            std::printf("gu_prod: divergence at %zu: stride=%a ptr=%a (zeros stride=%zu/%zu ptr=%zu/%zu)\n", i, ha[i], hb[i], nzero_a, ha.size(), nzero_b, hb.size());
            fails = 1;
            break;
        }
    if (!fails) std::printf("gu_prod: bitwise identical (%zu floats)\n", ha.size());
    ck(cudaStreamDestroy(cs), "sdestroy");
    return fails;
}
// production-shape case: full 16-expert group, 2048 rows (exercises tile
// prefetch over-read past the last expert's rows for sanitizer runs)
int run_big(ggml_type gt, ggml_type dt, int ragged) {
    std::mt19937 rng(777);
    const size_t gu_row = ggml_row_size(gt, N);
    const size_t up_off = gu_row * FF;
    const size_t down_off = 2 * up_off;
    const size_t blob_bytes = down_off + ggml_row_size(dt, FF) * N;
    const size_t gub = 2 * up_off, gdb = blob_bytes - down_off;
    const int nex = 16;
    int32_t bounds_h[17] = {0};
    for (int e = 0; e < nex; ++e) bounds_h[e + 1] = bounds_h[e] + (ragged ? 10 + (e * 37) % 31 : 128);
    const int nr = bounds_h[nex];
    int maxr = 0;
    for (int e = 0; e < nex; ++e) maxr = std::max(maxr, bounds_h[e + 1] - bounds_h[e]);
    std::vector<std::vector<uint8_t>> host_blob(nex, std::vector<uint8_t>(blob_bytes));
    for (int e = 0; e < nex; ++e)
        for (size_t i = 0; i < blob_bytes; ++i) host_blob[e][i] = (uint8_t) rng();
    std::vector<Dev> blob;
    blob.reserve(nex);
    for (int e = 0; e < nex; ++e) blob.emplace_back(blob_bytes + 16);
    for (int e = 0; e < nex; ++e)
        ck(cudaMemcpy(blob[e].p, host_blob[e].data(), blob_bytes, cudaMemcpyHostToDevice), "blob H2D");
    Dev grp_gu(nex * gub + TAIL), grp_d(nex * gdb + TAIL);
    for (int e = 0; e < nex; ++e) {
        ck(cudaMemcpy((uint8_t*) grp_gu.p + e * gub, host_blob[e].data(), gub, cudaMemcpyHostToDevice), "gu");
        ck(cudaMemcpy((uint8_t*) grp_d.p + e * gdb, host_blob[e].data() + down_off, gdb, cudaMemcpyHostToDevice), "d");
    }
    ck(cudaMemset((uint8_t*) grp_gu.p + nex * gub, 0, TAIL), "tail");
    ck(cudaMemset((uint8_t*) grp_d.p + nex * gdb, 0, TAIL), "tail");
    std::vector<float> fx(nr * N);
    for (auto& v : fx) v = (float) (rng() % 2000 - 1000) / 1000.0f;
    Dev hfx(fx.size() * 4), dxq(mmq::q8_bytes(nr, N));
    ck(cudaMemcpy(hfx.p, fx.data(), fx.size() * 4, cudaMemcpyHostToDevice), "fx");
    mmq::quantize((const float*) hfx.p, nullptr, dxq.p, (int) gt, N, N, nr, nullptr);
    Dev dbounds(17 * sizeof(int32_t)), dids(nr * sizeof(int32_t));
    std::vector<int32_t> ids(nr);
    for (int i = 0; i < nr; ++i) ids[i] = i;
    ck(cudaMemcpy(dbounds.p, bounds_h, sizeof(bounds_h), cudaMemcpyHostToDevice), "bounds");
    ck(cudaMemcpy(dids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    mmq::Context ctx;
    const void* blobs[16];
    for (int e = 0; e < nex; ++e) blobs[e] = blob[e].p;
    auto run = [&](bool ptr, float* dst) {
        if ((g_mode == 1 && ptr) || (g_mode == 2 && !ptr)) return; // sanitizer attribution runs
        mmq::Product p;
        p.w = ptr ? blob[0].p : grp_gu.p;
        p.type = (int) gt;
        p.w_rows = GUN;
        p.w_cols = N;
        p.expert_bytes = gub;
        p.n = nex;
        p.xq = dxq.p;
        p.bounds = (const int32_t*) dbounds.p;
        p.ids = (const int32_t*) dids.p;
        p.total_rows = nr;
        p.max_rows = maxr;
        p.dst = dst;
        p.ld_dst = GUN;
        p.ptr_list = ptr;
        p.w_off = 0;
        if (ptr) for (int e = 0; e < nex; ++e) p.blobs[e] = blobs[e];
        ctx.run(p, nullptr);
    };
    Dev da(nr * GUN * 4), db(nr * GUN * 4);
    run(false, (float*) da.p);
    run(true, (float*) db.p);
    ck(cudaDeviceSynchronize(), "sync");
    std::vector<float> ha(nr * GUN), hb(nr * GUN);
    ck(cudaMemcpy(ha.data(), da.p, ha.size() * 4, cudaMemcpyDeviceToHost), "rb");
    ck(cudaMemcpy(hb.data(), db.p, hb.size() * 4, cudaMemcpyDeviceToHost), "rb");
    for (size_t i = 0; i < ha.size(); ++i)
        if (memcmp(&ha[i], &hb[i], 4)) {
            std::printf("big: divergence at %zu: stride=%a ptr=%a\n", i, ha[i], hb[i]);
            return 1;
        }
    std::printf("big: bitwise identical (%zu floats)\n", ha.size());
    return 0;
}
// masked group gather: experts 0,2 copy down-half only (dn_only bits), 1,3
// full. Gu slots of masked experts must stay sentinel; everything else must
// match per-expert gather_native. Guards the batched gather.
int run_groupmask() {
    std::mt19937 rng(6161);
    const ggml_type gt = GGML_TYPE_IQ3_XXS;
    const size_t gu_row = ggml_row_size(gt, N);
    const size_t up_off = gu_row * FF;
    const size_t gub = 2 * up_off;
    const size_t gdb = ggml_row_size(GGML_TYPE_IQ4_NL, FF) * N;
    const int nex = 4;
    std::vector<std::vector<uint8_t>> host_blob(nex, std::vector<uint8_t>(gub + gdb));
    for (int e = 0; e < nex; ++e)
        for (size_t i = 0; i < gub + gdb; ++i) host_blob[e][i] = (uint8_t) rng();
    std::vector<Dev> blob;
    for (int e = 0; e < nex; ++e) blob.emplace_back(gub + gdb + 16);
    for (int e = 0; e < nex; ++e)
        ck(cudaMemcpy(blob[e].p, host_blob[e].data(), gub + gdb, cudaMemcpyHostToDevice), "blob");
    Dev grp_gu(nex * gub + TAIL), grp_d(nex * gdb + TAIL);
    ck(cudaMemset(grp_gu.p, 0xAA, nex * gub + TAIL), "sentinel");
    ck(cudaMemset(grp_d.p, 0, nex * gdb + TAIL), "zero");
    mmq::GatherGroup gg;
    gg.first = 0; gg.n = nex; gg.dn_only = 0x5; // experts 0,2: down only
    for (int e = 0; e < nex; ++e) gg.blob[e] = (const uint8_t*) blob[e].p;
    if (!mmq::gather_native_group(gg, up_off, gub / 2, gub, gdb, grp_gu.p, gub, grp_d.p, gdb, nullptr))
    return 1; // aligned here by construction
    ck(cudaDeviceSynchronize(), "sync");
    int fails = 0;
    for (int e = 0; e < nex; ++e) {
        std::vector<uint8_t> gu(gub), dn(gdb);
        ck(cudaMemcpy(gu.data(), (uint8_t*) grp_gu.p + e * gub, gub, cudaMemcpyDeviceToHost), "rb");
        ck(cudaMemcpy(dn.data(), (uint8_t*) grp_d.p + e * gdb, gdb, cudaMemcpyDeviceToHost), "rb");
        if (memcmp(dn.data(), host_blob[e].data() + gub, gdb)) {
            std::printf("groupmask: expert %d down mismatch\n", e);
            fails = 1;
        }
        if (e & 1) {
            if (memcmp(gu.data(), host_blob[e].data(), gub)) {
                std::printf("groupmask: expert %d gu mismatch\n", e);
                fails = 1;
            }
        } else {
            for (size_t i = 0; i < gub; ++i)
                if (gu[i] != 0xAA) {
                    std::printf("groupmask: expert %d gu slot written at %zu\n", e, i);
                    fails = 1;
                    break;
                }
        }
    }
    if (!fails) std::printf("groupmask: masked group gather exact\n");
    return fails;
}

// replay of captured production dn inputs (/tmp/cap0_dn_*.bin): stride vs
// ptr + non-finite scan. Answers whether captured bytes decode finite.
int run_dn_cap() {
    const ggml_type dt = GGML_TYPE_IQ4_NL;
    const size_t gdb = 921600;
    const int nex = 16, nr = 48, maxr = 15; // captured bounds give maxr 15 -> J=16 like production
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreate(&cs), "stream");
    int32_t bounds_h[17];
    { FILE* f = std::fopen("/tmp/cap0_dn_bounds.bin", "rb"); if (!f) { std::printf("dn_cap: no capture files, skip\n"); return 0; }
      if (std::fread(bounds_h, 4, 17, f) != 17) return 1; std::fclose(f); }
    std::vector<Dev> blob;
    blob.reserve(nex);
    for (int e = 0; e < nex; ++e) blob.emplace_back(gdb + 16);
    for (int e = 0; e < nex; ++e) { char fn[64]; std::snprintf(fn, 64, "/tmp/cap0_dn_w%d.bin", e);
        FILE* f = std::fopen(fn, "rb"); if (!f) return 1;
        std::vector<uint8_t> wb(gdb); if (std::fread(wb.data(), 1, gdb, f) != gdb) return 1; std::fclose(f);
        ck(cudaMemcpyAsync(blob[e].p, wb.data(), gdb, cudaMemcpyHostToDevice, cs), "blob"); }
    Dev grp(nex * gdb + TAIL);
    for (int e = 0; e < nex; ++e)
        ck(cudaMemcpyAsync((uint8_t*) grp.p + e * gdb, blob[e].p, gdb, cudaMemcpyDeviceToDevice, cs), "grp");
    ck(cudaMemsetAsync((uint8_t*) grp.p + nex * gdb, 0, TAIL, cs), "tail");
    Dev hq(mmq::q8_bytes(nr, FF));
    { FILE* f = std::fopen("/tmp/cap0_dn_hq.bin", "rb"); if (!f) return 1;
      std::vector<uint8_t> hb(mmq::q8_bytes(nr, FF)); if (std::fread(hb.data(), 1, hb.size(), f) != hb.size()) return 1; std::fclose(f);
      ck(cudaMemcpyAsync(hq.p, hb.data(), hb.size(), cudaMemcpyHostToDevice, cs), "hq"); }
    Dev dbounds(17 * sizeof(int32_t)), dids(nr * sizeof(int32_t));
    std::vector<int32_t> ids(nr);
    for (int i = 0; i < nr; ++i) ids[i] = i;
    ck(cudaMemcpyAsync(dbounds.p, bounds_h, sizeof(bounds_h), cudaMemcpyHostToDevice, cs), "bounds");
    ck(cudaMemcpyAsync(dids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice, cs), "ids");
    mmq::Context ctx;
    const void* blobs[16];
    for (int e = 0; e < nex; ++e) blobs[e] = blob[e].p;
    auto run = [&](bool ptr, float* dst) {
        mmq::Product p;
        p.w = ptr ? blob[0].p : grp.p;
        p.type = (int) dt;
        p.w_rows = N;
        p.w_cols = FF;
        p.expert_bytes = gdb;
        p.n = nex;
        p.xq = hq.p;
        p.bounds = (const int32_t*) dbounds.p;
        p.ids = (const int32_t*) dids.p;
        p.total_rows = nr;
        p.max_rows = maxr;
        p.dst = dst;
        p.ld_dst = N;
        p.ptr_list = ptr;
        p.w_off = 0;
        if (ptr) for (int e = 0; e < nex; ++e) p.blobs[e] = blobs[e];
        ctx.run(p, cs);
    };
    Dev da(nr * N * 4), db(nr * N * 4);
    run(false, (float*) da.p);
    run(true, (float*) db.p);
    ck(cudaStreamSynchronize(cs), "sync");
    std::vector<float> ha(nr * N), hb(nr * N);
    ck(cudaMemcpy(ha.data(), da.p, ha.size() * 4, cudaMemcpyDeviceToHost), "rb");
    ck(cudaMemcpy(hb.data(), db.p, hb.size() * 4, cudaMemcpyDeviceToHost), "rb");
    size_t nan_a = 0, nan_b = 0;
    for (size_t i = 0; i < ha.size(); ++i) { if (!std::isfinite(ha[i])) ++nan_a; if (!std::isfinite(hb[i])) ++nan_b; }
    size_t nza = 0, nzb = 0; double sa = 0, sb = 0; for (size_t i = 0; i < ha.size(); ++i) { if (ha[i] != 0.0f) { ++nza; sa += ha[i]; } if (hb[i] != 0.0f) { ++nzb; sb += hb[i]; } } std::printf("dn_cap: nonzero stride=%zu/%zu (sum=%a) ptr=%zu/%zu (sum=%a)\n", nza, ha.size(), sa, nzb, hb.size(), sb);
    int fails = (nan_a > 0 || nan_b > 0) ? 1 : 0;
    for (size_t i = 0; i < ha.size(); ++i)
        if (memcmp(&ha[i], &hb[i], 4)) {
            std::printf("dn_cap: divergence at %zu: stride=%a ptr=%a\n", i, ha[i], hb[i]);
            fails = 1;
            break;
        }
    if (!fails) std::printf("dn_cap: bitwise identical (%zu floats)\n", ha.size());
    ck(cudaStreamDestroy(cs), "sdestroy");
    return fails;
}
}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev < 1) return 77;
    cudaDeviceProp p;
    ck(cudaGetDeviceProperties(&p, 0), "props");
    if (p.major * 10 + p.minor < 80) return 77;
    int fails = 0;
    const char* mode = std::getenv("STRATA_PTR_TEST_MODE"); // stride|ptr|both (default both)
    if (mode && !strcmp(mode, "stride")) g_mode = 1;
    if (mode && !strcmp(mode, "ptr")) g_mode = 2;
    fails += run_case(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, GUN, N, "iq3xxs/iq4nl");
    fails += run_big(GGML_TYPE_IQ3_XXS, GGML_TYPE_Q2_0, 0);
    fails += run_big(GGML_TYPE_IQ3_XXS, GGML_TYPE_Q2_0, 1);
    fails += run_dn16_maxr(0);
    fails += run_dn16_maxr(1);
    fails += run_dn16_maxr(2);
    fails += run_dn_prod();
    fails += run_gu_prod();
    fails += run_groupmask();
    fails += run_dn_cap();
    std::printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
