/* Native grouped IQ1_S and IQ2_XXS parity at the released trunk's widths.
 * The runner restricts this process to one leased gfx1201 card. */
#include "rad_test.h"
#include "lq_ggml_planes.h"
#include "plugin/registry.h"
#include "rad_device.h"
#include "rad_plugin.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace {
using namespace rad;
using namespace lq::ggml;

const RadKernelInfo* named(const Registry& registry, const char* name) {
    for (const auto& row : registry.rows())
        if (std::strcmp(row.info->name, name) == 0) return row.info;
    return nullptr;
}

RadTensor tensor(void* data, uint32_t dtype, std::initializer_list<int64_t> shape) {
    RadTensor t{};
    t.data = data;
    t.dtype = dtype;
    t.rank = (uint32_t)shape.size();
    std::copy(shape.begin(), shape.end(), t.shape);
    rad_tensor_pack(&t);
    return t;
}

struct Stream {
    RadStream stream = nullptr;
    Stream() { CHECK_EQ(rad_stream_create(&stream, 0), RAD_OK); }
    ~Stream() { if (stream) rad_stream_destroy(stream); }
};

struct DeviceData {
    void* data;
    int64_t bytes;

    explicit DeviceData(int64_t count) : data(rad_dev_alloc(count, RAD_MEM_DEVICE)), bytes(count) {}
    DeviceData(const DeviceData&) = delete;
    DeviceData& operator=(const DeviceData&) = delete;
    ~DeviceData() { if (data) rad_dev_free(data, RAD_MEM_DEVICE); }
};

void grouped_case(const RadKernelInfo& ref, const RadKernelInfo& hip, ggml_type type,
                  int64_t n, int64_t k, int64_t m, bool down, RadStream stream, bool& launched) {
    launched = false;
    constexpr int64_t top_k = 6;
    const auto* kind = plane_kind(type);
    REQUIRE(kind != nullptr);
    const int64_t t = m * top_k, ar = down ? t : m;
    const int64_t ap = k + 16, yp = n + 16, rb = k / kind->block * (int64_t)kind->block_bytes;
    const int64_t wb = n * rb;
    const int64_t used = std::min<int64_t>(m, 3);
    const uint16_t sentinel = 0x3f80;
    std::vector<uint16_t> a((size_t)(ar * ap), sentinel), y((size_t)(t * yp), sentinel), actual = y;
    std::vector<uint8_t> weights((size_t)(2 * wb));
    std::vector<int32_t> sorted((size_t)t, -1);
    std::array<int32_t, 3> offsets = { 0, (int32_t)used, (int32_t)(used * 2) };
    for (int64_t e = 0; e < 2; ++e)
        for (int64_t i = 0; i < used; ++i) sorted[(size_t)(e * used + i)] = (int32_t)(i * top_k + e);
    for (int64_t r = 0; r < ar; ++r)
        for (int64_t c = 0; c < k; ++c)
            rad_store_f32(a.data(), RAD_BF16, r * ap + c,
                          (float)((r * 11 + c * 7) % 127 - 63) / 256.0f);

    uint32_t rng = 413;
    for (uint8_t& v : weights) {
        rng = rng * 1664525u + 1013904223u;
        v = (uint8_t)(rng >> 24);
    }
    for (size_t i = 0; i < weights.size(); i += kind->block_bytes) {
        const uint16_t scale = 0x1600;  /* finite, nonzero, and exact in f16 */
        std::memcpy(weights.data() + i, &scale, sizeof scale);
    }

    std::array<void*, 2> host_table = { weights.data(), weights.data() + wb };
    RadTensor inputs[] = {
        tensor(a.data(), RAD_BF16, { ar, k }), tensor(host_table.data(), RAD_U8, { 2, n, rb }),
        tensor(sorted.data(), RAD_I32, { t }), tensor(offsets.data(), RAD_I32, { 3 }),
        tensor(y.data(), RAD_BF16, { t, n }),
    };
    inputs[0].stride[0] = ap;
    inputs[1].stride[0] = 0;
    inputs[4].stride[0] = yp;
    const RadParam params[] = {
        RAD_INT("M", m), RAD_INT("N", n), RAD_INT("K", k), RAD_INT("n_expert", 2),
        RAD_INT("top_k", top_k), RAD_STR("a_order", down ? "sorted" : "token"),
        RAD_STR("type", kind->name), RAD_STR("dtype", "bf16"),
    };
    RadArgs args{ inputs, 5, params, 8, nullptr, 0, nullptr, 0, 1, -1 };
    REQUIRE_EQ(ref.launch(&args, nullptr), RAD_OK);

    DeviceData da((int64_t)(a.size() * sizeof(uint16_t))), dw(2 * wb), dt(sizeof host_table);
    DeviceData ds((int64_t)(sorted.size() * sizeof(int32_t))), doff(sizeof offsets);
    DeviceData dy((int64_t)(actual.size() * sizeof(uint16_t)));
    REQUIRE(da.data && dw.data && dt.data && ds.data && doff.data && dy.data);
    std::array<void*, 2> device_table = { dw.data, (uint8_t*)dw.data + wb };
    REQUIRE_EQ(rad_memcpy_async(da.data, a.data(), da.bytes, stream), RAD_OK);
    REQUIRE_EQ(rad_memcpy_async(dw.data, weights.data(), dw.bytes, stream), RAD_OK);
    REQUIRE_EQ(rad_memcpy_async(dt.data, device_table.data(), dt.bytes, stream), RAD_OK);
    REQUIRE_EQ(rad_memcpy_async(ds.data, sorted.data(), ds.bytes, stream), RAD_OK);
    REQUIRE_EQ(rad_memcpy_async(doff.data, offsets.data(), doff.bytes, stream), RAD_OK);
    REQUIRE_EQ(rad_memcpy_async(dy.data, actual.data(), dy.bytes, stream), RAD_OK);
    inputs[0].data = da.data;
    inputs[1].data = dt.data;
    inputs[2].data = ds.data;
    inputs[3].data = doff.data;
    inputs[4].data = dy.data;
    REQUIRE_EQ(hip.launch(&args, stream), RAD_OK);
    REQUIRE_EQ(rad_memcpy_async(actual.data(), dy.data, dy.bytes, stream), RAD_OK);
    REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
    launched = true;

    double sq_error = 0.0, sq_ref = 0.0;
    int altered_padding = 0, nonzero_dropped = 0;
    for (int64_t r = 0; r < t; ++r) {
        for (int64_t c = 0; c < n; ++c) {
            const int64_t i = r * yp + c;
            const double want = rad_load_f32(y.data(), RAD_BF16, i);
            const double got = rad_load_f32(actual.data(), RAD_BF16, i);
            sq_ref += want * want;
            sq_error += (got - want) * (got - want);
            if (r >= 2 * used && got != 0.0) ++nonzero_dropped;
        }
        for (int64_t c = n; c < yp; ++c)
            if (actual[(size_t)(r * yp + c)] != sentinel) ++altered_padding;
    }
    REQUIRE(sq_ref > 0.0);
    const double rel = std::sqrt(sq_error / sq_ref);
    std::fprintf(stderr, "%s %s %s M=%lld N=%lld K=%lld rel_l2=%.7f\n",
                 hip.name, kind->name, down ? "down" : "gate/up",
                 (long long)m, (long long)n, (long long)k, rel);
    CHECK(rel < 0.015);
    CHECK_EQ(altered_padding, 0);
    CHECK_EQ(nonzero_dropped, 0);
}
}  // namespace

TEST(native_ggml_expert_rows_match_scalar_at_bakon3_widths) {
    if (!std::getenv("BRILLIANCE_DS41_HIP_ROWS")) SKIP("requires the single-card DeepSeek fixture lease");
    REQUIRE_EQ(rad_dev_count(), 1);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    REQUIRE(std::strcmp(props.arch, "gfx1201") == 0);
    Registry registry;
    REQUIRE_EQ(registry.load_dir(home() + "/kernels", { "libr4d", "libref" }), RAD_OK);
    const auto* ref = named(registry, "ref_moe_gemm_ggml");
    const auto* gemv = named(registry, "moe_gemm_ggml_qk256_gemv");
    const auto* tiled = named(registry, "moe_gemm_ggml_qk256_tiled");
    REQUIRE(ref && gemv && tiled);
    Stream stream;
    REQUIRE(stream.stream != nullptr);
    bool launched = false;
    for (auto type : { GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_XXS }) {
        grouped_case(*ref, *gemv, type, 4608, 5120, 1, false, stream.stream, launched);
        REQUIRE(launched);
        grouped_case(*ref, *gemv, type, 5120, 2304, 1, true, stream.stream, launched);
        REQUIRE(launched);
        grouped_case(*ref, *tiled, type, 4608, 5120, 49, false, stream.stream, launched);
        REQUIRE(launched);
        grouped_case(*ref, *tiled, type, 5120, 2304, 49, true, stream.stream, launched);
        REQUIRE(launched);
    }
}

RAD_TEST_MAIN()
