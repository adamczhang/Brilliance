/* Brilliance: how r4d_selftest finds libref beside the plugin under test (win/compat/include/
 * brl_path.h). r4d_selftest cuts the plugin's path at its last '/' and opens "libref.so" there;
 * the POSIX layer's dlopen loads the libref.dll of that name. A plugin path written with '\'
 * (as a PowerShell gate passes it) has no '/', so before the fix the cut left a bare name and the
 * 27 rows that check a kernel against libref failed on Windows (2026-11-20, st1).
 *   - the cut of a '\' path leaves no directory, which is why the path is converted first;
 *   - the converted path's cut names the libref beside the plugin, and dlopen loads it, with the
 *     ops those rows ask it for;
 *   - a '/' path is unchanged by the conversion.
 * No card, no HIP: libref is the host reference library. The build tree's kernels directory comes
 * from CMake (BRL_TEST_KERNELS). */
#include "rad_test.h"

#include "rad_abi.h"

#include <brl_path.h>
#include <dlfcn.h>

#include <cstring>
#include <string>

namespace {

/* r4d_selftest's spelling, as every libref row writes it. */
std::string ref_beside(const std::string& plugin) {
    return plugin.substr(0, plugin.find_last_of('/') + 1) + "libref.so";
}

std::string backslashed(std::string p) {
    for (char& c : p)
        if (c == '/') c = '\\';
    return p;
}

const std::string kKernels = BRL_TEST_KERNELS;

}  /* namespace */

TEST(a_backslash_plugin_path_cuts_to_a_bare_name) {
    const std::string plugin = backslashed(kKernels + "/libr4d.dll");
    CHECK_EQ(ref_beside(plugin), std::string("libref.so"));
}

TEST(a_forward_slash_path_is_unchanged) {
    const std::string plugin = kKernels + "/libr4d.dll";
    CHECK_EQ(brl_forward_slashes(plugin), plugin);
    CHECK_EQ(brl_forward_slashes("C:\\a b\\kernels\\libr4d.dll"), std::string("C:/a b/kernels/libr4d.dll"));
    CHECK_EQ(brl_forward_slashes("\\\\host\\share\\k\\libr4d.dll"), std::string("//host/share/k/libr4d.dll"));
}

TEST(the_converted_path_finds_libref_beside_the_plugin) {
    NEED_ARTIFACT(kKernels + "/libref.dll");
    const std::string plugin = brl_forward_slashes(backslashed(kKernels + "/libr4d.dll"));
    const std::string ref = ref_beside(plugin);
    CHECK_EQ(ref, kKernels + "/libref.so");
    void* h = dlopen(ref.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char* e = dlerror();
        ::radtest::fail(__FILE__, __LINE__, std::string("dlopen: ") + (e ? e : "?"));
        return;
    }
    auto count = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    REQUIRE(count != nullptr);
    REQUIRE(at != nullptr);
    /* The ops of the rows that failed: the MLA, KDA, DSA, SSD, router, mHC, SwiGLU and KV ones. */
    for (const char* op : { "mla_attn_paged", "ssd_update", "kda_update", "router_sigmoid_topk",
                            "kv_resolve", "kv_store_streamed", "swiglu_clamped" }) {
        bool found = false;
        for (int i = 0; i < count() && !found; ++i) found = std::strcmp(at(i)->op, op) == 0;
        if (!found) ::radtest::fail(__FILE__, __LINE__, std::string("libref has no ") + op);
        ++::radtest::checks();
    }
    dlclose(h);
}

RAD_TEST_MAIN()
