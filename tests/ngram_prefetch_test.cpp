/* ngram_prefetch_test.cpp -- Brilliance: the Windows n-gram prefetch's bookkeeping and its system call.
 *
 * On Windows the disk-tier n-gram gather reads the table through the container's mapping and, before
 * the loads fault on its rows one at a time, hands their pages to PrefetchVirtualMemory together; a
 * read-ahead thread does the same for the following chunks (libavx/avx_ngram.cpp). What it decides is
 * in libavx/avx_ngram_pf.h, a pure header, checked here on any host:
 *
 *   - the pages of a set of rows: every page a row touches, a row across a page boundary two, each
 *     page once, padding and other ranks' ids none, and adjacent pages merged into one range;
 *   - a rank's share of a call's rows (m % world_size == rank) covers every row once between ranks;
 *   - an ahead list that moves on keeps what was read, hands over only the new ids, and a list that
 *     does not continue the last one starts again (its queued pieces dropped);
 *   - the read-ahead's progress moves only over contiguous finished pieces, so a failed piece's rows
 *     are never counted as mapped, and a call counts as covered only when all of its rows are.
 *
 * On Windows it also checks brl_prefetch_ranges on a mapped file (win/compat): nothing asked is
 * nothing done, a null array is EINVAL with a reason, and three ranges read in leave the view
 * holding the file's bytes. Adapted from wmmunn's tests for the same prefetch. */
#include "rad_test.h"
#include "../libavx/avx_ngram_pf.h"

#if defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>   /* brl_prefetch_ranges (win/compat) */
#include <unistd.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <numeric>
#include <set>
#include <vector>

using namespace avx::pf;

namespace {

/* A table at a page-aligned address that is never dereferenced: the page math only. */
const uint8_t* const kTab = reinterpret_cast<const uint8_t*>((uintptr_t)1 << 32);

std::vector<uintptr_t> pages_of(int64_t row_bytes, const std::vector<int32_t>& ids,
                                int64_t voff = 0, int64_t n_vocab = 1 << 20) {
    std::vector<uintptr_t> p;
    row_pages(kTab, row_bytes, voff, n_vocab, ids.data(), (int64_t)ids.size(),
              [](int64_t) { return true; }, p);
    return p;
}

uintptr_t page0() { return (uintptr_t)kTab / kPage; }

}  // namespace

/* ---- the pages of a call's rows --------------------------------------------------------- */

TEST(a_row_inside_one_page_is_one_page) {
    /* Flash-Next's rows: 160 bytes, so rows 0-24 share page 0 and row 25 straddles 0 and 1. */
    const auto p = pages_of(160, { 3 });
    REQUIRE_EQ(p.size(), (size_t)1);
    CHECK_EQ(p[0], page0());
}

TEST(a_row_across_a_page_boundary_is_two_pages) {
    const auto p = pages_of(160, { 25 });   /* bytes 4000-4159 */
    REQUIRE_EQ(p.size(), (size_t)2);
    CHECK_EQ(p[0], page0());
    CHECK_EQ(p[1], page0() + 1);
}

TEST(each_page_is_asked_for_once_and_sorted) {
    const auto p = pages_of(160, { 900, 3, 4, 900, 5, 1000 });
    for (size_t i = 1; i < p.size(); ++i) CHECK(p[i - 1] < p[i]);
    const std::set<uintptr_t> s(p.begin(), p.end());
    CHECK_EQ(s.size(), p.size());
    CHECK(s.count(page0()));
    CHECK(s.count(page0() + 900 * 160 / kPage));
}

TEST(padding_and_other_ranks_ids_have_no_pages) {
    /* A vocabulary slice [1000, 2000): ids below and at or past it are not this table's. */
    CHECK(pages_of(160, { -1, 999, 2000, 5000 }, 1000, 1000).empty());
    CHECK_EQ(pages_of(160, { 999, 1000 }, 1000, 1000).size(), (size_t)1);
}

TEST(adjacent_pages_merge_into_one_range) {
    std::vector<uintptr_t> p = { 10, 11, 12, 20, 22, 23 };
    std::vector<Range> r;
    merge_pages(p, r);
    REQUIRE_EQ(r.size(), (size_t)3);
    CHECK_EQ(r[0].addr, (uintptr_t)10 * kPage);
    CHECK_EQ(r[0].bytes, (size_t)3 * kPage);
    CHECK_EQ(r[1].addr, (uintptr_t)20 * kPage);
    CHECK_EQ(r[1].bytes, (size_t)kPage);
    CHECK_EQ(r[2].addr, (uintptr_t)22 * kPage);
    CHECK_EQ(r[2].bytes, (size_t)2 * kPage);
    p.clear();
    merge_pages(p, r);
    CHECK(r.empty());
}

TEST(spans_of_two_planes_share_pages) {
    /* Engram's lookups: a row of codes and a row of scales, each its own span. */
    const uint8_t* codes = kTab;
    const uint8_t* scales = kTab + 64 * kPage;
    const Span s[4] = { { codes, 2560 }, { scales, 320 }, { codes + 2560, 2560 }, { scales + 320, 320 } };
    std::vector<uintptr_t> p;
    span_pages(s, 4, p);
    /* codes: bytes 0-5119, pages 0 and 1; scales: one page. */
    REQUIRE_EQ(p.size(), (size_t)3);
    CHECK_EQ(p[0], page0());
    CHECK_EQ(p[1], page0() + 1);
    CHECK_EQ(p[2], page0() + 64);
    span_pages(nullptr, 0, p);
    CHECK(p.empty());
}

TEST(two_ranks_shares_cover_every_row_once) {
    std::vector<int32_t> ids(1000);
    std::iota(ids.begin(), ids.end(), 0);
    for (auto& v : ids) v = (v * 7919) % 50000;   /* scattered rows */
    std::vector<uintptr_t> all, r0, r1;
    row_pages(kTab, 160, 0, 1 << 20, ids.data(), (int64_t)ids.size(), [](int64_t) { return true; }, all);
    row_pages(kTab, 160, 0, 1 << 20, ids.data(), (int64_t)ids.size(),
              [](int64_t m) { return m % 2 == 0; }, r0);
    row_pages(kTab, 160, 0, 1 << 20, ids.data(), (int64_t)ids.size(),
              [](int64_t m) { return m % 2 == 1; }, r1);
    std::set<uintptr_t> u(r0.begin(), r0.end());
    u.insert(r1.begin(), r1.end());
    CHECK_EQ(u.size(), all.size());
    CHECK(r0.size() < all.size() && r1.size() < all.size());
}

/* ---- the ahead list ---------------------------------------------------------------------- */

TEST(a_list_that_moves_on_is_found) {
    const std::vector<int32_t> p = { 1, 2, 3, 4, 5, 6 };
    const std::vector<int32_t> w = { 3, 4, 5, 6, 7, 8 };
    int64_t s = 0;
    CHECK_EQ(list_shift(p.data(), 6, w.data(), 6, 64, &s), (int64_t)4);
    CHECK_EQ(s, (int64_t)2);
}

TEST(a_list_that_does_not_continue_is_new) {
    const std::vector<int32_t> p = { 1, 2, 3, 4 };
    const std::vector<int32_t> w = { 9, 9, 9 };
    int64_t s = 0;
    CHECK_EQ(list_shift(p.data(), 4, w.data(), 3, 64, &s), (int64_t)-1);
    CHECK_EQ(list_shift(p.data(), 0, w.data(), 3, 64, &s), (int64_t)-1);   /* no last list */
}

TEST(the_smallest_shift_is_taken_and_tries_are_bounded) {
    /* A text that repeats: 5 5 5 5 moved on by 1 matches at s = 1 first. */
    const std::vector<int32_t> p = { 5, 5, 5, 5 };
    const std::vector<int32_t> w = { 5, 5, 5, 5 };
    int64_t s = 0;
    CHECK_EQ(list_shift(p.data(), 4, w.data(), 4, 64, &s), (int64_t)3);
    CHECK_EQ(s, (int64_t)1);
    /* With no tries left nothing matches. */
    CHECK_EQ(list_shift(p.data(), 4, w.data(), 4, 0, &s), (int64_t)-1);
}

TEST(a_new_list_is_handed_over_in_pieces_nearest_first) {
    AheadStream st;
    std::vector<int32_t> w(5000);
    std::iota(w.begin(), w.end(), 100);
    bool restarted = true;
    const auto pcs = st.hand_over(w.data(), 5000, 2048, 64, &restarted);
    CHECK(!restarted);   /* the first list drops nothing */
    REQUIRE_EQ(pcs.size(), (size_t)3);
    CHECK_EQ(pcs[0].start, (int64_t)0);
    CHECK_EQ(pcs[0].end, (int64_t)2048);
    CHECK_EQ(pcs[0].ids.front(), 100);
    CHECK_EQ(pcs[2].start, (int64_t)4096);
    CHECK_EQ(pcs[2].end, (int64_t)5000);
    CHECK_EQ(pcs[2].ids.back(), 5099);
}

TEST(a_repeated_list_hands_over_nothing) {
    /* The other rank's call, with the same list. */
    AheadStream st;
    std::vector<int32_t> w = { 1, 2, 3 };
    bool restarted = false;
    CHECK_EQ(st.hand_over(w.data(), 3, 2, 64, &restarted).size(), (size_t)2);
    CHECK(st.hand_over(w.data(), 3, 2, 64, &restarted).empty());
    CHECK(!restarted);
}

TEST(a_list_that_moves_on_hands_over_only_its_new_ids) {
    AheadStream st;
    std::vector<int32_t> w1(10), w2(10);
    std::iota(w1.begin(), w1.end(), 0);    /* 0..9 */
    std::iota(w2.begin(), w2.end(), 4);    /* 4..13: moved on by 4 */
    bool restarted = false;
    for (const auto& pc : st.hand_over(w1.data(), 10, 100, 64, &restarted)) st.complete(pc);
    CHECK_EQ(st.done(), (int64_t)10);
    const auto pcs = st.hand_over(w2.data(), 10, 100, 64, &restarted);
    CHECK(!restarted);
    REQUIRE_EQ(pcs.size(), (size_t)1);
    CHECK_EQ(pcs[0].start, (int64_t)10);   /* positions continue */
    CHECK_EQ(pcs[0].end, (int64_t)14);
    REQUIRE_EQ(pcs[0].ids.size(), (size_t)4);
    CHECK_EQ(pcs[0].ids[0], 10);
    CHECK_EQ(st.base(), (int64_t)4);
    /* The next call's ids (4..9) are already mapped. */
    CHECK(st.covered(w2.data(), 6));
    CHECK(!st.covered(w2.data(), 7));
}

TEST(a_list_that_does_not_continue_restarts) {
    AheadStream st;
    std::vector<int32_t> w1 = { 1, 2, 3, 4 }, w2 = { 50, 51 };
    bool restarted = false;
    const auto old = st.hand_over(w1.data(), 4, 2, 64, &restarted);
    const uint64_t g = st.gen();
    const auto pcs = st.hand_over(w2.data(), 2, 2, 64, &restarted);
    CHECK(restarted);
    CHECK(st.gen() != g);
    REQUIRE_EQ(pcs.size(), (size_t)1);
    CHECK_EQ(pcs[0].start, st.base());
    /* A piece of the dropped list finishing late moves nothing. */
    st.complete(old[0]);
    CHECK_EQ(st.done(), st.base());
    CHECK(!st.covered(w2.data(), 2));
    st.complete(pcs[0]);
    CHECK(st.covered(w2.data(), 2));
}

TEST(progress_moves_only_over_contiguous_pieces) {
    AheadStream st;
    std::vector<int32_t> w(6);
    std::iota(w.begin(), w.end(), 0);
    bool restarted = false;
    const auto pcs = st.hand_over(w.data(), 6, 2, 64, &restarted);
    REQUIRE_EQ(pcs.size(), (size_t)3);
    /* The first piece failed (never completed): the later ones leave `done` where it was. */
    st.complete(pcs[1]);
    st.complete(pcs[2]);
    CHECK_EQ(st.done(), (int64_t)0);
    CHECK(!st.covered(w.data(), 1));
    st.complete(pcs[0]);
    CHECK_EQ(st.done(), (int64_t)2);   /* the later pieces finished before it are not recounted */
}

TEST(a_call_is_covered_only_by_its_own_ids) {
    AheadStream st;
    std::vector<int32_t> w = { 7, 8, 9, 10 };
    bool restarted = false;
    for (const auto& pc : st.hand_over(w.data(), 4, 4, 64, &restarted)) st.complete(pc);
    const std::vector<int32_t> same = { 7, 8 }, other = { 7, 99 };
    CHECK(st.covered(same.data(), 2));
    CHECK(!st.covered(other.data(), 2));
    CHECK(!st.covered(w.data(), 5));       /* longer than the list */
    CHECK(!st.covered(w.data(), 0));
}

/* ---- the system call (Windows) ----------------------------------------------------------- */

#if defined(_WIN32)
TEST(prefetch_ranges_reads_a_mapped_file) {
    CHECK_EQ(brl_prefetch_ranges(nullptr, 0, nullptr, 0), 0);   /* nothing asked: nothing to do */
    char why[256] = "";
    errno = 0;
    CHECK_EQ(brl_prefetch_ranges(nullptr, 2, why, sizeof why), -1);
    CHECK_EQ(errno, EINVAL);
    CHECK(why[0] != 0);

    const std::string path = radtest::tmp_path("ngram_prefetch.bin");
    const size_t bytes = 1u << 20, page = 4096;
    std::vector<unsigned char> data(bytes);
    uint64_t x = 0x2545f4914f6cdd1dull;
    for (auto& b : data) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; b = (unsigned char)x; }
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        REQUIRE(f != nullptr);
        CHECK_EQ(std::fwrite(data.data(), 1, bytes, f), bytes);
        std::fclose(f);
    }
    const int fd = ::open(path.c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    void* map = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    REQUIRE(map != MAP_FAILED);
    uint8_t* v = static_cast<uint8_t*>(map);
    /* One page, two adjacent pages as one range, and the file's last page. */
    const brl_prefetch_range r[3] = { { v + 5 * page, page }, { v + 40 * page, 2 * page },
                                      { v + bytes - page, page } };
    CHECK_EQ(brl_prefetch_ranges(r, 3, why, sizeof why), 0);
    CHECK(std::memcmp(v + 5 * page, data.data() + 5 * page, page) == 0);
    CHECK(std::memcmp(v + 40 * page, data.data() + 40 * page, 2 * page) == 0);
    CHECK(std::memcmp(v + bytes - page, data.data() + bytes - page, page) == 0);
    /* The ranges a gather builds, passed as libavx passes them (avx_ngram.cpp: pf_issue). */
    std::vector<uintptr_t> pages;
    std::vector<int32_t> ids = { 0, 25, 1000, 6000 };   /* 160-byte rows; row 25 straddles pages */
    row_pages(v, 160, 0, (int64_t)(bytes / 160), ids.data(), (int64_t)ids.size(),
              [](int64_t) { return true; }, pages);
    std::vector<Range> ranges;
    merge_pages(pages, ranges);
    CHECK_EQ(brl_prefetch_ranges(reinterpret_cast<const brl_prefetch_range*>(ranges.data()),
                                 ranges.size(), why, sizeof why), 0);
    CHECK_EQ(munmap(map, bytes), 0);
}

TEST(a_range_has_the_entry_layout) {
    /* libavx passes avx::pf::Range arrays as brl_prefetch_range (avx_ngram.cpp). */
    CHECK_EQ(sizeof(Range), sizeof(brl_prefetch_range));
    CHECK_EQ(offsetof(Range, bytes), offsetof(brl_prefetch_range, bytes));
}
#endif

RAD_TEST_MAIN()
