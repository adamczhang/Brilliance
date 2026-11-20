/* Brilliance: the process's pinned-memory limit (win/device/pin_ledger.h, choose_pin_limit), which
 * the native HIP backend takes from DXGI's shared-memory budget and BRILLIANCE_PIN_LIMIT_GIB. Here
 * the budget is handed in, so no card is needed:
 *   - unset, the limit is the budget / devices - the margin;
 *   - the variable at or below that is used as given;
 *   - above it, the limit stays at the budget's figure and the choice says it was clamped
 *     (2026-11-19, the multicard robustness run F3: 100 on a 47.83 GiB limit stalled a pin);
 *   - a value that is not a whole number of GiB above 0 is ignored and reported;
 *   - with no budget known, the variable is used as given. */
#include "rad_test.h"

#include "pin_ledger.h"

#include <cstdint>

namespace {

constexpr int64_t kGiB = (int64_t)1 << 30;
constexpr int64_t kMargin = 2 * kGiB;
/* The F3 machine: 99.67 GiB of shared budget, two cards, a 47.83 GiB limit. */
constexpr int64_t kBudget = (int64_t)(99.67 * (double)kGiB);
constexpr int64_t kLimit = kBudget / 2 - kMargin;

}  /* namespace */

TEST(unset_leaves_the_budget_over_the_devices_less_the_margin) {
    for (const char* e : { (const char*)nullptr, "" }) {
        const rad::PinLimitChoice c = rad::choose_pin_limit(kBudget, 2, kMargin, e);
        CHECK_EQ(c.computed, kLimit);
        CHECK_EQ(c.limit, kLimit);
        CHECK_EQ(c.requested, (int64_t)0);
        CHECK(!c.from_env);
        CHECK(!c.clamped);
        CHECK(!c.invalid);
    }
    /* Three devices visible (two cards and the integrated GPU) divide it by three. */
    CHECK_EQ(rad::choose_pin_limit(kBudget, 3, kMargin, nullptr).limit, kBudget / 3 - kMargin);
}

TEST(an_override_above_the_budget_is_clamped_to_it) {
    const rad::PinLimitChoice c = rad::choose_pin_limit(kBudget, 2, kMargin, "100");
    CHECK_EQ(c.limit, kLimit);
    CHECK_EQ(c.computed, kLimit);
    CHECK_EQ(c.requested, 100 * kGiB);
    CHECK(c.clamped);
    CHECK(!c.from_env);
    CHECK(!c.invalid);
    /* One GiB over is still over; a value too large to hold in bytes is clamped, not wrapped. */
    CHECK(rad::choose_pin_limit(64 * kGiB, 1, kMargin, "63").clamped);
    const rad::PinLimitChoice big = rad::choose_pin_limit(kBudget, 2, kMargin, "99999999999999999999");
    CHECK(big.clamped);
    CHECK_EQ(big.limit, kLimit);
    CHECK(big.requested > 0);
}

TEST(an_override_at_or_below_the_budget_is_kept) {
    const rad::PinLimitChoice c = rad::choose_pin_limit(kBudget, 2, kMargin, "8");
    CHECK_EQ(c.limit, 8 * kGiB);
    CHECK_EQ(c.requested, 8 * kGiB);
    CHECK(c.from_env);
    CHECK(!c.clamped);
    CHECK(!c.invalid);
    /* Exactly the budget's figure is kept, not clamped. */
    const rad::PinLimitChoice eq = rad::choose_pin_limit(64 * kGiB, 1, kMargin, "62");
    CHECK_EQ(eq.limit, 62 * kGiB);
    CHECK(eq.from_env);
    CHECK(!eq.clamped);
    /* Spaces around the number are allowed. */
    CHECK_EQ(rad::choose_pin_limit(kBudget, 2, kMargin, " 8 ").limit, 8 * kGiB);
}

TEST(an_invalid_override_is_ignored_and_reported) {
    for (const char* e : { "0", "-5", "abc", "8GiB", "1.5", " " }) {
        const rad::PinLimitChoice c = rad::choose_pin_limit(kBudget, 2, kMargin, e);
        CHECK(c.invalid);
        CHECK_EQ(c.limit, kLimit);
        CHECK(!c.from_env);
        CHECK(!c.clamped);
    }
}

TEST(with_no_budget_known_the_override_is_used_as_given) {
    const rad::PinLimitChoice none = rad::choose_pin_limit(-1, 2, kMargin, nullptr);
    CHECK_EQ(none.computed, (int64_t)-1);
    CHECK_EQ(none.limit, (int64_t)-1);
    const rad::PinLimitChoice c = rad::choose_pin_limit(-1, 0, kMargin, "100");
    CHECK_EQ(c.limit, 100 * kGiB);
    CHECK(c.from_env);
    CHECK(!c.clamped);
    /* A budget smaller than the margin leaves nothing, and any override is above that. */
    const rad::PinLimitChoice tiny = rad::choose_pin_limit(kGiB, 1, kMargin, "1");
    CHECK_EQ(tiny.computed, (int64_t)0);
    CHECK_EQ(tiny.limit, (int64_t)0);
    CHECK(tiny.clamped);
}

RAD_TEST_MAIN()
