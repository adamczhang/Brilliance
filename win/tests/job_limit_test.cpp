/* Brilliance (issue #1): a Windows job object's memory limit (win/os/job_limit.h).
 *
 *   - outside a job, or in one with no memory limit, nothing is limited and nothing is said;
 *   - the binding limit is the smaller of the job's and the per-process one;
 *   - a plan whose pools and pins pass the limit is refused, naming the limit and the figure that
 *     clears it: the issue's case, two Flash-Next ranks (about 2 x 28.4 GiB of card pools and
 *     16 GiB of pins) under a 56 GiB job limit;
 *   - a plan under it passes and says what it leaves;
 *   - os::job_memory_limit() reads a real job: this process assigns itself to one whose limits
 *     (1 TiB each) no test comes near, and reads them back. A process cannot leave a job, so this
 *     runs last and the limits are far above anything the test allocates. */
#include "rad_test.h"

#include "os/os.h"

#include <windows.h>

#include <cstdint>
#include <string>

namespace {

constexpr int64_t kGiB = (int64_t)1 << 30;

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

}  /* namespace */

TEST(no_limit_says_nothing_and_never_refuses) {
    rad::JobMemoryLimit none;
    CHECK_EQ(none.binding(), (int64_t)-1);
    CHECK(rad::job_limit_note(none).empty());
    const rad::JobLimitVerdict v = rad::job_limit_verdict(none, 1 * kGiB, 1000 * kGiB, 1000 * kGiB);
    CHECK(!v.limited);
    CHECK(!v.refuse);
    CHECK(v.message.empty());

    rad::JobMemoryLimit in_job;   /* a job with only kill-on-close, as the service's fetch step */
    in_job.in_job = true;
    CHECK_EQ(in_job.binding(), (int64_t)-1);
    CHECK(!rad::job_limit_verdict(in_job, 0, 1000 * kGiB, 0).refuse);
}

TEST(the_smaller_limit_binds) {
    rad::JobMemoryLimit j;
    j.in_job = true;
    j.job = 56 * kGiB;
    CHECK_EQ(j.binding(), 56 * kGiB);
    CHECK(has(rad::job_limit_which(j), "all of its processes together, 56.00 GiB"));
    j.process = 40 * kGiB;
    CHECK_EQ(j.binding(), 40 * kGiB);
    CHECK(has(rad::job_limit_which(j), "each process in it, 40.00 GiB"));
    j.job = -1;
    CHECK_EQ(j.binding(), 40 * kGiB);
}

TEST(the_issue_case_is_refused_naming_the_limit_and_the_clearing_figure) {
    rad::JobMemoryLimit j;
    j.in_job = true;
    j.job = 56 * kGiB;
    const int64_t vram = 2 * (int64_t)(28.42 * (double)kGiB);
    const int64_t pins = 16 * kGiB;
    const rad::JobLimitVerdict v = rad::job_limit_verdict(j, 2 * kGiB, vram, pins);
    CHECK(v.limited);
    CHECK(v.refuse);
    CHECK(has(v.message, "Windows job object"));
    CHECK(has(v.message, "56.00 GiB"));
    CHECK(has(v.message, "WDDM charges to commit as hipMalloc memory"));
    CHECK(has(v.message, "Raise the limit above 74.84 GiB"));
    CHECK(!has(v.message, "rocm-smi"));
    const std::string note = rad::job_limit_note(j);
    CHECK(has(note, "56.00 GiB"));
    CHECK(has(note, "room"));
}

TEST(a_plan_under_the_limit_passes_and_says_what_it_leaves) {
    rad::JobMemoryLimit j;
    j.in_job = true;
    j.process = 200 * kGiB;
    const rad::JobLimitVerdict v = rad::job_limit_verdict(j, 2 * kGiB, 60 * kGiB, 24 * kGiB);
    CHECK(v.limited);
    CHECK(!v.refuse);
    CHECK(has(v.message, "leaving 114.00 GiB"));
    /* A commit figure that could not be read counts as none, not as a negative. */
    CHECK(!rad::job_limit_verdict(j, -1, 100 * kGiB, 100 * kGiB).refuse);
    CHECK(rad::job_limit_verdict(j, -1, 150 * kGiB, 51 * kGiB).refuse);
}

/* With HIP virtual memory (one card visible) the cards' pools are not charged to commit: measured,
 * 8.61 GiB of VRAM at 0.87 GiB of private commit. Only the pins and what is committed count. */
TEST(virtual_memory_pools_are_not_counted_against_the_limit) {
    rad::JobMemoryLimit j;
    j.in_job = true;
    j.job = 6 * kGiB;
    const rad::JobLimitVerdict charged = rad::job_limit_verdict(j, kGiB / 4, (int64_t)(8.43 * (double)kGiB),
                                                                kGiB / 2, true);
    CHECK(charged.refuse);
    const rad::JobLimitVerdict vm = rad::job_limit_verdict(j, kGiB / 4, (int64_t)(8.43 * (double)kGiB),
                                                           kGiB / 2, false);
    CHECK(!vm.refuse);
    CHECK(has(vm.message, "HIP virtual memory, not charged"));
    CHECK(has(vm.message, "leaving 5.25 GiB"));
    const rad::JobLimitVerdict vm_small = rad::job_limit_verdict(j, kGiB / 4, 100 * kGiB, 6 * kGiB, false);
    CHECK(vm_small.refuse);
    CHECK(has(vm_small.message, "HIP virtual memory, which commit does not count"));
}

TEST(zz_a_real_job_s_limits_are_read_back) {
    const rad::JobMemoryLimit before = rad::os::job_memory_limit();
    CHECK(rad::os::process_commit() > 0);
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    CHECK(job != nullptr);
    if (!job) return;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    li.JobMemoryLimit = (SIZE_T)(1024 * kGiB);
    li.ProcessMemoryLimit = (SIZE_T)(1024 * kGiB);
    CHECK(SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof li));
    if (!AssignProcessToJobObject(job, GetCurrentProcess())) {
        /* A parent job that forbids nesting (rare since Windows 8): nothing to read back. */
        CloseHandle(job);
        CHECK(!before.in_job || before.binding() >= -1);
        return;
    }
    const rad::JobMemoryLimit after = rad::os::job_memory_limit();
    CHECK(after.in_job);
    CHECK_EQ(after.job, 1024 * kGiB);
    CHECK_EQ(after.process, 1024 * kGiB);
    CHECK_EQ(after.binding(), 1024 * kGiB);
    CloseHandle(job);   /* the process stays in it; the limits are far above this test's use */
}

RAD_TEST_MAIN()
