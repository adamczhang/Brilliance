/* rank_barrier.h -- the release point for the tensor-parallel rank threads.
 *
 * Radiance is one process with one thread per rank, each owning one device (spec §1), so the step
 * batch is shared by pointer rather than serialised to other processes. The barrier only
 * establishes happens-before between the scheduler's release and each rank's first rad_issue, and
 * between the ranks' last issue and the scheduler's next batch build. A condvar round trip costs
 * ~1-2 us; an IPC broadcast of the batch costs tens of microseconds per step.
 *
 * Generation-based (sense reversal), so a fast rank re-entering for the next step cannot slip
 * through a barrier its slower peers have not left yet.
 */
#pragma once
#include "../rad_core.h"

#include <condition_variable>
#include <mutex>

namespace rad {

class RankBarrier {
public:
    explicit RankBarrier(int n_ranks) : n_(n_ranks < 1 ? 1 : n_ranks), waiting_(0), gen_(0) {}

    RankBarrier(const RankBarrier&) = delete;
    RankBarrier& operator=(const RankBarrier&) = delete;

    /* True when every party arrived and the generation turned; false when abort() released the
     * caller (before or while it waited). Brilliance: the stop journal tells the two apart
     * (Engine::rank_loop); the ranks of one turned generation were all released together. */
    bool arrive_and_wait();

    /* Release everyone regardless of how many have arrived, and make every later arrival return
     * at once. Called on shutdown and on a device fault, so ranks are not left parked on a barrier
     * an exited rank can no longer complete; the engine must be able to exit (spec §17). */
    void abort();

    /* Brilliance: the stop's abort (Engine::serve). Every caller arrives in pairs, a release then
     * a join, so an odd generation means a step was released and its join is still open. Then
     * the abort waits for that join: the join turns as a full turn for every party, and only then
     * do later arrivals return false. With no step open it is abort(). Once a stop begins no new
     * step is released, and every rank released for a step issues it and reaches its join. */
    void close();

private:
    std::mutex              m_;
    std::condition_variable cv_;
    int                     n_;
    int                     waiting_;
    uint64_t                gen_;
    bool                    aborted_ = false;
    bool                    closing_ = false;    /* close() came while a join was open */
    int64_t                 normal_from_ = -1;   /* the generation the last full arrival turned */
};

}  /* namespace rad */
