/* rank_barrier.cpp -- see rank_barrier.h for why this is a condvar and not an IPC handshake. */
#include "rank_barrier.h"

namespace rad {

bool RankBarrier::arrive_and_wait() {
    std::unique_lock<std::mutex> lk(m_);
    if (aborted_) return false;

    const uint64_t my_gen = gen_;
    if (++waiting_ == n_) {
        /* Last one in resets the counter and flips the generation, so a rank that re-enters at
         * once waits on the new generation instead of racing through. */
        waiting_ = 0;
        normal_from_ = (int64_t)gen_;
        ++gen_;
        /* The join close() waited for: it turned for every party, and later arrivals are refused. */
        if (closing_) aborted_ = true;
        lk.unlock();
        cv_.notify_all();
        return true;
    }
    cv_.wait(lk, [&] { return aborted_ || gen_ != my_gen; });
    /* abort() turns the generation too. A waiter whose generation the last arrival turned was
     * released by that turn even when an abort came before it woke: every party was. */
    return normal_from_ == (int64_t)my_gen;
}

void RankBarrier::abort() {
    {
        std::lock_guard<std::mutex> lk(m_);
        aborted_ = true;
        waiting_ = 0;
        ++gen_;
    }
    cv_.notify_all();
}

void RankBarrier::close() {
    {
        std::lock_guard<std::mutex> lk(m_);
        if (aborted_) return;
        if (gen_ & 1) {
            closing_ = true;      /* a step is released and its join is open: let it turn */
            return;
        }
        aborted_ = true;
        waiting_ = 0;
        ++gen_;
    }
    cv_.notify_all();
}

}  /* namespace rad */
