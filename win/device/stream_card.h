/* stream_card.h -- Brilliance: the core's own kernels launch from the card their stream is on.
 *
 * A KERNEL LAUNCH BELONGS TO THE CALLING THREAD'S CURRENT CARD. HIP on Windows refuses a launch onto
 * another card's stream; the AQL backend has no current card at all, so the core never had to say
 * which card a launch comes from. It launches from the thread that has the work: the step loop
 * advances every rank's rows from one thread (sched/advance.cpp), and at --tp 2 the second rank's
 * launch failed on the first step with a device error (2026-11-16, Flash-Next on two R9700s).
 *
 * So each of the core's launchers (sched/advance.hip, runtime/wtab.hip, runtime/xprefetch.hip)
 * holds one of these across its launch: the thread moves to the stream's card when it is elsewhere
 * and goes back after. Plugins launch from their rank's own thread, which is already on its card.
 * Brought over from Brilliance 0.2.2; Windows only (win/), included by the launchers under _WIN32. */
#ifndef BRL_STREAM_CARD_H
#define BRL_STREAM_CARD_H

#include <hip/hip_runtime.h>

namespace rad {

struct OnStreamCard {
    int prev = -1;
    explicit OnStreamCard(void* stream) {
        if (!stream) return;   /* the null stream is the current card's by definition */
        int cur = -1, card = -1;
        if (hipStreamGetDevice((hipStream_t)stream, &card) != hipSuccess ||
            hipGetDevice(&cur) != hipSuccess) {
            (void)hipGetLastError();   /* the launch reports what is wrong with the stream */
            return;
        }
        if (card == cur) return;
        if (hipSetDevice(card) == hipSuccess) prev = cur;
        else (void)hipGetLastError();
    }
    /* A failed restore leaves this thread on the stream's card, which the next launch from it then
     * moves away from again; nothing the thread does depends on the card it was on. */
    ~OnStreamCard() {
        if (prev >= 0 && hipSetDevice(prev) != hipSuccess) (void)hipGetLastError();
    }
    OnStreamCard(const OnStreamCard&) = delete;
    OnStreamCard& operator=(const OnStreamCard&) = delete;
};

}  /* namespace rad */

#endif  /* BRL_STREAM_CARD_H */
