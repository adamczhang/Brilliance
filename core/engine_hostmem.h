/* engine_hostmem.h -- the memory cgroup limits bring-up checks before it locks a table into host
 * memory (engine_bringup.cpp, lockable_room). */
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rad {

/* Brilliance: one part of the host memory a process will still pin or allocate when a table asks
 * for its own (lockable_room's `still`): what it is, the setting that sizes it ("" when no one
 * setting does), the fix to name when it is what took the room ("" for none), and its bytes. A
 * refusal lists the parts, so it can name the setting that used the room rather than only the
 * table that asked last. */
struct HostPart {
    std::string what;     /* "the prefix cache's host tier" */
    std::string setting;  /* "--prefix-cache-mib 20480" */
    std::string fix;      /* "lower --prefix-cache-mib" */
    int64_t     bytes = 0;
};

/* "what, setting: size; ..." for the parts with bytes, largest first ("what: size" without a
 * setting); "" when none has any. */
std::string host_parts_text(const std::vector<HostPart>& parts);

/* The part with the most bytes (the first of equals), or nullptr when none has any. */
const HostPart* host_parts_largest(const std::vector<HostPart>& parts);

/* A cgroup's limit less the usage reclaim cannot give back: memory.current less file pages (v2)
 * or usage_in_bytes less cache (v1), since page cache yields to a locked allocation and anonymous
 * memory does not. "max" and v1's near-INT64_MAX sentinel mean no limit. True, with `room` set,
 * when `dir` has a limit. */
bool cgroup_room(const std::string& dir, bool v2, int64_t* room);

/* Every level of the memory cgroup at `path` (as /proc/self/cgroup names it), from that level up
 * to the hierarchy's root mounted at `root`, leaf first: `visit(dir, room)` for each level with a
 * limit, until it returns false. Each level is a limit of its own. */
void cgroup_levels(const std::string& root, std::string path, bool v2,
                   const std::function<bool(const std::string& dir, int64_t room)>& visit);

/* Every level of every memory cgroup the process is in, leaf first: `visit(dir, room)` until it
 * returns false (cgroup_levels). /proc/self/cgroup reads "0::/path" on v2 and "N:memory:/path"
 * (or a comma list naming memory) on v1; each level up to the visible root is a limit of its
 * own. */
void memory_cgroups(const std::function<bool(const std::string& dir, int64_t room)>& visit);

/* The least room any of those levels leaves (cgroup_room), or INT64_MAX when none sets a limit:
 * a container's --memory, which the OOM killer enforces whatever MemAvailable says. */
int64_t cgroup_least_room();

/* Whether `need` more bytes fit in host memory after `still` more this process will allocate
 * first and `margin` (--host-headroom-mib) stays free, or why not. Checks MemAvailable; every
 * memory cgroup from the process's own up to the visible root (Docker's --memory, enforced by the
 * OOM killer); and, for an mlock()ed copy (`mlocks`), RLIMIT_MEMLOCK unless the process has
 * CAP_IPC_LOCK. Pinned memory a card reads is a driver userptr mapping, which RLIMIT_MEMLOCK does
 * not count. An unreadable limit is ignored, never guessed. With `parts` (Brilliance), `why`
 * says what `still` is made of (host_parts_text). */
bool lockable_room(int64_t need, int64_t still, int64_t margin, bool mlocks, std::string* why,
                   const std::vector<HostPart>* parts = nullptr);

}  /* namespace rad */
