/* decision_e2e.h -- the radiance binary on a free loopback port and plain HTTP calls to it, for
 * the decision models' end-to-end tests (clef_decision_test.cpp, laya_decision_test.cpp).
 *
 * The server runs on the host backend with the reference kernels, as engine_e2e_test.cpp runs one.
 * RAD_E2E_SERVER and RAD_E2E_HOME come from tests/CMakeLists.txt.
 */
#pragma once

#include "rad_test.h"
#include "e2e_config.h"
#include "tiny_model.h"

#include "util/json.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

#if defined(_WIN32) && !defined(MSG_NOSIGNAL)
/* Brilliance: Windows has no SIGPIPE to suppress; a send on a closed connection just fails. */
#define MSG_NOSIGNAL 0
#endif

namespace decision_e2e {

using json = rad::Json;   /* the request order, which Laya reads */

#if defined(_WIN32)
/* Brilliance: the Winsock side of Call.
 *   - Winsock is started before the first socket. These binaries run the server in a child and
 *     link no client that would start it, so the first call does, once (WSAStartup counts, so it
 *     is harmless when something else already has).
 *   - SOCK_CLOEXEC: a socket() handle is inheritable by default on Windows, and
 *     WSA_FLAG_NO_HANDLE_INHERIT is the same promise -- a server started while a call is open cannot
 *     keep its connection open. (rad::os::Captured also hands its child an explicit handle list.)
 *   - The SOCKET lives in Call's int, as the descriptor does on Linux: a socket is a kernel handle,
 *     whose value 64-bit Windows keeps within 32 bits, and INVALID_SOCKET reads as -1 there. It is
 *     closed with closesocket, and its errors are WSAGetLastError()'s, not errno's. */
inline int win_socket() {
    static const int wsa = [] { WSADATA d; return WSAStartup(MAKEWORD(2, 2), &d); }();
    (void)wsa;   /* a failed start shows as WSANOTINITIALISED from the socket itself */
    return (int)WSASocketW(AF_INET, SOCK_STREAM, IPPROTO_TCP, nullptr, 0,
                           WSA_FLAG_OVERLAPPED | WSA_FLAG_NO_HANDLE_INHERIT);
}
inline void win_close(int fd) { ::closesocket((SOCKET)(intptr_t)fd); }
inline std::string win_sock_error() { return rad::os::win32_message((unsigned long)WSAGetLastError()); }
#endif

inline std::vector<std::string> engine_env() {
    return { "ROCR_VISIBLE_DEVICES=99", "RADIANCE_HOST_VRAM_MIB=1024", "OMP_WAIT_POLICY=passive",
             "OMP_NUM_THREADS=2", std::string("RADIANCE_HOME=") + RAD_E2E_HOME };
}

inline std::vector<std::string> engine_argv(const std::string& model,
                                     const std::vector<std::string>& args) {
    std::vector<std::string> a = { RAD_E2E_SERVER, "--model", model, "--kernels", "libref",
                                   "--debug-accept-reference-kernels" };
    a.insert(a.end(), args.begin(), args.end());
    return radtest::config_argv(a);
}

inline std::string tail(const std::string& log, int n = 14) {
    size_t at = log.size();
    for (int i = 0; i <= n && at != std::string::npos && at > 0; ++i) at = log.rfind('\n', at - 1);
    return at == std::string::npos || at >= log.size() ? log : log.substr(at + 1);
}

/* A server on a free loopback port (as engine_e2e_test.cpp runs one). */
struct Server {
    tiny::Child child;
    std::thread drain;
    std::mutex mu;
    std::condition_variable cv;
    std::string log;
    bool eof = false;
    int port = 0;

    bool start(const std::string& model, const std::vector<std::string>& args) {
        std::vector<std::string> a = { "--host", "127.0.0.1", "--port", "0" };
        a.insert(a.end(), args.begin(), args.end());
        if (!child.start(engine_argv(model, a), engine_env())) return false;
        drain = std::thread([this] {
            char buf[4096];
            for (;;) {
#if defined(_WIN32)
                /* Brilliance: the child's output is a pipe HANDLE, read through
                 * rad::os::Captured::read_some: 0 at its end, as read(2) is, and never EINTR. */
                const ssize_t n = (ssize_t)child.c.read_some(buf, sizeof buf);
#else
                const ssize_t n = ::read(child.fd, buf, sizeof buf);
#endif
                if (n < 0 && errno == EINTR) continue;
                std::lock_guard<std::mutex> lk(mu);
                if (n <= 0) { eof = true; cv.notify_all(); return; }
                log.append(buf, (size_t)n);
                cv.notify_all();
            }
        });
        static const std::string kTag = "listening on http://127.0.0.1:";
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return eof || log.find(kTag) != std::string::npos; });
        const size_t at = log.find(kTag);
        if (at != std::string::npos) port = std::atoi(log.c_str() + at + kTag.size());
        if (port <= 0) fprintf(stderr, "    the server did not start:\n%s\n", tail(log).c_str());
        return port > 0;
    }
    int stop() {
#if defined(_WIN32)
        /* Brilliance: SIGTERM is Ctrl-Break to the server's own process group
         * (rad::os::Captured::interrupt), which the engine's on_terminate handler takes as
         * SIGTERM: it drains and exits with its own status, as on Linux. A test process with no
         * console to share with the server cannot send it. That fails the case, naming why, and
         * the server is ended hard -- never a kill passed off as the drain. */
        if (child.c.pid() <= 0) return -1;
        std::string why;
        if (!child.c.interrupt(&why)) {
            ::radtest::fail(__FILE__, __LINE__, "no graceful stop for the server: " + why +
                                                "; it is killed instead");
            child.c.kill();
        }
        if (drain.joinable()) drain.join();
        child.c.close_read();
#else
        if (child.pid <= 0) return -1;
        kill(child.pid, SIGTERM);
        if (drain.joinable()) drain.join();
        close(child.fd);
        child.fd = -1;
#endif
        std::string rest;
        return child.wait(&rest);
    }
    std::string output() {
        std::lock_guard<std::mutex> lk(mu);
        return log;
    }
    ~Server() {
#if defined(_WIN32)
        if (child.c.pid() > 0) child.c.kill();   /* Brilliance: SIGKILL is TerminateProcess */
#else
        if (child.pid > 0) kill(child.pid, SIGKILL);
#endif
        if (drain.joinable()) drain.join();
    }
};

/* One request on its own loopback connection. */
struct Call {
    int fd = -1;
    std::string why;
    bool fail(const char* what) {
#if defined(_WIN32)
        why = std::string(what) + ": " + win_sock_error();   /* Brilliance: Winsock's error */
#else
        why = std::string(what) + ": " + std::strerror(errno);
#endif
        return false;
    }
    bool send(int port, const char* method, const std::string& path, const std::string& body) {
#if defined(_WIN32)
        fd = win_socket();   /* Brilliance: SOCK_CLOEXEC is WSA_FLAG_NO_HANDLE_INHERIT (win_socket) */
#else
        fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
#endif
        if (fd < 0) return fail("socket");
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t)port);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(fd, (sockaddr*)&a, sizeof a) != 0) return fail("connect");
        std::string req = std::string(method) + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n" +
                          "Connection: close\r\nContent-Type: application/json\r\n" +
                          "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
        for (size_t off = 0; off < req.size();) {
            const ssize_t n = ::send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
            if (n <= 0) return fail("send");
            off += (size_t)n;
        }
        return true;
    }
    int read(std::string* body) {
        std::string raw;
        char buf[16384];
        for (;;) {
#if defined(_WIN32)
            /* Brilliance: a blocking Winsock recv is never interrupted, and its errors are not in
             * errno -- a stale EINTR there would turn a failed recv into an endless loop. */
            const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
#else
            const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
            if (n < 0 && errno == EINTR) continue;
#endif
            if (n <= 0) break;
            raw.append(buf, (size_t)n);
        }
#if defined(_WIN32)
        win_close(fd);   /* Brilliance: a SOCKET is closed with closesocket */
#else
        close(fd);
#endif
        fd = -1;
        const size_t h = raw.find("\r\n\r\n");
        if (raw.compare(0, 9, "HTTP/1.1 ") != 0 || h == std::string::npos) {
            why = "no response (" + std::to_string(raw.size()) + " bytes)";
            return -1;
        }
        std::string head = raw.substr(0, h), rest = raw.substr(h + 4);
        for (char& c : head) c = (char)std::tolower((unsigned char)c);
        if (head.find("transfer-encoding: chunked") != std::string::npos) {
            std::string out;
            for (size_t at = 0;;) {
                const size_t eol = rest.find("\r\n", at);
                if (eol == std::string::npos) break;
                const size_t n = std::strtoul(rest.c_str() + at, nullptr, 16);
                if (n == 0) break;
                out += rest.substr(eol + 2, n);
                at = eol + 2 + n + 2;
            }
            rest = out;
        }
        *body = rest;
        return std::atoi(raw.c_str() + 9);
    }
#if defined(_WIN32)
    ~Call() { if (fd >= 0) win_close(fd); }   /* Brilliance: closesocket */
#else
    ~Call() { if (fd >= 0) close(fd); }
#endif
};

inline json call(int port, const char* method, const std::string& path, const json& body, int* status) {
    Call c;
    std::string out;
    const int st = c.send(port, method, path, body.is_null() ? "" : body.dump()) ? c.read(&out) : -1;
    *status = st;
    if (st < 0) fprintf(stderr, "    %s %s: %s\n", method, path.c_str(), c.why.c_str());
    return json::parse(out, nullptr, false);
}

inline uint64_t fnv1a(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : bytes) h = (h ^ c) * 0x100000001b3ull;
    return h;
}

inline std::string hex16(uint64_t v) {
    char s[24];
    std::snprintf(s, sizeof s, "%016llx", (unsigned long long)v);
    return s;
}

}  /* namespace decision_e2e */
