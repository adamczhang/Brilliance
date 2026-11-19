/* Brilliance: radiance as a Windows service (win/os/service.h), tested without the Service Control
 * Manager, which an unelevated shell cannot use:
 *   - the command line, the log's name and rotation, and the secret files;
 *   - the control handler and the stop pump, driven directly with fake SCM hooks: STOP,
 *     PRESHUTDOWN and SHUTDOWN start one drain, STOP_PENDING's checkpoint moves, an overdue drain is
 *     reported and never ended, and the exit status maps as the recovery actions need;
 *   - os::request_terminate keeping a stop that comes before the engine registers;
 *   - the real radiance.exe started with --service from a console, where it runs as the service
 *     would: a fleet with no model loaded (nothing to read, no card -- HIP_VISIBLE_DEVICES=-1),
 *     its key read from a file, stopped through the service's own STOP path and through Ctrl-Break.
 * win/SERVICE.md has the qualification that needs an elevated shell. */
#include "rad_test.h"

#include "os/os.h"
#include "os/service.h"

#include <winsock2.h>
#include <windows.h>

#include <chrono>
#include <csignal>
#include <fstream>
#include <sstream>
#include <thread>

#ifndef RAD_SERVICE_TEST_RADIANCE
#error "RAD_SERVICE_TEST_RADIANCE names the radiance.exe under test (win/tests/CMakeLists.txt)"
#endif

namespace svc = rad::os::service;

namespace {

/* ------------------------------------------------------------------ helpers */
std::vector<char*> argv_of(std::vector<std::string>& a) {
    std::vector<char*> v;
    for (std::string& s : a) v.push_back(s.data());
    v.push_back(nullptr);
    return v;
}

bool parse(std::vector<std::string> a, svc::Args* out, std::string* why) {
    a.insert(a.begin(), "radiance.exe");
    std::vector<char*> v = argv_of(a);
    return svc::parse((int)a.size(), v.data(), out, why);
}

std::string refusal(std::vector<std::string> a) {
    svc::Args out;
    std::string why;
    if (parse(std::move(a), &out, &why)) return "(accepted)";
    return why;
}

std::string win_path(std::string p) {
    for (char& c : p)
        if (c == '/') c = '\\';
    return p;
}

void write_file(const std::string& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

/* The fake SCM: every status reported, every terminate asked, every log line. */
struct Fake {
    std::vector<svc::Status> reports;
    std::vector<int>         terminates;
    std::vector<std::string> lines;
    int                      pumps_started = 0;
    bool                     terminate_result = true;

    svc::Control::Hooks hooks() {
        return {[this](const svc::Status& s) { reports.push_back(s); },
                [this](int sig) { terminates.push_back(sig); return terminate_result; },
                [this](const std::string& l) { lines.push_back(l); },
                [this] { ++pumps_started; }};
    }
    bool logged(const std::string& part) const {
        for (const std::string& l : lines)
            if (contains(l, part)) return true;
        return false;
    }
};

/* ------------------------------------------------------------------ a child radiance */
std::vector<std::string> child_env(const std::vector<std::string>& set, const std::vector<std::string>& drop) {
    std::vector<std::string> env;
    if (wchar_t* block = GetEnvironmentStringsW()) {
        for (const wchar_t* p = block; *p; p += wcslen(p) + 1) {
            const std::string kv = rad::os::narrow(p);
            if (kv.empty() || kv[0] == '=') continue;   // the per-drive current directories
            const std::string name = kv.substr(0, kv.find('='));
            bool skip = false;
            for (const std::string& s : set) skip |= _stricmp(s.substr(0, s.find('=')).c_str(), name.c_str()) == 0;
            for (const std::string& d : drop) skip |= _stricmp(d.c_str(), name.c_str()) == 0;
            if (!skip) env.push_back(kv);
        }
        FreeEnvironmentStringsW(block);
    }
    env.insert(env.end(), set.begin(), set.end());
    return env;
}

/* Waits up to `ms` for `pid` to exit. */
bool exited_within(long pid, int64_t ms) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (rad::os::process_alive(pid)) {
        if (std::chrono::steady_clock::now() > until) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return true;
}

/* Waits up to `ms` for `path` to contain `what`. */
bool file_says(const std::string& path, const std::string& what, int64_t ms) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    for (;;) {
        if (contains(read_file(path), what)) return true;
        if (std::chrono::steady_clock::now() > until) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

/* One GET on its own loopback connection; the HTTP status, or -1 with why in *body. */
int http_get(int port, const std::string& path, const std::string& key, std::string* body) {
    static const int wsa = [] { WSADATA d; return WSAStartup(MAKEWORD(2, 2), &d); }();
    (void)wsa;
    SOCKET s = WSASocketW(AF_INET, SOCK_STREAM, IPPROTO_TCP, nullptr, 0, WSA_FLAG_NO_HANDLE_INHERIT);
    if (s == INVALID_SOCKET) { *body = "socket: " + rad::os::win32_message(WSAGetLastError()); return -1; }
    DWORD tmo = 10000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tmo, sizeof tmo);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((u_short)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(s, (sockaddr*)&a, sizeof a) != 0) {
        *body = "connect: " + rad::os::win32_message(WSAGetLastError());
        closesocket(s);
        return -1;
    }
    std::string req = "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n";
    if (!key.empty()) req += "Authorization: Bearer " + key + "\r\n";
    req += "\r\n";
    send(s, req.data(), (int)req.size(), 0);
    std::string resp;
    char buf[4096];
    for (int n; (n = recv(s, buf, sizeof buf, 0)) > 0;) resp.append(buf, (size_t)n);
    closesocket(s);
    const size_t sp = resp.find(' ');
    if (resp.compare(0, 5, "HTTP/") != 0 || sp == std::string::npos) { *body = "no HTTP response: " + resp; return -1; }
    const size_t hdr_end = resp.find("\r\n\r\n");
    *body = hdr_end == std::string::npos ? "" : resp.substr(hdr_end + 4);
    return std::atoi(resp.c_str() + sp + 1);
}

/* radiance --service NAME ... from this process's console; its first output (the banner, before the
 * log takes over) collected by a drain thread. */
struct Child {
    rad::os::Captured c;
    std::thread       drain;
    std::string       out;
    long              pid = -1;

    bool start(const std::vector<std::string>& argv, const std::vector<std::string>& env) {
        if (!c.start(argv, env)) return false;
        pid = c.pid();
        drain = std::thread([this] {
            char buf[4096];
            for (int64_t n; (n = c.read_some(buf, sizeof buf)) > 0;) out.append(buf, (size_t)n);
        });
        return true;
    }
    /* The exit status, after at most `ms`. A child still running then is asked again with
     * Ctrl-Break (the drain), and fails the case; it is never killed here. */
    int finish(int64_t ms) {
        if (!exited_within(pid, ms)) {
            ::radtest::fail(__FILE__, __LINE__, "the service did not exit in time; Ctrl-Break sent");
            std::string why;
            c.interrupt(&why);
            exited_within(pid, 60000);
        }
        if (drain.joinable()) drain.join();
        c.close_read();
        return c.wait();
    }
    ~Child() {
        if (drain.joinable()) {
            if (pid > 0 && rad::os::process_alive(pid)) c.interrupt();
            drain.join();
        }
    }
};

std::string fleet_yaml(int port, const std::string& key_env) {
    // Two models, neither pinned and no warm-up: nothing loads, and the server listens at once.
    std::string y = "server:\n  host: 127.0.0.1\n  port: " + std::to_string(port) + "\n";
    if (!key_env.empty()) y += "  api-key-env: " + key_env + "\n";
    y += "scheduler:\n  warm-up: false\n"
         "models:\n  a:\n    model: C:/brilliance-service-test/nonexistent/a.rad\n"
         "  b:\n    model: C:/brilliance-service-test/nonexistent/b.rad\n";
    return y;
}

}   // namespace

/* ------------------------------------------------------------------ the command line */
TEST(service_wanted_anywhere) {
    std::vector<std::string> a = {"radiance.exe", "--config", "x.yaml", "--service", "S"};
    std::vector<char*> v = argv_of(a);
    CHECK(svc::wanted((int)a.size(), v.data()));
    std::vector<std::string> b = {"radiance.exe", "--config", "x.yaml"};
    std::vector<char*> w = argv_of(b);
    CHECK(!svc::wanted((int)b.size(), w.data()));
    CHECK(!svc::wanted(1, w.data()));
}

TEST(service_parse_minimal) {
    svc::Args a;
    std::string why;
    REQUIRE(parse({"--service", "Brilliance-q27", "--config", "C:\\svc dir\\q27.yaml"}, &a, &why));
    CHECK_EQ(a.name, "Brilliance-q27");
    CHECK_EQ(a.config, "C:\\svc dir\\q27.yaml");
    CHECK_EQ(a.log, "C:\\svc dir\\q27.log");
    CHECK(a.env_files.empty());
    CHECK_EQ(a.stop_timeout_ms, 180000);
}

TEST(service_parse_everything) {
    svc::Args a;
    std::string why;
    REQUIRE(parse({"--config", "C:/cfg/q.yaml", "--service", "S", "--log", "D:\\logs\\s.log",
                   "--env-from-file", "RADIANCE_API_KEY=C:\\keys\\s.key", "--env-from-file", "OTHER=C:\\k2",
                   "--stop-timeout", "45"},
                  &a, &why));
    CHECK_EQ(a.config, "C:\\cfg\\q.yaml");   // made absolute and normalised
    CHECK_EQ(a.log, "D:\\logs\\s.log");
    REQUIRE_EQ(a.env_files.size(), 2u);
    CHECK_EQ(a.env_files[0].first, "RADIANCE_API_KEY");
    CHECK_EQ(a.env_files[0].second, "C:\\keys\\s.key");
    CHECK_EQ(a.env_files[1].first, "OTHER");
    CHECK_EQ(a.stop_timeout_ms, 45000);
}

TEST(service_parse_relative_config_is_made_absolute) {
    svc::Args a;
    std::string why;
    REQUIRE(parse({"--service", "S", "--config", "rel\\q.yaml"}, &a, &why));
    CHECK_EQ(a.config, rad::os::full_path("rel\\q.yaml"));
    CHECK(a.config.size() > 2 && a.config[1] == ':');
}

TEST(service_parse_refusals) {
    CHECK(contains(refusal({"--config", "c.yaml"}), "--service NAME is missing"));
    CHECK(contains(refusal({"--service", "S"}), "--config FILE is missing"));
    CHECK(contains(refusal({"--service", "S", "--config"}), "--config needs a value"));
    CHECK(contains(refusal({"--service", "S", "--config", ""}), "--config needs a value"));
    CHECK(contains(refusal({"--service"}), "--service needs a value"));
    CHECK(contains(refusal({"--service", "S", "--service", "T", "--config", "c"}), "--service is given twice"));
    CHECK(contains(refusal({"--service", "S", "--config", "c", "--config", "d"}), "--config is given twice"));
    CHECK(contains(refusal({"--service", "S", "--config", "c", "--log", "a", "--log", "b"}), "--log is given twice"));
    CHECK(contains(refusal({"--service", "a\\b", "--config", "c"}), "a service name has no"));
    // radiance's own flags and the old flag-style command line do not belong in a service
    CHECK(contains(refusal({"--service", "S", "--config", "c", "--check"}), "'--check' is not a service option"));
    CHECK(contains(refusal({"--service", "S", "--config", "c", "--help"}), "'--help' is not a service option"));
    CHECK(contains(refusal({"--service", "S", "--model", "m.rad"}), "'--model' is not a service option"));
    // a key itself is not a service option: it would be on the command line every user can read
    CHECK(contains(refusal({"--service", "S", "--config", "c", "--api-key", "k"}), "'--api-key' is not a service option"));
    for (const char* bad : {"VAR", "=C:\\f", "VAR="})
        CHECK(contains(refusal({"--service", "S", "--config", "c", "--env-from-file", bad}), "takes VAR=FILE"));
    for (const char* bad : {"0", "-5", "x", "10s", "86401", ""})
        CHECK(!refusal({"--service", "S", "--config", "c", "--stop-timeout", bad}).empty() &&
              refusal({"--service", "S", "--config", "c", "--stop-timeout", bad}) != "(accepted)");
    CHECK_EQ(refusal({"--service", "S", "--config", "c", "--stop-timeout", "86400"}), "(accepted)");
}

TEST(service_default_log_beside_the_config) {
    CHECK_EQ(svc::default_log("C:\\ProgramData\\Brilliance\\services\\S\\S.yaml"),
             "C:\\ProgramData\\Brilliance\\services\\S\\S.log");
    CHECK_EQ(svc::default_log("C:\\a.b\\svc"), "C:\\a.b\\svc.log");
    CHECK_EQ(svc::default_log("C:\\cfg\\q27.v4.yaml"), "C:\\cfg\\q27.v4.log");
    CHECK_EQ(svc::default_log("C:\\cfg\\.hidden"), "C:\\cfg\\.hidden.log");
}

/* ------------------------------------------------------------------ files */
TEST(service_value_file) {
    const std::string dir = radtest::tmpdir();
    std::string v, why;
    write_file(dir + "/key1", "\xEF\xBB\xBFsk-local-0123\r\n");
    REQUIRE(svc::read_value_file(dir + "/key1", &v, &why));
    CHECK_EQ(v, "sk-local-0123");
    write_file(dir + "/key2", "plain-key");
    REQUIRE(svc::read_value_file(dir + "/key2", &v, &why));
    CHECK_EQ(v, "plain-key");
    write_file(dir + "/empty", "");
    CHECK(!svc::read_value_file(dir + "/empty", &v, &why) && contains(why, "is empty"));
    write_file(dir + "/blank", " \r\n\t\n");
    CHECK(!svc::read_value_file(dir + "/blank", &v, &why) && contains(why, "is empty"));
    write_file(dir + "/two", "key\r\nsecond line\r\n");
    CHECK(!svc::read_value_file(dir + "/two", &v, &why) && contains(why, "more than one line"));
    write_file(dir + "/big", std::string(4096, 'k'));
    CHECK(!svc::read_value_file(dir + "/big", &v, &why) && contains(why, "4 KiB"));
    CHECK(!svc::read_value_file(dir + "/absent", &v, &why) && contains(why, "cannot read"));
}

TEST(service_log_rotation) {
    const std::string log = radtest::tmp_path("rot.log");
    svc::rotate_log(log);   // nothing there: nothing happens
    CHECK(!rad::os::is_file(log) && !rad::os::is_file(log + ".1"));
    write_file(log, "first run\n");
    svc::rotate_log(log);
    CHECK(!rad::os::is_file(log));
    CHECK_EQ(read_file(log + ".1"), "first run\n");
    write_file(log, "second run\n");
    svc::rotate_log(log);
    CHECK_EQ(read_file(log + ".1"), "second run\n");   // the older .1 replaced
}

/* ------------------------------------------------------------------ the control handler */
TEST(service_control_start_and_run) {
    Fake f;
    svc::Control c(f.hooks(), 180000, 4242);
    c.starting();
    REQUIRE_EQ(f.reports.size(), 1u);
    CHECK_EQ(f.reports[0].state, svc::kStartPending);
    CHECK_EQ(f.reports[0].accepted, 0u);
    CHECK(f.reports[0].checkpoint > 0);
    CHECK(f.reports[0].wait_hint_ms > 0);
    c.running();
    REQUIRE_EQ(f.reports.size(), 2u);
    CHECK_EQ(f.reports[1].state, svc::kRunning);
    CHECK_EQ(f.reports[1].accepted, svc::kAcceptStop | svc::kAcceptPreshutdown | svc::kAcceptShutdown);
    CHECK_EQ(f.reports[1].checkpoint, 0u);
    // INTERROGATE needs nothing; a control the service does not take is refused as the SCM expects
    CHECK_EQ(c.control(svc::kControlInterrogate), svc::kNoError);
    CHECK_EQ(c.control(0x2 /* PAUSE */), svc::kErrorCallNotImplemented);
    CHECK_EQ(c.control(0x80 /* user-defined */), svc::kErrorCallNotImplemented);
    CHECK_EQ(f.reports.size(), 2u);
    CHECK(f.terminates.empty());
    CHECK(!c.stop_asked());
}

TEST(service_control_each_stop_is_the_one_drain) {
    for (uint32_t first : {svc::kControlStop, svc::kControlPreshutdown, svc::kControlShutdown}) {
        Fake f;
        svc::Control c(f.hooks(), 180000, 4242);
        c.starting();
        c.running();
        CHECK_EQ(c.control(first), svc::kNoError);
        CHECK(c.stop_asked());
        REQUIRE_EQ(f.terminates.size(), 1u);
        CHECK_EQ(f.terminates[0], SIGTERM);   // what Ctrl-Break gives the engine in a console
        CHECK_EQ(f.pumps_started, 1);
        CHECK_EQ(f.reports.back().state, svc::kStopPending);
        CHECK_EQ(f.reports.back().accepted, 0u);
        CHECK_EQ(f.reports.back().wait_hint_ms, (uint32_t)svc::Control::kWaitHintMs);
        CHECK(f.logged("draining"));
        // Another stop, however it comes, changes nothing: one drain, one pump.
        const size_t n = f.reports.size();
        for (uint32_t again : {svc::kControlStop, svc::kControlPreshutdown, svc::kControlShutdown})
            CHECK_EQ(c.control(again), svc::kNoError);
        CHECK_EQ(f.terminates.size(), 1u);
        CHECK_EQ(f.pumps_started, 1);
        CHECK_EQ(f.reports.size(), n);
        // RUNNING is not reported over a stop that has begun
        c.running();
        CHECK_EQ(f.reports.size(), n);
    }
}

TEST(service_control_pump_moves_the_checkpoint_until_stopped) {
    Fake f;
    svc::Control c(f.hooks(), 180000, 4242);
    c.starting();
    c.running();
    c.control(svc::kControlStop);
    uint32_t last = f.reports.back().checkpoint;
    for (int64_t t = svc::Control::kPumpMs; t <= 20 * svc::Control::kPumpMs; t += svc::Control::kPumpMs) {
        CHECK(c.pump(t));
        CHECK_EQ(f.reports.back().state, svc::kStopPending);
        CHECK(f.reports.back().checkpoint > last);   // the SCM sees progress every pump
        CHECK(f.reports.back().wait_hint_ms > svc::Control::kPumpMs);
        last = f.reports.back().checkpoint;
    }
    CHECK(f.logged("still draining, 30 s after the stop"));
    c.stopped(0);
    CHECK_EQ(f.reports.back().state, svc::kStopped);
    CHECK_EQ(f.reports.back().exit_code, svc::kNoError);
    CHECK_EQ(f.reports.back().checkpoint, 0u);
    CHECK_EQ(f.lines.back(), "stopped");
    // after STOPPED: the pump ends, and nothing more is reported or asked
    const size_t n = f.reports.size();
    CHECK(!c.pump(60000));
    CHECK_EQ(c.control(svc::kControlStop), svc::kNoError);
    c.running();
    c.stopped(1);
    CHECK_EQ(f.reports.size(), n);
    CHECK_EQ(f.terminates.size(), 1u);
}

TEST(service_control_overdue_drain_is_reported_not_ended) {
    Fake f;
    svc::Control c(f.hooks(), 10000, 4242);
    c.starting();
    c.running();
    c.control(svc::kControlStop);
    CHECK(c.pump(8000));
    const svc::Status before = f.reports.back();
    const size_t n = f.reports.size();
    CHECK(c.pump(10000));   // the timeout: reported once, and the checkpoint stops moving
    CHECK_EQ(f.reports.size(), n);
    CHECK(f.logged("NOT terminated"));
    CHECK(f.logged("brl-stacks 4242"));
    const size_t lines = f.lines.size();
    CHECK(c.pump(30000));   // no reminder before a minute has passed
    CHECK_EQ(f.lines.size(), lines);
    CHECK(c.pump(70000));
    CHECK(f.logged("still not stopped, 70 s after the stop; not terminated"));
    CHECK(c.pump(600000));   // the pump goes on for as long as the engine does
    CHECK_EQ(f.reports.size(), n);
    CHECK_EQ(f.reports.back().checkpoint, before.checkpoint);
    CHECK_EQ(f.terminates.size(), 1u);   // asked once; nothing harder, ever
    c.stopped(0);   // the engine came back after all
    CHECK_EQ(f.reports.back().state, svc::kStopped);
    CHECK_EQ(f.reports.back().exit_code, svc::kNoError);
}

TEST(service_control_exit_status) {
    {   // a failure nobody asked for: the recovery actions see it
        Fake f;
        svc::Control c(f.hooks(), 180000, 1);
        c.starting();
        c.running();
        c.stopped(1);
        CHECK_EQ(f.reports.back().state, svc::kStopped);
        CHECK_EQ(f.reports.back().exit_code, svc::kErrorServiceSpecific);
        CHECK_EQ(f.reports.back().specific, 1u);
        CHECK(f.logged("reported as a failure"));
    }
    {   // a bad end to a requested stop: stopped, not restarted
        Fake f;
        svc::Control c(f.hooks(), 180000, 1);
        c.starting();
        c.running();
        c.control(svc::kControlStop);
        c.stopped(3);
        CHECK_EQ(f.reports.back().state, svc::kStopped);
        CHECK_EQ(f.reports.back().exit_code, svc::kNoError);
        CHECK_EQ(f.reports.back().specific, 3u);
        CHECK(f.logged("do not restart it"));
    }
    {   // stopped before RUNNING (the log could not be opened)
        Fake f;
        svc::Control c(f.hooks(), 180000, 1);
        c.starting();
        c.stopped(3);
        CHECK_EQ(f.reports.size(), 2u);
        CHECK_EQ(f.reports.back().exit_code, svc::kErrorServiceSpecific);
        CHECK_EQ(f.reports.back().specific, 3u);
    }
}

TEST(service_control_stop_before_the_engine_registers) {
    Fake f;
    f.terminate_result = false;   // os::request_terminate: kept for the engine
    svc::Control c(f.hooks(), 180000, 1);
    c.starting();
    c.running();
    CHECK_EQ(c.control(svc::kControlStop), svc::kNoError);
    CHECK_EQ(f.terminates.size(), 1u);
    CHECK(f.logged("still starting"));
    CHECK_EQ(f.reports.back().state, svc::kStopPending);
    CHECK_EQ(f.pumps_started, 1);
}

/* ------------------------------------------------------------------ the real radiance.exe */
TEST(service_console_run_serves_and_stops_through_the_service_path) {
    const std::string dir = win_path(radtest::tmp_path("svc-a"));
    rad::os::make_dirs(dir);
    const int port = 18361;
    write_file(dir + "\\svc.yaml", fleet_yaml(port, "BRL_SERVICE_TEST_KEY"));
    write_file(dir + "\\key.txt", "brl-test-key-5e1f\r\n");
    write_file(dir + "\\svc.log", "the previous run's log\n");

    Child ch;
    REQUIRE(ch.start({RAD_SERVICE_TEST_RADIANCE, "--service", "brl-service-test", "--config", dir + "\\svc.yaml",
                      "--env-from-file", "BRL_SERVICE_TEST_KEY=" + dir + "\\key.txt", "--stop-timeout", "60"},
                     child_env({"HIP_VISIBLE_DEVICES=-1", "OMP_NUM_THREADS=2"}, {"BRL_SERVICE_TEST_KEY"})));
    const std::string log = dir + "\\svc.log";
    const std::string tag = "listening on http://127.0.0.1:" + std::to_string(port);
    if (!file_says(log, tag, 60000)) {
        ::radtest::fail(__FILE__, __LINE__, "the service did not listen:\n" + read_file(log) + "\n" + ch.out);
        ch.finish(60000);
        return;
    }
    std::string body;
    CHECK_EQ(http_get(port, "/health", "", &body), 200);
    CHECK(contains(body, "\"status\""));
    // the key the YAML's api-key-env names came from the file, so a request needs it
    CHECK_EQ(http_get(port, "/v1/models", "", &body), 401);
    CHECK_EQ(http_get(port, "/v1/models", "wrong-key", &body), 401);
    CHECK_EQ(http_get(port, "/v1/models", "brl-test-key-5e1f", &body), 200);

    // the SCM's STOP, through the console's stand-in for it
    const std::wstring ev = L"Local\\brilliance-service-stop-" + std::to_wstring(ch.pid);
    HANDLE h = OpenEventW(EVENT_MODIFY_STATE, FALSE, ev.c_str());
    REQUIRE(h != nullptr);
    CHECK(SetEvent(h));
    CloseHandle(h);
    CHECK_EQ(ch.finish(60000), 0);

    const std::string text = read_file(log);
    CHECK(contains(ch.out, "not started by the Service Control Manager"));
    CHECK(contains(text, "service: 'brl-service-test' (pid " + std::to_string(ch.pid) + ", in a console)"));
    CHECK(contains(text, "--config " + dir + "\\svc.yaml"));
    CHECK(contains(text, "BRL_SERVICE_TEST_KEY set from " + dir + "\\key.txt"));
    // START_PENDING is told before the log opens (the SCM wants it first), so a console shows it
    CHECK(contains(ch.out, "(console) status START_PENDING"));
    CHECK(contains(text, "(console) status RUNNING"));
    CHECK(contains(text, "service: STOP: draining"));
    CHECK(contains(text, "(console) status STOP_PENDING"));
    CHECK(contains(text, "stopping: draining in-flight requests"));   // the fleet's own drain
    CHECK(contains(text, "service: stopped"));
    CHECK(contains(text, "(console) status STOPPED, checkpoint 0, wait hint 0 ms, exit 0/0"));
    CHECK(!contains(text, "brl-test-key-5e1f"));   // the key is never written down
    CHECK(!contains(ch.out, "brl-test-key-5e1f"));
    CHECK(text.find("STOP: draining") < text.find("stopping: draining in-flight requests"));
    CHECK_EQ(read_file(log + ".1"), "the previous run's log\n");
}

TEST(service_console_run_stops_on_ctrl_break) {
    const std::string dir = win_path(radtest::tmp_path("svc-b"));
    rad::os::make_dirs(dir);
    const int port = 18362;
    write_file(dir + "\\svc.yaml", fleet_yaml(port, ""));
    const std::string log = dir + "\\named\\b.log";
    rad::os::make_dirs(dir + "\\named");

    Child ch;
    REQUIRE(ch.start({RAD_SERVICE_TEST_RADIANCE, "--service", "brl-service-test-b", "--config", dir + "\\svc.yaml",
                      "--log", log},
                     child_env({"HIP_VISIBLE_DEVICES=-1", "OMP_NUM_THREADS=2"}, {})));
    if (!file_says(log, "listening on http://127.0.0.1:" + std::to_string(port), 60000)) {
        ::radtest::fail(__FILE__, __LINE__, "the service did not listen:\n" + read_file(log) + "\n" + ch.out);
        ch.finish(60000);
        return;
    }
    std::string body, why;
    CHECK_EQ(http_get(port, "/health", "", &body), 200);
    CHECK_EQ(http_get(port, "/v1/models", "", &body), 200);   // no key in this YAML
    REQUIRE(ch.c.interrupt(&why));   // win\tools\stop-server.ps1's Ctrl-Break
    CHECK_EQ(ch.finish(60000), 0);
    const std::string text = read_file(log);
    CHECK(!rad::os::is_file(dir + "\\svc.log"));   // --log named the file
    CHECK(contains(text, "stopping: draining in-flight requests"));
    CHECK(contains(text, "the engine returned 0"));
    CHECK(contains(text, "(console) status STOPPED"));
}

TEST(service_console_refusals) {
    const std::string dir = win_path(radtest::tmp_path("svc-c"));
    rad::os::make_dirs(dir);
    write_file(dir + "\\svc.yaml", fleet_yaml(18363, "BRL_SERVICE_TEST_KEY"));
    auto run = [&](const std::vector<std::string>& args, std::string* out) {
        Child ch;
        std::vector<std::string> argv = {RAD_SERVICE_TEST_RADIANCE};
        argv.insert(argv.end(), args.begin(), args.end());
        if (!ch.start(argv, child_env({"HIP_VISIBLE_DEVICES=-1"}, {"BRL_SERVICE_TEST_KEY"}))) return -1;
        const int rc = ch.finish(30000);
        *out = ch.out;
        return rc;
    };
    std::string out;
    CHECK_EQ(run({"--service", "x"}, &out), 2);
    CHECK(contains(out, "--config FILE is missing"));
    CHECK(contains(out, "usage: radiance --service NAME"));
    CHECK_EQ(run({"--service", "x", "--config", dir + "\\svc.yaml", "--check"}, &out), 2);
    CHECK(contains(out, "'--check' is not a service option"));
    // a secret file that is not there: refused before the engine starts, said in the log
    CHECK_EQ(run({"--service", "x", "--config", dir + "\\svc.yaml", "--env-from-file",
                  "BRL_SERVICE_TEST_KEY=" + dir + "\\no-such.key"}, &out), 4);
    const std::string text = read_file(dir + "\\svc.log");
    CHECK(contains(text, "--env-from-file BRL_SERVICE_TEST_KEY: cannot read"));
    CHECK(contains(text, "status STOPPED, checkpoint 0, wait hint 0 ms, exit 1066/4"));
    // without --service the command line is radiance's own, unchanged
    CHECK_EQ(run({"--config", dir + "\\svc.yaml", "--check"}, &out), 2);   // api-key-env unset: refused
    CHECK(contains(out, "BRL_SERVICE_TEST_KEY is unset or empty"));
    write_file(dir + "\\nokey.yaml", fleet_yaml(18363, ""));
    CHECK_EQ(run({"--config", dir + "\\nokey.yaml", "--check"}, &out), 0);
    CHECK(contains(out, "model a ="));
    CHECK_EQ(run({"--model", "x.rad"}, &out), 2);
    CHECK(contains(out, "radiance takes its settings from a YAML file"));
}

/* Last: it registers this process's termination function, which stays for the process's life. */
TEST(service_terminate_kept_until_the_engine_registers) {
    static std::atomic<int> calls{0}, last_sig{0};
    // a stop before anything has registered is kept, not dropped and not an exit
    CHECK(!rad::os::request_terminate(SIGTERM));
    CHECK_EQ(calls.load(), 0);
    rad::os::on_terminate([](int sig) { calls.fetch_add(1); last_sig.store(sig); });
    CHECK_EQ(calls.load(), 1);   // delivered by the registration itself
    CHECK_EQ(last_sig.load(), SIGTERM);
    // once delivered, never again: the engine drains once
    CHECK(rad::os::request_terminate(SIGTERM));
    CHECK_EQ(calls.load(), 1);
}

RAD_TEST_MAIN()
