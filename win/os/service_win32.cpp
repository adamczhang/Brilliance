/* Brilliance: radiance as a Windows service (win/os/service.h says what it does and does not do;
 * win/SERVICE.md how to install and qualify it).
 *
 * Adapted from Brilliance 0.2.2's core/os/service_win32.cpp: the status pump, the one-handle log
 * and the exit-status rule are its. What changed: installing is win\serve.ps1's (sc.exe), the
 * service runs radiance's YAML configuration rather than stored engine flags, a key comes from a
 * file through the environment variable the YAML names, and a stop that arrives while the engine is
 * still starting is kept for it (os::request_terminate) instead of ending the process. */
#if defined(_WIN32)

#include "os/service.h"
#include "os/os.h"

#include <windows.h>
#include <fcntl.h>
#include <io.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern "C" int brl_errno_from_win32(unsigned long err);   // win/compat

namespace rad::os {

namespace service {

static_assert(kStopped == SERVICE_STOPPED && kStartPending == SERVICE_START_PENDING &&
              kStopPending == SERVICE_STOP_PENDING && kRunning == SERVICE_RUNNING);
static_assert(kControlStop == SERVICE_CONTROL_STOP && kControlInterrogate == SERVICE_CONTROL_INTERROGATE &&
              kControlShutdown == SERVICE_CONTROL_SHUTDOWN && kControlPreshutdown == SERVICE_CONTROL_PRESHUTDOWN);
static_assert(kAcceptStop == SERVICE_ACCEPT_STOP && kAcceptShutdown == SERVICE_ACCEPT_SHUTDOWN &&
              kAcceptPreshutdown == SERVICE_ACCEPT_PRESHUTDOWN);
static_assert(kNoError == NO_ERROR && kErrorCallNotImplemented == ERROR_CALL_NOT_IMPLEMENTED &&
              kErrorServiceSpecific == ERROR_SERVICE_SPECIFIC_ERROR);

/* ------------------------------------------------------------------------- the command line */
bool wanted(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (argv[i] && std::strcmp(argv[i], "--service") == 0) return true;
    return false;
}

bool parse(int argc, char** argv, Args* a, std::string* why) {
    *a = Args{};
    bool have_name = false, have_config = false, have_log = false, have_timeout = false;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto value = [&](std::string* out) {
            if (i + 1 >= argc || !*argv[i + 1]) {
                *why = s + " needs a value";
                return false;
            }
            *out = argv[++i];
            return true;
        };
        auto once = [&](bool* seen) {
            if (*seen) {
                *why = s + " is given twice";
                return false;
            }
            return *seen = true;
        };
        if (s == "--service") {
            if (!once(&have_name) || !value(&a->name)) return false;
            if (a->name.find_first_of("/\\") != std::string::npos) {
                *why = "--service " + a->name + ": a service name has no / or \\";
                return false;
            }
        } else if (s == "--config") {
            if (!once(&have_config) || !value(&a->config)) return false;
        } else if (s == "--log") {
            if (!once(&have_log) || !value(&a->log)) return false;
        } else if (s == "--env-from-file") {
            std::string kv;
            if (!value(&kv)) return false;
            const size_t eq = kv.find('=');
            if (eq == 0 || eq == std::string::npos || eq + 1 == kv.size()) {
                *why = "--env-from-file takes VAR=FILE, not '" + kv + "'";
                return false;
            }
            a->env_files.emplace_back(kv.substr(0, eq), full_path(kv.substr(eq + 1)));
        } else if (s == "--stop-timeout") {
            std::string v;
            if (!once(&have_timeout) || !value(&v)) return false;
            char* end = nullptr;
            const long long sec = std::strtoll(v.c_str(), &end, 10);
            if (!end || *end || sec < 1 || sec > 86400) {
                *why = "--stop-timeout takes whole seconds from 1 to 86400, not '" + v + "'";
                return false;
            }
            a->stop_timeout_ms = sec * 1000;
        } else {
            *why = "'" + s + "' is not a service option (a service takes --service NAME --config FILE"
                   " [--log FILE] [--env-from-file VAR=FILE]... [--stop-timeout SECONDS])";
            return false;
        }
    }
    if (!have_name) {
        *why = "--service NAME is missing";
        return false;
    }
    if (!have_config) {
        *why = "--config FILE is missing: a service serves a YAML configuration";
        return false;
    }
    a->config = full_path(a->config);
    a->log = have_log ? full_path(a->log) : default_log(a->config);
    return true;
}

std::string default_log(const std::string& config) {
    const size_t slash = config.find_last_of("/\\");
    const size_t dot = config.find_last_of('.');
    const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash + 1);
    return (has_ext ? config.substr(0, dot) : config) + ".log";
}

bool read_value_file(const std::string& path, std::string* value, std::string* why) {
    File f = File::open(path, File::Read);
    if (!f.valid()) {
        *why = "cannot read " + path + ": " + std::strerror(errno);
        return false;
    }
    char buf[4096];
    const int64_t n = f.read_at(buf, sizeof buf, 0);
    if (n < 0) {
        *why = "cannot read " + path + ": " + std::strerror(errno);
        return false;
    }
    if (n == (int64_t)sizeof buf) {
        *why = path + " is 4 KiB or more: not a one-line value";
        return false;
    }
    std::string v(buf, (size_t)n);
    if (v.compare(0, 3, "\xEF\xBB\xBF") == 0) v.erase(0, 3);
    while (!v.empty() && (v.back() == '\r' || v.back() == '\n' || v.back() == ' ' || v.back() == '\t'))
        v.pop_back();
    if (v.empty()) {
        *why = path + " is empty";
        return false;
    }
    if (v.find_first_of("\r\n") != std::string::npos) {
        *why = path + " holds more than one line; a value is one line";
        return false;
    }
    *value = std::move(v);
    return true;
}

void rotate_log(const std::string& path) {
    if (is_file(path)) (void)rename_replace(path, path + ".1");
}

/* ------------------------------------------------------------------------- the state */
namespace {
const char* control_name(uint32_t c) {
    switch (c) {
        case kControlStop: return "STOP";
        case kControlShutdown: return "SHUTDOWN";
        case kControlPreshutdown: return "PRESHUTDOWN";
        default: return "?";
    }
}
std::string seconds(int64_t ms) { return std::to_string((ms + 500) / 1000) + " s"; }
}   // namespace

Control::Control(Hooks hooks, int64_t stop_timeout_ms, long pid)
    : h_(std::move(hooks)), stop_timeout_ms_(stop_timeout_ms), pid_(pid) {}

void Control::report_locked(uint32_t state, uint32_t wait_hint_ms) {
    if (st_.state == kStopped) return;   // nothing after STOPPED: the process may be ended any time
    st_.state = state;
    st_.accepted = state == kRunning ? (kAcceptStop | kAcceptPreshutdown | kAcceptShutdown) : 0;
    st_.wait_hint_ms = wait_hint_ms;
    st_.checkpoint = (state == kStartPending || state == kStopPending) ? next_checkpoint_++ : 0;
    if (h_.report) h_.report(st_);
}

void Control::starting() {
    std::lock_guard<std::mutex> lk(mu_);
    report_locked(kStartPending, (uint32_t)kWaitHintMs);
}

void Control::running() {
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_asked_) return;   // a stop already came; RUNNING would undo its STOP_PENDING
    report_locked(kRunning, 0);
}

uint32_t Control::control(uint32_t code) {
    if (code == kControlInterrogate) return kNoError;
    if (code != kControlStop && code != kControlShutdown && code != kControlPreshutdown)
        return kErrorCallNotImplemented;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_asked_ || st_.state == kStopped) return kNoError;   // one drain, however asked
        stop_asked_ = true;
        report_locked(kStopPending, (uint32_t)kWaitHintMs);
    }
    if (h_.log)
        h_.log(std::string(control_name(code)) + ": draining -- the engine finishes the step in flight, "
               "ends its requests and releases the cards");
    const bool now = h_.terminate && h_.terminate(SIGTERM);
    if (!now && h_.log)
        h_.log("the engine is still starting and has not taken its stop yet; it takes it as soon as it "
               "does, before it loads a model");
    if (h_.stop_begun) h_.stop_begun();
    return kNoError;
}

bool Control::pump(int64_t since_stop_ms) {
    std::string line;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (st_.state == kStopped) return false;
        if (since_stop_ms < stop_timeout_ms_) {
            report_locked(kStopPending, (uint32_t)kWaitHintMs);
            if (since_stop_ms >= next_log_ms_) {
                line = "still draining, " + seconds(since_stop_ms) + " after the stop";
                while (next_log_ms_ <= since_stop_ms) next_log_ms_ += kDrainLogMs;
            }
        } else if (!overdue_) {
            overdue_ = true;
            next_log_ms_ = since_stop_ms + kOverdueLogMs;
            line = "the engine has not finished its drain " + seconds(since_stop_ms) +
                   " after the stop (--stop-timeout " + seconds(stop_timeout_ms_) +
                   "). It is NOT terminated: a process with work on a GPU is never killed "
                   "(win/SERVICE.md). The service stays STOP_PENDING, its checkpoint no longer "
                   "moving, until the engine returns. Where it waits: brl-stacks " +
                   std::to_string(pid_) + " (beside radiance.exe)";
        } else if (since_stop_ms >= next_log_ms_) {
            next_log_ms_ = since_stop_ms + kOverdueLogMs;
            line = "still not stopped, " + seconds(since_stop_ms) + " after the stop; not terminated";
        }
    }
    if (!line.empty() && h_.log) h_.log(line);
    return true;
}

void Control::stopped(int engine_rc) {
    std::lock_guard<std::mutex> lk(mu_);
    if (st_.state == kStopped) return;
    std::string line;
    st_.specific = (uint32_t)engine_rc;
    if (engine_rc == 0) {
        st_.exit_code = kNoError;
        line = stop_asked_ ? "stopped"
                           : "the engine returned 0 (in a console, Ctrl-C or Ctrl-Break stops it directly); stopped";
    } else if (stop_asked_) {
        st_.exit_code = kNoError;
        line = "ended with status " + std::to_string(engine_rc) +
               " after a requested stop; reported as stopped, so the recovery actions do not restart it";
    } else {
        st_.exit_code = kErrorServiceSpecific;
        line = "ended with status " + std::to_string(engine_rc) +
               " without a stop being asked: reported as a failure (service-specific error " +
               std::to_string(engine_rc) + "), which the recovery actions answer";
    }
    // Said before STOPPED is reported: after it, the process may be ended at any time.
    if (h_.log) h_.log(line);
    report_locked(kStopped, 0);
}

bool Control::stop_asked() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stop_asked_;
}

Status Control::status() const {
    std::lock_guard<std::mutex> lk(mu_);
    return st_;
}

}   // namespace service

/* ------------------------------------------------------------------------- the log
 * ONE handle, opened for appending only, under every way out (the HIP runtime writes to the
 * standard handles): every write lands at the end as one piece, so the two streams interleave line
 * by line instead of overwriting each other, as two "a" streams with their own positions would.
 * Inheritable, so the model fetcher's progress reaches it too. FILE_SHARE_DELETE so the next start
 * can rotate a log a leftover child still holds. stdout unbuffered as well: the MSVC CRT's _IOLBF is
 * full buffering, which would lose the tail of the log to a crash. */
int redirect_output(const std::string& path) {
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE h = CreateFileW(widen(path).c_str(), FILE_APPEND_DATA | SYNCHRONIZE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, &sa, OPEN_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        errno = brl_errno_from_win32(GetLastError());
        return -1;
    }
    const int fd = _open_osfhandle((intptr_t)h, _O_APPEND | _O_WRONLY);
    if (fd < 0) {
        CloseHandle(h);
        return -1;
    }
    std::fflush(stdout);
    std::fflush(stderr);
    /* A PROCESS STARTED WITH NO CONSOLE -- every service -- HAS CRT STREAMS ATTACHED TO NOTHING:
     * _fileno(stdout) is -2, a _dup2 onto 1 and 2 never reaches the stream, and every fprintf
     * fails, so the log stays empty. Such a stream is given a descriptor of its own first, and the
     * log goes under that one as well as under 1 and 2. */
    for (FILE* s : {stdout, stderr}) {
        FILE* again = nullptr;
        if (_fileno(s) < 0 && freopen_s(&again, "NUL", "w", s) != 0) {
            _close(fd);
            return -1;
        }
    }
    int rc = _dup2(fd, 1) | _dup2(fd, 2);
    if (_fileno(stdout) != 1) rc |= _dup2(fd, _fileno(stdout));
    if (_fileno(stderr) != 2) rc |= _dup2(fd, _fileno(stderr));
    _close(fd);
    if (rc != 0) return -1;
    SetStdHandle(STD_OUTPUT_HANDLE, (HANDLE)_get_osfhandle(1));
    SetStdHandle(STD_ERROR_HANDLE, (HANDLE)_get_osfhandle(2));
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    return 0;
}

/* ------------------------------------------------------------------------- under the SCM */
namespace {

using service::Control;

/* Service-specific exit codes of the service's own (the engine's are 1 and 2). */
constexpr int kExitArgs = 2;     // the command line (as radiance's own refusals)
constexpr int kExitLog = 3;      // the log cannot be opened
constexpr int kExitEnvFile = 4;  // an --env-from-file file cannot be read

service::Args          g_args;
std::string            g_args_error;          // parse()'s complaint, reported through the SCM
int                    (*g_run)(int, char**) = nullptr;
SERVICE_STATUS_HANDLE  g_handle = nullptr;
Control*               g_ctl = nullptr;
std::atomic<int>       g_rc{0};
bool                   g_console = false;     // started from a console, not by the SCM

void log_line(const std::string& s) {
    SYSTEMTIME t;
    GetLocalTime(&t);
    std::fprintf(stderr, "%04u-%02u-%02u %02u:%02u:%02u service: %s\n", t.wYear, t.wMonth, t.wDay,
                 t.wHour, t.wMinute, t.wSecond, s.c_str());
    std::fflush(stderr);
}

const char* state_name(uint32_t s) {
    switch (s) {
        case service::kStopped: return "STOPPED";
        case service::kStartPending: return "START_PENDING";
        case service::kStopPending: return "STOP_PENDING";
        case service::kRunning: return "RUNNING";
        default: return "?";
    }
}

void report(const service::Status& s) {
    if (g_console) {
        // What the SCM would have been told, so a console run shows the same sequence.
        log_line(std::string("(console) status ") + state_name(s.state) + ", checkpoint " +
                 std::to_string(s.checkpoint) + ", wait hint " + std::to_string(s.wait_hint_ms) +
                 " ms, exit " + std::to_string(s.exit_code) + "/" + std::to_string(s.specific));
        return;
    }
    SERVICE_STATUS st{};
    st.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    st.dwCurrentState = s.state;
    st.dwControlsAccepted = s.accepted;
    st.dwWin32ExitCode = s.exit_code;
    st.dwServiceSpecificExitCode = s.specific;
    st.dwCheckPoint = s.checkpoint;
    st.dwWaitHint = s.wait_hint_ms;
    SetServiceStatus(g_handle, &st);
}

/* From the first stop until STOPPED: the next checkpoint every two seconds (Control::pump). */
void start_pump() {
    std::thread([] {
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            Sleep((DWORD)Control::kPumpMs);
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0).count();
            if (!g_ctl->pump(ms)) return;
        }
    }).detach();
}

DWORD WINAPI handler(DWORD control, DWORD, LPVOID, LPVOID) { return g_ctl->control(control); }

/* The service, from its log to the engine's return; the same under the SCM and in a console. */
int run(const service::Args& a) {
    g_ctl->starting();
    service::rotate_log(a.log);
    if (redirect_output(a.log) != 0) {
        const std::string why = "cannot open the log " + a.log + ": " + std::strerror(errno);
        OutputDebugStringA(("radiance service: " + why + "\n").c_str());
        log_line(why);   // reaches a console, if there is one
        g_ctl->stopped(kExitLog);
        return kExitLog;
    }
    log_line("'" + a.name + "' (pid " + std::to_string(GetCurrentProcessId()) + (g_console ? ", in a console" : "") +
             "): " + exe_path() + " --config " + a.config);
    for (const auto& [var, file] : a.env_files) {
        std::string v, why;
        if (!service::read_value_file(file, &v, &why) || _putenv_s(var.c_str(), v.c_str()) != 0) {
            log_line("--env-from-file " + var + ": " + (why.empty() ? "cannot set it" : why));
            g_ctl->stopped(kExitEnvFile);
            return kExitEnvFile;
        }
        log_line(var + " set from " + file + " (its value is not shown)");
    }
    process_init();   // os.h: priority class, power throttling, timer -- after the service's environment
    // Relative paths in the YAML mean the config's folder, not System32 where the SCM starts us.
    const size_t slash = a.config.find_last_of("/\\");
    if (slash != std::string::npos) {
        const std::string dir = a.config.substr(0, slash + 1);
        if (!SetCurrentDirectoryW(widen(dir).c_str()))
            log_line("cannot make " + dir + " the working directory: " + win32_message());
    }
    g_ctl->running();
    std::vector<std::string> args = {exe_path(), "--config", a.config};
    std::vector<char*> argv;
    for (std::string& s : args) argv.push_back(s.data());
    argv.push_back(nullptr);
    const int rc = g_run((int)args.size(), argv.data());
    std::fflush(stdout);
    std::fflush(stderr);
    g_ctl->stopped(rc);
    return rc;
}

void WINAPI service_main_cb(DWORD, LPWSTR*) {
    g_handle = RegisterServiceCtrlHandlerExW(widen(g_args.name).c_str(), handler, nullptr);
    if (!g_handle) {
        g_rc.store(1);
        return;
    }
    if (!g_args_error.empty()) {
        // Nowhere to write it but the debugger's stream; `sc query` shows the code.
        OutputDebugStringA(("radiance service: " + g_args_error + "\n").c_str());
        g_ctl->stopped(kExitArgs);
        g_rc.store(kExitArgs);
        return;
    }
    g_rc.store(run(g_args));
}

/* In a console, the SCM's STOP is this event: what a test or a person uses to take the service's
 * own stop path (Control::control) rather than Ctrl-Break's. */
std::wstring console_stop_event_name() {
    return L"Local\\brilliance-service-stop-" + std::to_wstring(GetCurrentProcessId());
}

}   // namespace

bool service_wanted(int argc, char** argv) { return service::wanted(argc, argv); }

int service_main(int argc, char** argv, int (*run_engine)(int, char**)) {
    g_run = run_engine;
    std::string why;
    if (!service::parse(argc, argv, &g_args, &why)) g_args_error = why;
    static Control ctl({report, request_terminate, log_line, start_pump}, g_args.stop_timeout_ms,
                       (long)GetCurrentProcessId());
    g_ctl = &ctl;

    // For a SERVICE_WIN32_OWN_PROCESS service the name in the table is not used, so a command line
    // that names none can still be dispatched and say what is wrong with it.
    std::wstring wname = widen(g_args.name);
    SERVICE_TABLE_ENTRYW table[] = {{wname.data(), service_main_cb}, {nullptr, nullptr}};
    if (StartServiceCtrlDispatcherW(table)) return g_rc.load();
    const DWORD err = GetLastError();
    if (err != ERROR_FAILED_SERVICE_CONTROLLER_CONNECT) {
        std::fprintf(stderr, "radiance --service: StartServiceCtrlDispatcher: %s\n", win32_message(err).c_str());
        return 1;
    }

    // Started from a console, not by the SCM: run here exactly as the service would.
    if (!g_args_error.empty()) {
        std::fprintf(stderr,
                     "radiance --service: %s\n"
                     "usage: radiance --service NAME --config FILE [--log FILE] [--env-from-file VAR=FILE]...\n"
                     "                [--stop-timeout SECONDS]\n"
                     "  what the Service Control Manager starts (win\\serve.ps1 -Install; win\\SERVICE.md)\n",
                     g_args_error.c_str());
        return kExitArgs;
    }
    g_console = true;
    const std::wstring ev_name = console_stop_event_name();
    HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, ev_name.c_str());
    std::fprintf(stderr,
                 "radiance --service %s: not started by the Service Control Manager, so it runs in this\n"
                 "  console as the service would. Its output goes to %s.\n"
                 "  Ctrl-C or Ctrl-Break stops it (win\\tools\\stop-server.ps1 -ProcessId %lu), as does\n"
                 "  setting the event %s, which takes the service's own STOP path.\n",
                 g_args.name.c_str(), g_args.log.c_str(), GetCurrentProcessId(), narrow(ev_name).c_str());
    std::fflush(stderr);
    if (ev)
        std::thread([ev] {
            if (WaitForSingleObject(ev, INFINITE) == WAIT_OBJECT_0) g_ctl->control(service::kControlStop);
        }).detach();
    return run(g_args);
}

}   // namespace rad::os

#endif  // _WIN32
