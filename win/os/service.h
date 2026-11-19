/* Brilliance: radiance as a Windows service, the parts a test drives without the Service Control
 * Manager (win/SERVICE.md). service_win32.cpp connects them to the SCM; os.h has the entry point.
 *
 *   radiance --service NAME --config FILE [--log FILE] [--env-from-file VAR=FILE]...
 *            [--stop-timeout SECONDS]
 *
 * THE SERVICE IS THIN. It runs radiance's own main with `--config FILE`, and a stop is the engine's
 * own drain: STOP, PRESHUTDOWN and SHUTDOWN all become os::request_terminate(SIGTERM), which is what
 * Ctrl-Break does in a console. What it adds is only what a service needs and a console has:
 *   - a log file (--log, else <the config's folder>\<its name>.log), the previous one kept as .1;
 *   - secrets from files (--env-from-file VAR=FILE sets VAR from FILE's text before the engine
 *     reads its configuration), so a YAML's `api-key-env: VAR` finds the key while the key is on
 *     no command line;
 *   - the working directory, the config's folder (the SCM starts services in System32);
 *   - the SCM's status: STOP_PENDING with a moving checkpoint while the engine drains.
 *
 * NOTHING HERE ENDS THE PROCESS. A drain that takes longer than --stop-timeout is reported -- in the
 * log, and by the checkpoint no longer moving -- and the service stays STOP_PENDING until the engine
 * returns. A process with work on a GPU is never killed: under 0.2.2 a kill took the display driver
 * down. No <windows.h> here, as in os.h. */
#pragma once
#if defined(_WIN32)

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rad::os::service {

/* The SCM's numbers (winsvc.h, winerror.h), so that a test needs no windows.h; service_win32.cpp
 * checks each against the SDK's. */
constexpr uint32_t kStopped = 1, kStartPending = 2, kStopPending = 3, kRunning = 4;
constexpr uint32_t kControlStop = 0x1, kControlInterrogate = 0x4, kControlShutdown = 0x5,
                   kControlPreshutdown = 0xF;
constexpr uint32_t kAcceptStop = 0x1, kAcceptShutdown = 0x4, kAcceptPreshutdown = 0x100;
constexpr uint32_t kNoError = 0, kErrorCallNotImplemented = 120, kErrorServiceSpecific = 1066;

/* The command line, what parse() makes of it. */
struct Args {
    std::string name;                                            /* --service NAME */
    std::string config;                                          /* --config FILE, made absolute */
    std::string log;                                             /* --log FILE, made absolute;
                                                                    default_log(config) when absent */
    std::vector<std::pair<std::string, std::string>> env_files;  /* --env-from-file VAR=FILE */
    int64_t stop_timeout_ms = 180000;                            /* --stop-timeout SECONDS */
};

/* A --service anywhere on the command line. */
bool        wanted(int argc, char** argv);
/* The service's command line. False, with what is wrong in *why, for a missing or repeated
 * --service or --config, a flag without its value, and anything else on it: radiance's own
 * --check and --help, and the flags radiance refuses, do not belong in a service. */
bool        parse(int argc, char** argv, Args* a, std::string* why);
/* <the config's folder>\<its file name without the extension>.log */
std::string default_log(const std::string& config);
/* What a --env-from-file file holds: its text without a leading UTF-8 byte-order mark and without
 * trailing CR, LF, spaces and tabs (what Set-Content and an editor leave). False, with why, when the
 * file cannot be read, is empty or holds more than one line -- a key is one line, and a file of
 * several is more likely the wrong file than a key. */
bool        read_value_file(const std::string& path, std::string* value, std::string* why);
/* `path` moved to path.1 (an older .1 replaced), so each start begins its own log and the one
 * before it is kept. Nothing happens when there is no `path`. */
void        rotate_log(const std::string& path);

/* What the service tells the SCM: SERVICE_STATUS's fields that change. */
struct Status {
    uint32_t state = 0;
    uint32_t accepted = 0;      /* dwControlsAccepted */
    uint32_t exit_code = 0;     /* dwWin32ExitCode */
    uint32_t specific = 0;      /* dwServiceSpecificExitCode: the engine's exit status */
    uint32_t checkpoint = 0;
    uint32_t wait_hint_ms = 0;
};

/* The service's state and its control handler, with the SCM behind three functions: a test gives
 * it fakes and drives it; service_win32.cpp gives it SetServiceStatus, os::request_terminate and
 * the log. Thread-safe: the SCM's handler thread, the pump and the service thread all call it. */
class Control {
public:
    struct Hooks {
        std::function<void(const Status&)>      report;       /* SetServiceStatus */
        std::function<bool(int sig)>            terminate;    /* os::request_terminate */
        std::function<void(const std::string&)> log;          /* one line of the service log */
        std::function<void()>                   stop_begun;   /* the first stop: start the pump */
    };
    /* The pump reports STOP_PENDING every kPumpMs with the next checkpoint, each report promising
     * the next within kWaitHintMs. */
    static constexpr int64_t kPumpMs = 2000;
    static constexpr int64_t kWaitHintMs = 10000;
    /* While the drain is within the stop timeout, a log line every kDrainLogMs; past it, one every
     * kOverdueLogMs. */
    static constexpr int64_t kDrainLogMs = 30000;
    static constexpr int64_t kOverdueLogMs = 60000;

    Control(Hooks hooks, int64_t stop_timeout_ms, long pid);

    /* START_PENDING, no control accepted: the service is setting up its log and environment. */
    void     starting();
    /* RUNNING, accepting STOP, PRESHUTDOWN and SHUTDOWN: the engine is starting. A model loads for
     * minutes; /health says when it serves. */
    void     running();
    /* The SCM's handler. STOP, PRESHUTDOWN and SHUTDOWN: the first reports STOP_PENDING, asks the
     * engine to drain (Hooks::terminate with SIGTERM) and starts the pump (Hooks::stop_begun); a
     * later one changes nothing. INTERROGATE: nothing to do. NO_ERROR for those,
     * ERROR_CALL_NOT_IMPLEMENTED for anything else. */
    uint32_t control(uint32_t code);
    /* One step of the pump, `since_stop_ms` after the first stop: within the stop timeout the next
     * checkpoint; past it, the report that the drain is overdue (once, then a reminder every
     * kOverdueLogMs) and a checkpoint that no longer moves. False once STOPPED has been reported:
     * the pump ends. */
    bool     pump(int64_t since_stop_ms);
    /* The engine returned `engine_rc`: STOPPED. 0 is a clean stop. Anything else is a failure the
     * SCM's recovery actions answer (ERROR_SERVICE_SPECIFIC_ERROR, with the status as the
     * service-specific code) -- unless a stop had been asked for: a service its operator stopped
     * stays stopped, and the log has the status. Nothing is reported after STOPPED. */
    void     stopped(int engine_rc);

    bool     stop_asked() const;
    Status   status() const;

private:
    void report_locked(uint32_t state, uint32_t wait_hint_ms);

    Hooks           h_;
    const int64_t   stop_timeout_ms_;
    const long      pid_;
    mutable std::mutex mu_;
    Status          st_;
    uint32_t        next_checkpoint_ = 1;
    bool            stop_asked_ = false;
    int64_t         next_log_ms_ = kDrainLogMs;
    bool            overdue_ = false;
};

}   // namespace rad::os::service

#endif /* _WIN32 */
