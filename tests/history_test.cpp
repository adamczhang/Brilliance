/* history_test.cpp -- the dashboard's hour of history (core/server/history.h).
 *
 *   Rows     what two consecutive /stats objects yield: instantaneous figures from one, rates from
 *            the difference, and nothing where a counter did not move
 *   Fleet    a card's telemetry, a model's state and its bytes on each card
 *   Ring     an hour is kept, a value a sample did not yield is null, and `since` cuts
 *   Requests a request is recorded once when it finishes, seen done or gone between two samples
 *   Starts   a model's counters start over with it: no rate across a restart
 *   Live     a response body carries the latest sample's object, and none for a model not open
 *
 * No server: JSON objects in, JSON out.
 */
#include "rad_test.h"

#include "server/history.h"

#include <cmath>
#include <cstdlib>
#include <string>

using namespace rad;
using namespace rad::server;

namespace {

double of(const History::Row& r, const std::string& k) {
    for (const auto& [name, v] : r)
        if (name == k) return v;
    return std::nan("");
}
bool has(const History::Row& r, const std::string& k) { return std::isfinite(of(r, k)); }

json stats(int64_t started, double steps_decode, double step_ns_decode, double total_requests) {
    return json::parse(R"({"started_at": )" + std::to_string(started) + R"(, "decode_tps": 120.5, "prefill_tps": 900,
        "running": 2, "waiting": 1, "preempted": 0, "in_flight": 3, "prefix_hit_rate": 0.75,
        "kv_bytes_used": 4000, "kv_bytes_total": 16000, "kv_tokens_used": 1000, "kv_tokens_cached": 250,
        "step_ns_decode_best": 21000000, "jobs": {"running": 0, "done": 0, "last_ms": 0},
        "steps": )" + std::to_string(steps_decode) + R"(, "step_ns": )" + std::to_string(step_ns_decode) + R"(,
        "steps_decode": )" + std::to_string(steps_decode) + R"(, "step_ns_decode": )" + std::to_string(step_ns_decode) + R"(,
        "total_requests": )" + std::to_string(total_requests) + R"(,
        "draft_positions": [{"reached": )" + std::to_string(steps_decode) + R"(, "accepted": )" +
        std::to_string(steps_decode / 2) + R"(}], "requests": []})");
}

}  // namespace

TEST(one_stats_object_yields_the_instantaneous_figures_and_no_rate) {
    const History::Row r = History::model_row(stats(7, 100, 2.2e9, 10), nullptr, 0);
    CHECK_EQ(of(r, "decode"), 120.5);
    CHECK_EQ(of(r, "running"), 2.0);
    CHECK_EQ(of(r, "flight"), 3.0);
    CHECK_EQ(of(r, "hit"), 0.75);
    /* A quarter of the tokens are kept turns, so a quarter of the bytes. */
    CHECK_EQ(of(r, "kvKept"), 1000.0);
    CHECK_EQ(of(r, "kvRead"), 3000.0);
    CHECK_EQ(of(r, "stepBest"), 21.0);
    CHECK(!has(r, "step"));
    CHECK(!has(r, "reqRate"));
    CHECK(!has(r, "accept"));
}

TEST(two_stats_objects_yield_rates_over_the_time_between_them) {
    const json a = stats(7, 100, 2.2e9, 10), b = stats(7, 150, 3.3e9, 14);
    const History::Row r = History::model_row(b, &a, 2.0);
    CHECK_EQ(of(r, "reqRate"), 2.0);                 /* four requests in two seconds */
    CHECK_EQ(of(r, "step"), 22.0);                   /* 1.1e9 ns over 50 steps */
    CHECK_EQ(of(r, "accept"), 0.5);                  /* 25 accepted of 50 reached */
    CHECK_EQ(of(r, "tokStep"), 1.5);
    /* No step ran: no step time, and the counters that did not move yield no acceptance. */
    const History::Row idle = History::model_row(a, &a, 1.0);
    CHECK(!has(idle, "step"));
    CHECK(!has(idle, "accept"));
    CHECK_EQ(of(idle, "reqRate"), 0.0);
}

TEST(a_prompt_s_prefill_rate_is_its_fresh_tokens_over_its_own_prefill_time_once) {
    json a = stats(7, 100, 2.2e9, 10), b = stats(7, 100, 2.2e9, 10);
    b["requests"] = json::parse(R"([{"id": "r1", "state": "decode", "prompt": 5000, "cached": 1000, "prefill_s": 2.0}])");
    CHECK_EQ(of(History::model_row(b, &a, 1.0), "prefillRate"), 2000.0);
    /* The next sample sees the same request past its prefill: it is counted once. */
    CHECK(!has(History::model_row(b, &b, 1.0), "prefillRate"));
}

TEST(the_fleet_row_holds_telemetry_states_and_bytes_per_card) {
    const json f = json::parse(R"({"models": [
        {"id": "a", "state": "resident", "cards": [0, 1], "vram_bytes_per_card": 1000},
        {"id": "b", "state": "parked", "cards": [1], "vram_bytes_per_card": 0},
        {"id": "c", "state": "cold"}],
      "cards": [{"card": 0, "busy": 0.5, "power_w": 200, "temp_c": 60, "free_bytes": 7},
                {"card": 1, "free_bytes": 9}]})");
    const History::Row r = History::fleet_row(f);
    CHECK_EQ(of(r, "s:a"), 3.0);
    CHECK_EQ(of(r, "s:b"), 1.0);
    CHECK_EQ(of(r, "s:c"), 0.0);
    CHECK_EQ(of(r, "busy0"), 0.5);
    CHECK_EQ(of(r, "power0"), 200.0);
    CHECK_EQ(of(r, "temp0"), 60.0);
    CHECK(!has(r, "busy1"));                         /* not sampled is a gap, not a zero */
    CHECK_EQ(of(r, "c0:a"), 1000.0);
    CHECK_EQ(of(r, "c1:a"), 1000.0);
    CHECK_EQ(of(r, "c0:b"), 0.0);                    /* a model with nothing on a card is an empty band */
    CHECK_EQ(of(r, "c1:c"), 0.0);
}

TEST(the_ring_keeps_an_hour_and_since_cuts) {
    History h;
    for (int i = 0; i < History::kKeepS + 10; ++i) h.model_sample("m", stats(7, 100 + i, 2.2e9 + 2e7 * i, 10), 1000.0 + i);
    json all = h.model_since("m", 0);
    CHECK_EQ((int)all["t"].size(), History::kKeepS);
    CHECK_EQ(all["t"][0].get<double>(), 1010.0);                    /* the ten oldest are gone */
    CHECK_EQ((int)all["c"]["decode"].size(), History::kKeepS);
    CHECK_EQ(all["keep_s"].get<int>(), History::kKeepS);
    json tail = h.model_since("m", 1000.0 + History::kKeepS + 7);
    CHECK_EQ((int)tail["t"].size(), 2);
    CHECK_EQ((int)tail["c"]["step"].size(), 2);
    CHECK_EQ(tail["c"]["step"][0].get<double>(), 20.0);
    /* A model nobody sampled answers with nothing. */
    CHECK_EQ((int)h.model_since("other", 0)["t"].size(), 0);
}

TEST(a_value_a_sample_did_not_yield_is_null_and_an_empty_column_is_left_out) {
    History h;
    h.model_sample("m", stats(7, 100, 2.2e9, 10), 10.0);
    h.model_sample("m", stats(7, 150, 3.3e9, 10), 11.0);
    json j = h.model_since("m", 0);
    CHECK(j["c"]["step"][0].is_null());              /* the first sample has no predecessor */
    CHECK_EQ(j["c"]["step"][1].get<double>(), 22.0);
    CHECK(!j["c"].contains("lkAr"));                 /* no link counters in these objects */
}

TEST(a_request_is_recorded_once_when_it_finishes) {
    History h;
    json s = stats(7, 100, 2.2e9, 10);
    s["requests"] = json::parse(R"([{"id": "r1", "state": "decode", "prompt": 400, "cached": 0, "output": 10, "ctx": 410, "tps": 50},
                                    {"id": "r2", "state": "decode", "prompt": 100, "cached": 0, "output": 5, "ctx": 105, "tps": 40}])");
    h.model_sample("m", s, 10.0);
    CHECK_EQ((int)h.model_since("m", 0)["requests"].size(), 0);
    /* r1 is seen done and stays in the table for a while; r2 leaves between two samples. */
    s["requests"] = json::parse(R"([{"id": "r1", "state": "done", "prompt": 400, "cached": 100, "output": 90, "ctx": 490,
                                     "ttft_s": 0.25, "prefill_s": 0.5, "tps": 55}])");
    h.model_sample("m", s, 11.0);
    h.model_sample("m", s, 12.0);
    json r = h.model_since("m", 0)["requests"];
    REQUIRE_EQ((int)r.size(), 2);
    CHECK_EQ(r[0]["id"].get<std::string>(), "r1");
    CHECK_EQ(r[0]["out"].get<double>(), 90.0);
    CHECK_EQ(r[0]["pf"].get<double>(), 600.0);       /* 300 fresh tokens in half a second */
    CHECK_EQ(r[0]["ttft"].get<double>(), 0.25);
    CHECK_EQ(r[1]["id"].get<std::string>(), "r2");
    CHECK(r[1]["ttft"].is_null());
    /* `since` cuts the requests as it cuts the samples. */
    CHECK_EQ((int)h.model_since("m", 11.5)["requests"].size(), 0);
}

/* A page asks for what came after the last sample time it was given. Times are sent to the
 * millisecond, so a sample taken between two milliseconds must not be later than its own sent
 * time: asked with it, the history answers with nothing of that sample, and with no request that
 * finished in it. */
TEST(asking_with_the_last_time_given_repeats_neither_a_sample_nor_a_request) {
    for (double frac : { 0.0, 0.0004, 0.0006, 0.84561, 0.9996 }) {
        History h;
        json s = stats(7, 100, 2.2e9, 10);
        s["requests"] = json::parse(R"([{"id": "r1", "state": "decode", "prompt": 40, "cached": 0, "output": 3, "ctx": 43, "tps": 50}])");
        h.model_sample("m", s, 1794590850.0 + frac);
        s["requests"] = json::array();
        h.model_sample("m", s, 1794590851.0 + frac);       /* r1 is gone: it finished */
        const json all = h.model_since("m", 0);
        REQUIRE_EQ((int)all["t"].size(), 2);
        REQUIRE_EQ((int)all["requests"].size(), 1);
        const double last = all["t"][1].get<double>();
        /* The page sends the number back as text. */
        const double asked = std::strtod(json(last).dump().c_str(), nullptr);
        const json next = h.model_since("m", asked);
        CHECK_EQ((int)next["t"].size(), 0);
        CHECK_EQ((int)next["requests"].size(), 0);
        /* Asked with the sample before, it gets exactly the last one and its request. */
        const json one = h.model_since("m", all["t"][0].get<double>());
        CHECK_EQ((int)one["t"].size(), 1);
        CHECK_EQ((int)one["requests"].size(), 1);
        h.fleet_sample(json::parse(R"({"models": [], "cards": [{"card": 0, "free_bytes": 7}]})"), 1794590851.0 + frac);
        const json f = h.fleet_since(0);
        REQUIRE_EQ((int)f["t"].size(), 1);
        CHECK_EQ((int)h.fleet_since(f["t"][0].get<double>())["t"].size(), 0);
    }
}

TEST(a_restarted_model_yields_no_rate_across_the_restart) {
    History h;
    h.model_sample("m", stats(7, 5000, 9e10, 400), 10.0);
    h.model_sample("m", stats(8, 10, 2.2e8, 1), 11.0);        /* started_at moved: counters began again */
    json j = h.model_since("m", 0);
    CHECK(!j["c"].contains("step"));
    h.model_sample("m", stats(8, 20, 4.4e8, 3), 12.0);
    j = h.model_since("m", 0);
    CHECK_EQ(j["c"]["step"][2].get<double>(), 22.0);
    CHECK_EQ(j["c"]["reqRate"][2].get<double>(), 2.0);
    h.forget("m");
    CHECK_EQ((int)h.model_since("m", 0)["t"].size(), 0);
}

TEST(a_body_carries_the_latest_sample_s_object_as_live) {
    History h;
    CHECK_EQ(h.fleet_body(0), R"({"keep_s":3600,"t":[],"c":{},"live":null})");
    const json s = stats(7, 100, 2.2e9, 10);
    h.model_sample("m", s, 10.0, R"({"decode_tps":120.5})");
    json b = json::parse(h.model_body("m", 0));
    CHECK_EQ(b["live"]["decode_tps"].get<double>(), 120.5);
    CHECK_EQ((int)b["t"].size(), 1);
    /* A model that is not open keeps its history and has no live object. */
    h.model_closed("m");
    b = json::parse(h.model_body("m", 0));
    CHECK(b["live"].is_null());
    CHECK_EQ((int)b["t"].size(), 1);
    CHECK(json::parse(h.model_body("unknown", 0))["live"].is_null());
}

RAD_TEST_MAIN()
