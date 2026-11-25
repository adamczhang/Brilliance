/* Hy4-flash's chat template, reasoning_effort contract and reply format.
 *
 * The template under test is the release's own (tests/data/chat/hy4flash/chat_template.jinja,
 * unchanged); its renders against transformers' are compared by chat_test over the golden files.
 * Here: the generation prompt for both effort words, the refusals, the reply format
 * (arch/hyv4flash/chat.h) read at every byte, and replies taken back out of the template's own
 * renders of assistant turns, so the format cannot drift from what the template writes. */
#include "rad_test.h"
#include "text/chat.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"
#include "../arch/hyv4flash/chat.h"

#include "util/json.h"
#include <fstream>
#include <sstream>

using namespace rad;
using json = Json;

static const char* kBos = "<｜hy_start:opensource｜>";
static const char* kEos = "<｜hy_end:opensource｜>";
static const char* kThink = "<think:opensource>";
static const char* kThinkEnd = "</think:opensource>";
static const char* kCalls = "<tool_calls:opensource>";
static const char* kCallsEnd = "</tool_calls:opensource>";
static const char* kCall = "<tool_call:opensource>";
static const char* kCallEnd = "</tool_call:opensource>";
static const char* kKey = "<arg_key:opensource>";
static const char* kKeyEnd = "</arg_key:opensource>";
static const char* kVal = "<arg_value:opensource>";
static const char* kValEnd = "</arg_value:opensource>";

static std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

static ChatTemplate& tmpl() {
    static ChatTemplate* t = [] {
        auto* x = new ChatTemplate();
        const std::string src = read_file(std::string(RAD_CHAT_FIXTURES) + "/hy4flash/chat_template.jinja");
        CHECK(!src.empty());
        CHECK_EQ(x->load(src, kBos, kEos), RAD_OK);
        return x;
    }();
    return *t;
}

static const json& tools() {
    static const json t = json::parse(R"([
      {"type":"function","function":{"name":"weather","parameters":{"type":"object",
       "properties":{"city":{"type":"string"},"days":{"type":"integer"},
                     "verbose":{"type":"boolean"},"units":{"type":"array","items":{"type":"string"}}}}}},
      {"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}
    ])");
    return t;
}

static ChatFormat format() {
    ChatFormat f;
    std::string why;
    CHECK_OK(chat_format_from_abi(hyv4flash::hy4_chat_format(), &f, &why));
    CHECK_OK(chat_format_check(f, &why));
    return f;
}

static std::shared_ptr<const ReplyParser> parser(const std::string& gen_prompt, bool with_tools = true) {
    std::string why;
    std::shared_ptr<const ReplyFormat> rf;
    std::shared_ptr<const ReplyParser> p;
    CHECK_OK(ReplyFormat::compile(format(), &rf, &why));
    ReplyOptions opts;
    opts.generation_prompt = gen_prompt;
    opts.tools = with_tools ? &tools() : nullptr;
    opts.terminators = {kEos};
    CHECK(rf != nullptr);
    if (rf) CHECK_OK(ReplyParser::create(rf, opts, &p, &why));
    return p;
}

/* Renders `messages` and returns the prompt; `effort` empty leaves the variable out. */
static ChatPrompt render(const json& messages, const std::string& effort, bool with_tools,
                         int* status = nullptr) {
    ChatOptions opt;
    opt.quiet = true;
    if (!effort.empty()) opt.template_kwargs["reasoning_effort"] = json(effort).dump();
    ChatPrompt p;
    const int st = tmpl().apply(messages, with_tools ? tools() : json::array(), opt, &p);
    if (status) *status = st;
    return p;
}

static const json& hello() {
    static const json m = json::parse(R"([{"role":"user","content":"Hello"}])");
    return m;
}

/* ------------------------------------------------------------------ the effort contract */

TEST(the_generation_prompt_opens_a_think_block_for_high_and_closes_it_for_no_think) {
    const std::string head = std::string(kBos) + "assistant<｜hy_middle:opensource｜>";
    for (const char* e : {"", "high"}) {
        const ChatPrompt p = render(hello(), e, false);
        CHECK(p.prompt.ends_with(head + kThink));
        CHECK_EQ(p.generation_prompt, head + kThink);
        /* The effort is written into the leading system turn the template makes. */
        CHECK(p.prompt.find("<｜reasoning_mode:opensource｜>reasoning_effort:high") != std::string::npos);
    }
    const ChatPrompt n = render(hello(), "no_think", false);
    CHECK(n.prompt.ends_with(head + kThink + kThinkEnd));
    CHECK_EQ(n.generation_prompt, head + kThink + kThinkEnd);
    CHECK(n.prompt.find("reasoning_effort:no_think") != std::string::npos);
}

TEST(every_other_effort_is_refused_by_the_template_in_its_own_words) {
    for (const char* e : {"low", "medium", "xhigh", "max", "none", "minimal", "HIGH", ""}) {
        if (!*e) continue;
        int st = 0;
        const ChatPrompt p = render(hello(), e, false, &st);
        CHECK(st < 0);
        CHECK(p.error.find(std::string("reasoning_effort error : ") + e + ", should be no_think/high") !=
              std::string::npos);
    }
    /* A null effort has its own sentence (the template tests for none after the membership). */
    ChatOptions opt;
    opt.quiet = true;
    opt.template_kwargs["reasoning_effort"] = "null";
    ChatPrompt p;
    CHECK(tmpl().apply(hello(), json::array(), opt, &p) < 0);
    CHECK(p.error.find("should be no_think/high") != std::string::npos);
}

TEST(no_think_replays_history_with_empty_think_tags_and_high_keeps_reasoning_of_the_open_turn) {
    const json h = json::parse(R"([
      {"role":"user","content":"First."},
      {"role":"assistant","reasoning_content":"Old thought.","content":"One."},
      {"role":"user","content":"Second."}])");
    const std::string empty = std::string(kThink) + kThinkEnd + "One.";
    const std::string kept = std::string(kThink) + "Old thought." + kThinkEnd + "One.";
    CHECK(render(h, "no_think", false).prompt.find(empty) != std::string::npos);
    CHECK(render(h, "no_think", false).prompt.find("Old thought.") == std::string::npos);
    /* Without tools an answered turn's reasoning is dropped; with tools it is preserved. */
    CHECK(render(h, "high", false).prompt.find(empty) != std::string::npos);
    CHECK(render(h, "high", true).prompt.find(kept) != std::string::npos);
    CHECK(render(h, "no_think", true).prompt.find(kept) == std::string::npos);
}

/* ------------------------------------------------------------------ the reply format */

TEST(the_format_names_the_opensource_markers_and_the_grammar_triggers_on_the_section) {
    const ChatFormat f = format();
    CHECK_EQ(f.reasoning_open, std::string(kThink));
    CHECK_EQ(f.reasoning_close, std::string(kThinkEnd));
    CHECK_EQ(f.section_open, std::string(kCalls));
    CHECK_EQ(f.name_prefix, std::string(kCall));
    ChatGrammarRequest req;
    req.tools = &tools();
    req.generation_prompt = std::string(kThink);
    ChatGrammar g;
    std::string why;
    CHECK_OK(chat_format_grammar(f, req, &g, &why));
    CHECK(g.lazy);
    CHECK(!g.text.empty());
    REQUIRE_EQ(g.trigger_words.size(), (size_t)1);
    CHECK_EQ(g.trigger_words[0], std::string(kCalls));
    /* The decoder must show the structure to the parser. */
    bool has_calls = false;
    for (const std::string& m : chat_format_markers(f)) has_calls |= m == kCalls;
    CHECK(has_calls);
}

static std::string typed_calls() {
    return std::string(kCalls) +
           kCall + "weather" + kKey + "city" + kKeyEnd + kVal + "Paris" + kValEnd +
           kKey + "days" + kKeyEnd + kVal + "3" + kValEnd +
           kKey + "verbose" + kKeyEnd + kVal + "true" + kValEnd +
           kKey + "units" + kKeyEnd + kVal + "[\"C\",\"F\"]" + kValEnd + kCallEnd +
           kCall + "ping" + kCallEnd + kCallsEnd;
}

static void check_typed(const ReplyMessage& m) {
    CHECK_EQ(m.reasoning, std::string("I should look."));
    CHECK_EQ(m.content, std::string("Checking."));
    REQUIRE_EQ(m.calls.size(), (size_t)2);
    CHECK_EQ(m.calls[0].name, std::string("weather"));
    CHECK(m.calls[0].closed);
    const json args = json::parse(m.calls[0].arguments);
    CHECK_EQ(args["city"].get<std::string>(), std::string("Paris"));
    CHECK_EQ(args["days"].get<int>(), 3);
    CHECK_EQ(args["verbose"].get<bool>(), true);
    CHECK(args["units"] == json::array({"C", "F"}));
    CHECK_EQ(m.calls[1].name, std::string("ping"));
    CHECK(json::parse(m.calls[1].arguments) == json::object());
}

TEST(thinking_and_typed_calls_survive_one_byte_streaming) {
    const std::string gen = std::string(kBos) + "assistant<｜hy_middle:opensource｜>" + kThink;
    auto p = parser(gen);
    REQUIRE(p != nullptr);
    const std::string s = std::string("I should look.") + kThinkEnd + "Checking." + typed_calls();
    check_typed(reply_parse(*p, s));
    /* The terminator the model ends with is not content. */
    check_typed(reply_parse(*p, s + kEos));
    auto stream = p->open();
    for (char c : s) stream->feed(std::string_view(&c, 1), nullptr);
    stream->finish(nullptr);
    check_typed(stream->message());
}

TEST(no_think_replies_start_in_content_and_an_unclosed_thought_still_yields_its_call) {
    const std::string off = std::string(kBos) + "assistant<｜hy_middle:opensource｜>" + kThink + kThinkEnd;
    auto p = parser(off, false);
    const ReplyMessage a = reply_parse(*p, std::string("Hello.") + kEos);
    CHECK(a.reasoning.empty());
    CHECK_EQ(a.content, std::string("Hello."));
    /* A tool request is refused only by the tools, not by the format: with none, calls are text. */
    auto on = parser(std::string(kThink));
    const ReplyMessage b = reply_parse(*on, std::string("Plan.") + kCalls + kCall + "ping" + kCallEnd + kCallsEnd);
    CHECK_EQ(b.reasoning, std::string("Plan."));
    REQUIRE_EQ(b.calls.size(), (size_t)1);
    CHECK_EQ(b.calls[0].name, std::string("ping"));
}

TEST(a_string_argument_keeps_its_bytes_and_an_object_argument_is_json) {
    auto p = parser(std::string(kThink));
    const std::string s = std::string("</think:opensource>") + kCalls + kCall + "weather" +
                          kKey + "city" + kKeyEnd + kVal + "  two\nlines <b>&\"q\"  " + kValEnd + kCallEnd + kCallsEnd;
    const ReplyMessage m = reply_parse(*p, s);
    REQUIRE_EQ(m.calls.size(), (size_t)1);
    CHECK_EQ(json::parse(m.calls[0].arguments)["city"].get<std::string>(),
             std::string("  two\nlines <b>&\"q\"  "));
}

/* ------------------------------------------------------------------ the template's own turns */

/* The last assistant turn of a render, past its header and before the terminator. */
static std::string last_turn(const std::string& prompt) {
    const std::string head = std::string(kBos) + "assistant<｜hy_middle:opensource｜>";
    const size_t at = prompt.rfind(head);
    CHECK(at != std::string::npos);
    std::string t = at == std::string::npos ? std::string() : prompt.substr(at + head.size());
    const size_t end = t.find(kEos);
    if (end != std::string::npos) t.resize(end);
    return t;
}

TEST(replies_the_template_writes_are_read_back_as_the_messages_they_came_from) {
    const json args_edit = json::parse(R"({"city":"Zürich","days":2,"verbose":false,"units":["C"]})");
    const json messages = json::parse(R"([
      {"role":"user","content":"Weather, then ping."},
      {"role":"assistant","reasoning_content":"Both calls.","content":"On it.","tool_calls":[
        {"id":"a","type":"function","function":{"name":"weather",
          "arguments":"{\"city\":\"Zürich\",\"days\":2,\"verbose\":false,\"units\":[\"C\"]}"}},
        {"id":"b","type":"function","function":{"name":"ping","arguments":"{}"}}]}
    ])");
    /* The template needs arguments as a mapping; the engine's apply() converts them. */
    const ChatPrompt p = render(messages, "high", true);
    REQUIRE(p.error.empty());
    const std::string turn = last_turn(p.prompt);
    CHECK(turn.rfind(std::string(kThink) + "Both calls." + kThinkEnd + "On it." + kCalls, 0) == 0);
    /* The history holds the open `<think>` already, so the parser starts from a prompt that ends
     * in it, and the turn it reads is what follows the opener. */
    auto parse = parser(std::string(kThink));
    const ReplyMessage m = reply_parse(*parse, turn.substr(std::string(kThink).size()));
    CHECK_EQ(m.reasoning, std::string("Both calls."));
    CHECK_EQ(m.content, std::string("On it."));
    REQUIRE_EQ(m.calls.size(), (size_t)2);
    CHECK_EQ(m.calls[0].name, std::string("weather"));
    CHECK(json::parse(m.calls[0].arguments) == args_edit);
    CHECK_EQ(m.calls[1].name, std::string("ping"));
    CHECK(json::parse(m.calls[1].arguments) == json::object());
}

TEST(a_no_think_turn_the_template_writes_has_empty_reasoning_and_its_content) {
    const json messages = json::parse(R"([
      {"role":"user","content":"Hi"},
      {"role":"assistant","content":"Hello there."}])");
    const ChatPrompt p = render(messages, "no_think", false);
    const std::string turn = last_turn(p.prompt);
    CHECK_EQ(turn, std::string(kThink) + kThinkEnd + "Hello there.");
    auto parse = parser(std::string(kThink) + kThinkEnd, false);
    const ReplyMessage m = reply_parse(*parse, turn.substr(std::string(kThink).size() + std::string(kThinkEnd).size()));
    CHECK(m.reasoning.empty());
    CHECK_EQ(m.content, std::string("Hello there."));
}

RAD_TEST_MAIN()
