/* terms_test.cpp -- Brilliance: the terms-of-use page, GET /terms (core/server/terms.cpp).
 *
 *   Template  with no file named, the page is win/TERMS-OF-USE.md as the binary carries it, the
 *             licence restrictions it quotes included
 *   Operator  a file named by server.terms-file replaces it: Markdown or text as plain text,
 *             .html as HTML, a byte-order mark dropped
 *   Refused   a file that cannot be served stops the start, by name, and leaves the page as it was
 *   Taken     terms-file is a server: key and a command-line flag
 *
 * No device, no model, no socket: files in, answers out.
 */
#include "rad_test.h"

#include "rad_core.h"
#include "fleet/config.h"
#include "server/server.h"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace rad;
using namespace rad::server;

namespace {

std::string read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::string write_file(const std::string& leaf, const std::string& bytes) {
    const std::string p = radtest::tmp_path(leaf);
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << bytes;
    return p;
}

std::string header(const HttpResponse& r, const std::string& name) {
    for (const auto& [k, v] : r.headers)
        if (k == name) return v;
    return "";
}

/* The page as a server answers it. */
HttpResponse get_terms() {
    Server srv(Deps{}, ServerOptions{});
    return srv.router().call("GET", "/terms");
}

const std::string kTemplate = RAD_CHAT_FIXTURES "/../../../win/TERMS-OF-USE.md";

}  // namespace

/* ------------------------------------------------------------------ template */

TEST(with_no_file_named_the_page_is_the_shipped_template) {
    CHECK_EQ(load_terms_page(""), (int)RAD_OK);
    const HttpResponse r = get_terms();
    CHECK_EQ(r.status, 200);
    CHECK_EQ(r.content_type, std::string("text/plain; charset=utf-8"));
    const std::string file = read_all(kTemplate);
    CHECK(!file.empty());
    /* Compiled in from the file: a stale copy in the binary fails here. */
    CHECK(r.body == file);
    CHECK(has(r.body, "# Terms of use"));
    CHECK(has(header(r, "Content-Security-Policy"), "default-src 'none'"));
    /* The same page before any load: terms_page() never answers empty. */
    CHECK(terms_page().body == file);
}

TEST(the_template_carries_the_ltx_restrictions_and_the_policies_by_address) {
    const std::string t = terms_page().body;
    CHECK(has(t, "4. Use-based restrictions. The restrictions set forth in Attachment A"));
    CHECK(has(t, "use LTX-2.x or a Derivative of LTX-2.x to comply with the terms of\n      this Section 4."));
    CHECK(has(t, "Use Restrictions"));
    CHECK(has(t, "https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf"));
    CHECK(has(t, "without expressly and intelligibly disclaiming that the"));
    CHECK(has(t, "obtaining a separate commercial license from Licensor."));
    CHECK(has(t, "machine generated"));
    CHECK(has(t, "CreativeML Open RAIL++-M"));
    CHECK(has(t, "https://ai.google.dev/gemma/prohibited_use_policy"));
    CHECK(has(t, "not legal\n> advice"));
}

/* ------------------------------------------------------------------ operator */

TEST(an_operator_file_replaces_the_template) {
    const std::string md = write_file("terms.md", "# Our terms\n\nBe kind. \xC3\xA9t\xC3\xA9\n");
    CHECK_EQ(load_terms_page(md), (int)RAD_OK);
    HttpResponse r = get_terms();
    CHECK_EQ(r.status, 200);
    CHECK_EQ(r.content_type, std::string("text/plain; charset=utf-8"));
    CHECK_EQ(r.body, std::string("# Our terms\n\nBe kind. \xC3\xA9t\xC3\xA9\n"));

    /* .html (any case) is served as HTML, under a policy that runs no script and fetches
     * nothing. A byte-order mark is dropped. */
    const std::string html = write_file("terms.HTML", "\xEF\xBB\xBF<p>Our terms</p>");
    CHECK_EQ(load_terms_page(html), (int)RAD_OK);
    r = get_terms();
    CHECK_EQ(r.content_type, std::string("text/html; charset=utf-8"));
    CHECK_EQ(r.body, std::string("<p>Our terms</p>"));
    CHECK(has(header(r, "Content-Security-Policy"), "default-src 'none'"));
    CHECK(!has(header(r, "Content-Security-Policy"), "script-src"));
    CHECK_EQ(header(r, "X-Frame-Options"), std::string("DENY"));

    /* An empty name goes back to the template. */
    CHECK_EQ(load_terms_page(""), (int)RAD_OK);
    CHECK(get_terms().body == read_all(kTemplate));
}

TEST(loading_names_the_page_in_the_log) {
    const std::string md = write_file("logged.md", "terms\n");
    std::string log = stderr_of([&] { CHECK_EQ(load_terms_page(md), (int)RAD_OK); });
    CHECK(has(log, "terms of use at /terms: " + md));
    log = stderr_of([&] { CHECK_EQ(load_terms_page(""), (int)RAD_OK); });
    CHECK(has(log, "terms of use at /terms: the built-in template"));
}

/* ------------------------------------------------------------------ refused */

TEST(a_file_that_cannot_be_served_stops_the_start_by_name) {
    const std::string kept = write_file("kept.txt", "kept terms\n");
    CHECK_EQ(load_terms_page(kept), (int)RAD_OK);
    struct Case { std::string path; int status; const char* says; };
    std::string big(kMaxTermsBytes + 1, 'a');
    const Case cases[] = {
        { radtest::tmp_path("no-such-terms.md"), RAD_E_IO, "no such file" },
        { radtest::tmpdir(), RAD_E_IO, "no such file" },
        { write_file("empty.md", ""), RAD_E_INVAL, "empty" },
        { write_file("latin1.md", "caf\xE9\n"), RAD_E_INVAL, "not UTF-8" },
        { write_file("overlong.md", "\xC0\xAF"), RAD_E_INVAL, "not UTF-8" },
        { write_file("big.md", big), RAD_E_INVAL, "at most" },
    };
    for (const Case& c : cases) {
        int st = RAD_OK;
        const std::string log = stderr_of([&] { st = load_terms_page(c.path); });
        CHECK_EQ(st, c.status);
        CHECK(has(log, "server.terms-file " + c.path));
        CHECK(has(log, c.says));
        /* The page in force stays; the start stops anyway. */
        CHECK_EQ(get_terms().body, std::string("kept terms\n"));
    }
    CHECK_EQ(load_terms_page(""), (int)RAD_OK);
}

/* ------------------------------------------------------------------ taken */

TEST(terms_file_is_a_server_key_and_a_flag) {
    fleet::FleetConfig fc;
    CHECK_EQ(fleet::parse("server:\n  terms-file: D:/srv/terms.md\nmodels:\n  a:\n    model: /m/a.rad\n"
                          "  b:\n    model: /m/b.rad\n", "fleet.yaml", &fc), (int)RAD_OK);
    CHECK_EQ(fc.models.size(), 2u);
    for (const fleet::ModelEntry& m : fc.models) {
        CHECK_EQ(m.cfg.terms_file, std::string("D:/srv/terms.md"));
        CHECK_EQ(m.cfg.given.at("--terms-file"), std::string("server.terms-file"));
    }
    /* A model may not set it: it is the process's. */
    fleet::FleetConfig fm;
    std::string log = stderr_of([&] {
        CHECK(fleet::parse("models:\n  a:\n    model: /m/a.rad\n    terms-file: /t.md\n", "fleet.yaml",
                           &fm) != RAD_OK);
    });
    CHECK(has(log, "terms-file"));
    /* An empty name is refused; not naming one serves the template. */
    fleet::FleetConfig fe;
    log = stderr_of([&] {
        CHECK(fleet::parse("server:\n  terms-file: \"\"\nmodels:\n  a:\n    model: /m/a.rad\n",
                           "fleet.yaml", &fe) != RAD_OK);
    });
    CHECK(has(log, "terms-file"));
    fleet::FleetConfig fd;
    CHECK_EQ(fleet::parse("models:\n  a:\n    model: /m/a.rad\n", "fleet.yaml", &fd), (int)RAD_OK);
    CHECK(fd.models[0].cfg.terms_file.empty());
}

/* ------------------------------------------------------------------ licences */

namespace {

const std::string kLicences = RAD_CHAT_FIXTURES "/../../../win/model-licenses/";

/* Every /terms/licenses/<id> a text names, in order of appearance. */
std::vector<std::string> licence_links(const std::string& text) {
    std::vector<std::string> out;
    const std::string pre = "/terms/licenses/";
    for (size_t at = text.find(pre); at != std::string::npos; at = text.find(pre, at + 1)) {
        size_t e = at + pre.size();
        while (e < text.size() && (std::isalnum((unsigned char)text[e]) || text[e] == '-' ||
                                   text[e] == '.' || text[e] == '_'))
            ++e;
        std::string id = text.substr(at + pre.size(), e - at - pre.size());
        while (!id.empty() && id.back() == '.') id.pop_back();   /* a sentence's full stop */
        out.push_back(id);
    }
    return out;
}

}  // namespace

TEST(the_licence_list_is_served_and_every_link_in_it_answers_its_file) {
    Server srv(Deps{}, ServerOptions{});
    const HttpResponse list = srv.router().call("GET", "/terms/licenses");
    CHECK_EQ(list.status, 200);
    CHECK_EQ(list.content_type, std::string("text/plain; charset=utf-8"));
    CHECK(list.body == read_all(kLicences + "index.md"));
    CHECK(has(list.body, "https://ai.google.dev/gemma/docs/gemma_4_license"));
    const std::vector<std::string> ids = licence_links(list.body);
    CHECK(ids.size() >= 7);
    for (const std::string& id : ids) {
        const HttpResponse r = srv.router().call("GET", "/terms/licenses/" + id);
        CHECK_EQ(r.status, 200);
        CHECK(has(header(r, "Content-Security-Policy"), "default-src 'none'"));
        /* The served text is the shipped file, byte for byte. */
        const std::string file = read_all(kLicences + id + ".txt");
        CHECK(!file.empty());
        if (r.body != file) std::fprintf(stderr, "    %s differs from its file\n", id.c_str());
        CHECK(r.body == file);
        CHECK(terms_path("/terms/licenses/" + id));
    }
    /* Each licence the template points to is there too. */
    for (const std::string& id : licence_links(read_all(kTemplate))) {
        CHECK_EQ(srv.router().call("GET", "/terms/licenses/" + id).status, 200);
        CHECK(terms_path("/terms/licenses/" + id));
    }
    CHECK(has(read_all(kTemplate), "`/terms/licenses` on this server"));
}

TEST(the_licences_are_the_makers_texts) {
    const std::string ltx = licence_page("ltx-2.x-community-license").body;
    CHECK(has(ltx, "LTX-2.x Community License Agreement"));
    CHECK(has(ltx, "substitute Licensor's offerings in the market, without"));
    const std::string sdxl = licence_page("creativeml-openrail-pp-m").body;
    CHECK(has(sdxl, "CreativeML Open RAIL++-M License dated July 26, 2023"));
    CHECK(has(sdxl, "comply with the terms of this paragraph (paragraph 5)."));
    CHECK(has(licence_page("apache-2.0").body, "Version 2.0, January 2004"));
    CHECK(has(licence_page("qwen-community-1.0").body, "Model as a Service"));
    CHECK(has(licence_page("openmdw-1.1").body, "OpenMDW License Agreement, version 1.1"));
    CHECK(licence_page("").body == read_all(kLicences + "index.md"));
    /* No carriage returns, no byte-order marks: the repository's bytes. */
    for (const char* id : { "ltx-2.x-community-license", "creativeml-openrail-pp-m", "apache-2.0",
                            "qwen-community-1.0", "openmdw-1.1", "ltx-2.5-distilled-iq8r-notice",
                            "nemotron-3.5-rerank-notice" }) {
        const std::string b = licence_page(id).body;
        CHECK(!b.empty() && b.find('\r') == std::string::npos && b.compare(0, 3, "\xEF\xBB\xBF") != 0);
    }
}

TEST(an_unknown_licence_is_a_404_and_outside_the_open_set) {
    Server srv(Deps{}, ServerOptions{});
    for (const char* id : { "nope", "index", "APACHE-2.0", "apache-2.0.txt", ".." }) {
        const std::string p = std::string("/terms/licenses/") + id;
        const HttpResponse r = srv.router().call("GET", p);
        CHECK(r.status == 404);
        CHECK(!terms_path(p));
    }
    CHECK(terms_path("/terms"));
    CHECK(terms_path("/terms/licenses"));
    for (const char* p : { "/terms/", "/terms/licenses/", "/Terms", "/terms/licenses/apache-2.0/",
                           "/terms/licenses/apache-2.0/x", "/terms/licensesx", "/stats" })
        CHECK(!terms_path(p));
}

RAD_TEST_MAIN()
