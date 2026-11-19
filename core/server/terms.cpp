/* core/server/terms.cpp -- Brilliance: the terms-of-use page, GET /terms.
 *
 * Some model licences bind whoever serves the model to other people to pass its use restrictions
 * on to them as enforceable terms (the LTX-2.x Community License, section 4 and Attachment A;
 * CreativeML Open RAIL++-M, paragraph 5). This page is where a server's users find those terms:
 * the operator's own file when `server.terms-file` names one, else the template shipped as
 * win/TERMS-OF-USE.md, compiled in (terms_template.h, written by core/CMakeLists.txt) so the binary
 * serves it wherever it is installed. Not an upstream route.
 *
 * The page is read once, at startup (load_terms_page, from fleet::run), and served from memory.
 * A file named by the operator that cannot be served stops the start, so a server never answers
 * with terms other than the ones its operator chose.
 *
 * GET /terms/licenses serves the full licence texts the page refers to: an index
 * (win/model-licenses/index.md) at /terms/licenses, and each licence at /terms/licenses/<id>, the
 * id being its file name in win/model-licenses without the extension. They are compiled in the
 * same way and are not the operator's to replace: they are the model makers' texts.
 */
#include "server.h"

#include "rad_internal.h"
#include "terms_template.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>

namespace rad {
namespace server {

namespace {

struct TermsPage {
    std::string body;
    bool        html = false;
};

std::mutex                       g_mu;
std::shared_ptr<const TermsPage> g_page;   /* null: the built-in template */

std::shared_ptr<const TermsPage> builtin_page() {
    static const std::shared_ptr<const TermsPage> t = std::make_shared<const TermsPage>(TermsPage{
        std::string(reinterpret_cast<const char*>(kTermsTemplate), kTermsTemplateSize), false });
    return t;
}

/* Well-formed UTF-8 (RFC 3629): no overlong forms, no surrogates, nothing past U+10FFFF. */
bool utf8_valid(const std::string& s) {
    const auto* p = reinterpret_cast<const unsigned char*>(s.data());
    const size_t n = s.size();
    for (size_t i = 0; i < n;) {
        const unsigned c = p[i];
        size_t len = 0;
        uint32_t cp = 0, min = 0;
        if (c < 0x80)                { ++i; continue; }
        else if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; min = 0x80; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; min = 0x800; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; min = 0x10000; }
        else return false;
        if (i + len > n) return false;
        for (size_t k = 1; k < len; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (p[i + k] & 0x3F);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        i += len;
    }
    return true;
}

bool html_name(const std::filesystem::path& p) {
    std::string e = p.extension().string();
    for (char& ch : e) ch = (char)std::tolower((unsigned char)ch);
    return e == ".html" || e == ".htm";
}

}  /* namespace */

int load_terms_page(const std::string& path) {
    if (path.empty()) {
        std::lock_guard<std::mutex> lk(g_mu);
        g_page.reset();
        RAD_INFO("server: terms of use at /terms: the built-in template (set server.terms-file "
                 "to serve your own)");
        return RAD_OK;
    }
    namespace fs = std::filesystem;
    /* The path is UTF-8 whatever the platform; a char path on Windows would be read in the ANSI
     * code page. */
    const fs::path p(std::u8string(reinterpret_cast<const char8_t*>(path.data()), path.size()));
    const char* key = "server.terms-file";
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) {
        RAD_ERR("%s %s: no such file. Name the file with your terms of use, or remove the key to "
                "serve the built-in template", key, path.c_str());
        return RAD_E_IO;
    }
    const uintmax_t size = fs::file_size(p, ec);
    if (ec) {
        RAD_ERR("%s %s: cannot read the file's size: %s", key, path.c_str(), ec.message().c_str());
        return RAD_E_IO;
    }
    if (size == 0) {
        RAD_ERR("%s %s: the file is empty", key, path.c_str());
        return RAD_E_INVAL;
    }
    if (size > kMaxTermsBytes) {
        RAD_ERR("%s %s: the file is %llu bytes; the page takes at most %zu", key, path.c_str(),
                (unsigned long long)size, kMaxTermsBytes);
        return RAD_E_INVAL;
    }
    std::ifstream f(p, std::ios::binary);
    std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (!f.is_open() || body.size() != size) {
        RAD_ERR("%s %s: cannot read the file", key, path.c_str());
        return RAD_E_IO;
    }
    /* A byte-order mark, as Windows editors write one, is not part of the page. */
    if (body.compare(0, 3, "\xEF\xBB\xBF") == 0) body.erase(0, 3);
    if (!utf8_valid(body)) {
        RAD_ERR("%s %s: the file is not UTF-8 text; save it as UTF-8", key, path.c_str());
        return RAD_E_INVAL;
    }
    const bool html = html_name(p);
    auto page = std::make_shared<TermsPage>();
    page->html = html;
    page->body = std::move(body);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_page = std::move(page);
    }
    RAD_INFO("server: terms of use at /terms: %s (as %s)", path.c_str(),
             html ? "HTML" : "plain text");
    return RAD_OK;
}

namespace {

/* The id's licence file, or null; "index" is the list itself, served at /terms/licenses only. */
const TermsFile* licence_file(const std::string& id) {
    for (const TermsFile& f : kLicenceFiles)
        if (id == f.id) return &f;
    return nullptr;
}

HttpResponse page_response(std::string body, bool html) {
    HttpResponse r;
    r.status = 200;
    r.content_type = html ? "text/html; charset=utf-8" : "text/plain; charset=utf-8";
    r.body = std::move(body);
    r.set_header("Cache-Control", "no-cache");
    /* An operator's HTML page is one document: inline styles and data: pictures, no script, and
     * nothing fetched from anywhere, this origin included. */
    r.set_header("Content-Security-Policy",
                 "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
                 "frame-ancestors 'none'; base-uri 'none'; form-action 'none'");
    r.set_header("X-Frame-Options", "DENY");
    return r;
}

}  /* namespace */

HttpResponse terms_page() {
    std::shared_ptr<const TermsPage> page;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        page = g_page ? g_page : builtin_page();
    }
    return page_response(page->body, page->html);
}

HttpResponse licence_page(const std::string& id) {
    const TermsFile* f = licence_file(id.empty() ? "index" : id);
    if (!f || id == "index") {
        return HttpResponse::text(404, "no licence named '" + id +
                                           "' here; the list is at /terms/licenses\n");
    }
    return page_response(std::string(reinterpret_cast<const char*>(f->data), f->size), false);
}

bool terms_path(const std::string& path) {
    static const std::string kList = "/terms/licenses";
    if (path == "/terms" || path == kList) return true;
    if (path.size() <= kList.size() + 1 || path.compare(0, kList.size() + 1, kList + "/") != 0)
        return false;
    const std::string id = path.substr(kList.size() + 1);
    return id != "index" && licence_file(id) != nullptr;
}

}  /* namespace server */
}  /* namespace rad */
