/* hf_model_test.cpp -- a --model that names a Hugging Face repo (core/hf_model.h).
 *
 * Both questions are answered before anything is fetched; a wrong answer costs a download or a
 * confusing refusal.
 *
 *   Is it a repo   a path that exists is never looked up, and a missing model file is reported
 *                  as missing rather than sent to the hub as a repo name
 *   What to serve  a model repo holds one container beside a README, a recipe and a licence;
 *                  a checkpoint repo holds none and is served as the directory
 *
 * No network: where the fetch itself runs (scripts/hfget.sh, from the build tree's libexec/), a
 * `curl` placed first on PATH answers for the hub.
 */
#include "rad_test.h"

#include "hf_model.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace rad;
namespace fs = std::filesystem;

namespace {

/* An empty directory of its own under the test's temporary directory. */
struct TempDir {
    std::string path;
    TempDir() {
        std::string t = radtest::tmp_path("dir.XXXXXX");
        path = mkdtemp(t.data()) ? t : "";
    }
    void touch(const std::string& name) const {
        if (FILE* f = std::fopen((path + "/" + name).c_str(), "w")) std::fclose(f);
    }
};

}  /* namespace */

TEST(a_repo_id_is_two_names_of_the_hubs_characters) {
    CHECK(hf_repo_id("StillDeadcode/minicpm5-2b-fp8"));
    CHECK(hf_repo_id("StillDeadcode/qwen3.8-27b-mxfp4"));
    CHECK(hf_repo_id("Qwen/Qwen3.8-Flash-Next"));
    CHECK(hf_repo_id("hf-internal-testing/tiny_random.v2"));

    CHECK(!hf_repo_id(""));
    CHECK(!hf_repo_id("minicpm5-2b-fp8"));               /* no owner */
    CHECK(!hf_repo_id("/abs/model"));                     /* absolute */
    CHECK(!hf_repo_id("a/b/c"));                          /* a path, not a repo */
    CHECK(!hf_repo_id("owner/"));
    CHECK(!hf_repo_id("/name"));
    CHECK(!hf_repo_id("../x"));                           /* a relative path, not a name */
    CHECK(!hf_repo_id("./x"));
    CHECK(!hf_repo_id("owner/.hidden"));
    CHECK(!hf_repo_id("-x/y"));                           /* would read as an option */
    CHECK(!hf_repo_id("owner/name with space"));
    CHECK(!hf_repo_id("owner/na$me"));
    CHECK(!hf_repo_id("C:/models"));
}

TEST(a_missing_model_file_is_not_sent_to_the_hub) {
    CHECK(!hf_repo_id("models/qwen.rad"));
    CHECK(!hf_repo_id("ckpt/model.safetensors"));
    CHECK(!hf_repo_id("ckpt/model.gguf"));
    CHECK(!hf_repo_id("ckpt/model.safetensors.index.json"));
}

TEST(a_path_that_exists_is_never_a_repo) {
    TempDir t;
    CHECK(!t.path.empty());
    const std::string cwd = fs::current_path().string();
    CHECK_EQ(chdir(t.path.c_str()), 0);
    fs::create_directories("owner/name");
    CHECK(!hf_repo_id("owner/name"));                     /* exists: a checkpoint directory */
    CHECK(hf_repo_id("owner/other"));                     /* does not: a repo */
    CHECK_EQ(chdir(cwd.c_str()), 0);
}

TEST(the_one_container_in_a_repo_is_what_is_served) {
    TempDir t;
    t.touch("README.md");
    t.touch("LICENSE");
    t.touch("q38-27b.recipe");
    t.touch(".hfget-manifest");
    std::string why;
    CHECK_EQ(hf_served_path(t.path, &why), t.path);       /* no container: a checkpoint */

    t.touch("qwen3.8-27b-mxfp4.rad");
    CHECK_EQ(hf_served_path(t.path, &why), t.path + "/qwen3.8-27b-mxfp4.rad");

    fs::create_directories(t.path + "/sub");
    t.touch("sub/other.rad");                             /* only the top level counts */
    CHECK_EQ(hf_served_path(t.path, &why), t.path + "/qwen3.8-27b-mxfp4.rad");
}

TEST(two_containers_are_refused_by_name) {
    TempDir t;
    t.touch("a.rad");
    t.touch("b.rad");
    std::string why;
    CHECK_EQ(hf_served_path(t.path, &why), std::string());
    CHECK(why.find("a.rad") != std::string::npos);
    CHECK(why.find("b.rad") != std::string::npos);
    CHECK(why.find("--model") != std::string::npos);
}

/* A repo fetched with hfget holds its source weights in the directory; a lone .rad among them
 * -- whoever put it there -- must not be served in their place without one word said. The
 * engine names the ambiguity instead of resolving it by rank. */
TEST(a_container_beside_other_weights_is_not_served_silently) {
    TempDir t;
    t.touch("model-00001-of-00002.safetensors");
    t.touch("model-00002-of-00002.safetensors");
    t.touch("model.safetensors.index.json");
    t.touch("decoy.rad");
    std::string why;
    CHECK_EQ(hf_served_path(t.path, &why), std::string());
    CHECK(why.find("decoy.rad") != std::string::npos);
    CHECK(why.find("--model") != std::string::npos);
    /* gguf competes too, and without any competitor the lone container is still the answer. */
    TempDir g;
    g.touch("model.gguf");
    g.touch("converted.rad");
    CHECK_EQ(hf_served_path(g.path, &why), std::string());
    TempDir r;
    r.touch("qwen.rad");
    r.touch("config.json");
    CHECK_EQ(hf_served_path(r.path, &why), r.path + "/qwen.rad");
}

#if defined(_WIN32)
/* Brilliance: on Windows the default is BRILLIANCE_HF_CACHE, then XDG_CACHE_HOME as upstream reads
 * it, then the per-user cache (%LOCALAPPDATA%\radiance\hf) where Linux falls back to ~/.cache
 * (core/hf_model.cpp). Each variable is restored as it was. */
TEST(the_download_dir_defaults_to_the_xdg_cache) {
    const char* b0 = std::getenv("BRILLIANCE_HF_CACHE");
    const char* x0 = std::getenv("XDG_CACHE_HOME");
    const std::string b = b0 ? b0 : "", x = x0 ? x0 : "";
    _putenv_s("BRILLIANCE_HF_CACHE", "D:/store");
    _putenv_s("XDG_CACHE_HOME", "/c");
    CHECK_EQ(hf_default_download_dir(), std::string("D:/store"));
    _putenv_s("BRILLIANCE_HF_CACHE", "");
    CHECK_EQ(hf_default_download_dir(), std::string("/c/radiance/hf"));
    _putenv_s("XDG_CACHE_HOME", "");
    const std::string per_user = hf_default_download_dir();
    CHECK(has(per_user, "radiance/hf"));
    CHECK(!per_user.empty() && per_user != "/radiance/hf");
    _putenv_s("BRILLIANCE_HF_CACHE", b.c_str());
    _putenv_s("XDG_CACHE_HOME", x.c_str());
}
#else
TEST(the_download_dir_defaults_to_the_xdg_cache) {
    const char* x = std::getenv("XDG_CACHE_HOME");
    const std::string saved = x ? x : "";
    setenv("XDG_CACHE_HOME", "/c", 1);
    CHECK_EQ(hf_default_download_dir(), std::string("/c/radiance/hf"));
    unsetenv("XDG_CACHE_HOME");
    if (const char* h = std::getenv("HOME"); h && *h)
        CHECK_EQ(hf_default_download_dir(), std::string(h) + "/.cache/radiance/hf");
    if (x) setenv("XDG_CACHE_HOME", saved.c_str(), 1);
}
#endif

/* ================================================================== a restart with the hub out of reach */
namespace {

/* Where resolve_hf_model looks for the downloader: libexec/radiance/ beside this binary's
 * directory, where the build copies it. */
std::string hfget_beside_this_binary() {
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    /* Brilliance: .string(), which a narrow-path std::filesystem gives implicitly and a wide one
     * (Windows) does not. */
    return (exe.parent_path() / ".." / "libexec" / "radiance" / "hfget.sh").lexically_normal().string();
}
#if defined(_WIN32)
/* Brilliance: these cases drive hfget.sh under sh with a fake curl first on PATH. Windows fetches
 * with win/libexec/hfget.ps1, which calls System32's curl.exe by its full path, so a fake on PATH
 * would not answer for the hub; its own cases, against a stand-in hub through HF_ENDPOINT, are
 * the win_ tests at the end of this file. */
#define NEED_HFGET() SKIP("these cases fake curl for hfget.sh; Windows fetches with hfget.ps1 (the win_ cases)")
#else
#define NEED_HFGET() NEED_ARTIFACT(hfget_beside_this_binary())
#endif

bool on_path(const char* tool) {
    const char* path = std::getenv("PATH");
    std::string all = path ? path : "";
    size_t at = 0;
    while (at <= all.size()) {
        size_t c = all.find(':', at);
        if (c == std::string::npos) c = all.size();
        const std::string d = all.substr(at, c - at);
        if (!d.empty() && ::access((d + "/" + tool).c_str(), X_OK) == 0) return true;
        at = c + 1;
    }
    return false;
}

/* PATH with a directory holding a `curl` script first, restored on destruction. */
struct FakeCurl {
    TempDir bin;
    std::string saved;
    explicit FakeCurl(const std::string& script) {
        const char* p = std::getenv("PATH");
        saved = p ? p : "";
        const std::string f = bin.path + "/curl";
        if (FILE* o = std::fopen(f.c_str(), "w")) {
            std::fputs(script.c_str(), o);
            std::fclose(o);
        }
        ::chmod(f.c_str(), 0755);
        setenv("PATH", (bin.path + ":" + saved).c_str(), 1);
    }
    ~FakeCurl() { setenv("PATH", saved.c_str(), 1); }
};

/* A hub that cannot be reached: every request fails as curl does without a network. hfget.sh
 * exits 3 (nothing on disk changed). */
const char* kOffline = "#!/bin/sh\nexit 6\n";
/* A hub that lists a file and then fails to send it: hfget.sh exits 1 (a file may be partial). */
const char* kBrokenTransfer =
    "#!/bin/sh\n"
    "for a in \"$@\"; do case \"$a\" in *api/models*)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"new.bin\",\"size\":5}]'; exit 0;; esac; done\n"
    "exit 22\n";

}  /* namespace */

/* hfget.sh exits 3 when it changed nothing on disk, and a copy a complete fetch left (it has the
 * success manifest) is then served, with a warning, so a restart does not need the hub. Without
 * the manifest the copy may be incomplete; after any other failure a file may be partial; either
 * way nothing is served. */
TEST(a_restart_serves_the_complete_copy_when_the_hub_cannot_be_reached) {
    NEED_HFGET();
    if (!on_path("jq")) SKIP("no jq on PATH, which hfget.sh needs");
    TempDir root;
    REQUIRE(!root.path.empty());
    const std::string repo = "radiance-test-owner/radiance-test-model";
    const std::string dest = root.path + "/" + repo;
    fs::create_directories(dest);
    if (FILE* f = std::fopen((dest + "/m.rad").c_str(), "w")) std::fclose(f);

    auto resolve = [&](std::string* model, std::string* log) {
        Config c;
        c.model = repo;
        c.download_dir = root.path;
        bool ok = false;
        *log = stderr_of([&] { ok = resolve_hf_model(c); });
        *model = c.model;
        return ok;
    };
    std::string model, log;
    {
        FakeCurl curl(kOffline);
        /* No manifest: an interrupted first fetch, refused. */
        CHECK(!resolve(&model, &log));
        CHECK(has(log, "exit 3"));

        if (FILE* f = std::fopen((dest + "/.hfget-manifest").c_str(), "w")) {
            std::fputs("0\tm.rad\n", f);
            std::fclose(f);
        }
        CHECK(resolve(&model, &log));
        CHECK_EQ(model, dest + "/m.rad");
        CHECK(has(log, "serving the copy fetched before"));
    }
    {
        FakeCurl curl(kBrokenTransfer);
        CHECK(!resolve(&model, &log));
        CHECK(has(log, "exit 1"));
    }
}

/* A hub whose repo holds safetensors beside a torch copy of the same weights and a codec that only
 * ships as a .pth: the copies (pytorch_model*.pth, a pickle .bin) are skipped, the codec is
 * fetched, since the checkpoint reader reads it (core/format/torch_zip.h). */
const char* kCodecRepo =
    "#!/bin/sh\n"
    "out=''; url=''; prev=''\n"
    "for a in \"$@\"; do\n"
    "  [ \"$prev\" = -o ] && out=\"$a\"\n"
    "  case \"$a\" in https://*) url=\"$a\";; esac\n"
    "  prev=\"$a\"\n"
    "done\n"
    "case \"$url\" in *api/models*)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"model.safetensors\",\"size\":3},"
    "{\"type\":\"file\",\"path\":\"codec.pth\",\"size\":3},"
    "{\"type\":\"file\",\"path\":\"pytorch_model.pth\",\"size\":3},"
    "{\"type\":\"file\",\"path\":\"pytorch_model.bin\",\"size\":3}]'; exit 0;; esac\n"
    "[ -n \"$out\" ] && printf abc > \"$out\"\n"
    "exit 0\n";

TEST(a_pth_that_is_not_the_weights_again_is_fetched_beside_the_safetensors) {
    NEED_HFGET();
    if (!on_path("jq")) SKIP("no jq on PATH, which hfget.sh needs");
    TempDir root;
    REQUIRE(!root.path.empty());
    FakeCurl curl(kCodecRepo);
    const std::string cmd = "sh '" + hfget_beside_this_binary() +
                            "' radiance-test-owner/radiance-test-model '" + root.path +
                            "' >/dev/null 2>&1";
    CHECK_EQ(std::system(cmd.c_str()), 0);
    CHECK(fs::exists(root.path + "/model.safetensors"));
    CHECK(fs::exists(root.path + "/codec.pth"));
    CHECK(!fs::exists(root.path + "/pytorch_model.pth"));
    CHECK(!fs::exists(root.path + "/pytorch_model.bin"));
}

/* A listing that names a file outside the destination -- absolute, or a `..` segment -- is not
 * a listing of a repo at all, so hfget.sh refuses the whole thing rather than half-obeying it:
 * silently skipping the entry would leave a directory that reads as complete and is not, and
 * the manifest would certify it. */
const char* kEscapeListing =
    "#!/bin/sh\n"
    "for a in \"$@\"; do case \"$a\" in *api/models*)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"../escape.bin\",\"size\":1},"
    "{\"type\":\"file\",\"path\":\"kept.bin\",\"size\":1}]'; exit 0;; esac; done\n"
    "exit 22\n";

TEST(a_listing_naming_a_path_outside_the_destination_is_refused) {
    NEED_HFGET();
    if (!on_path("jq")) SKIP("no jq on PATH, which hfget.sh needs");
    TempDir root;
    REQUIRE(!root.path.empty());
    FakeCurl curl(kEscapeListing);
    const std::string cmd = "sh '" + hfget_beside_this_binary() +
                            "' radiance-test-owner/radiance-test-model '" + root.path +
                            "' >/dev/null 2>'" + root.path + "/err'";
    CHECK(std::system(cmd.c_str()) != 0);
    CHECK(!fs::exists(root.path + "/escape.bin"));
    CHECK(!fs::exists(root.path + "/kept.bin"));          /* the whole listing is refused */
    CHECK(!fs::exists(root.path + "/.hfget-manifest"));
    std::string err;
    if (FILE* e = std::fopen((root.path + "/err").c_str(), "r")) {
        char buf[256];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf - 1, e)) > 0) { buf[n] = '\0'; err += buf; }
        std::fclose(e);
    }
    CHECK(has(err, "does not land inside the destination"));
}

/* `#` and `?` end the URL's path: curl requests the stripped object instead. A listing that
 * spells out a truncated name with the truncated object's size and oid passes every check
 * afterwards -- the bytes arrive under a name nobody asked for, and the manifest certifies the
 * directory complete. Refused as an escaping name is. */
const char* kFragmentListing =
    "#!/bin/sh\n"
    "for a in \"$@\"; do case \"$a\" in *api/models*)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"model.bin#x\",\"size\":1},"
    "{\"type\":\"file\",\"path\":\"kept.bin\",\"size\":1}]'; exit 0;; esac; done\n"
    "exit 22\n";

TEST(a_listing_name_that_curl_would_truncate_is_refused_whole) {
    NEED_ARTIFACT(hfget_beside_this_binary());
    if (!on_path("jq")) SKIP("no jq on PATH, which hfget.sh needs");
    TempDir root;
    REQUIRE(!root.path.empty());
    FakeCurl curl(kFragmentListing);
    const std::string cmd = "sh '" + hfget_beside_this_binary() +
                            "' radiance-test-owner/radiance-test-model '" + root.path +
                            "' >/dev/null 2>&1";
    CHECK(std::system(cmd.c_str()) != 0);
    CHECK(!fs::exists(root.path + "/model.bin#x"));
    CHECK(!fs::exists(root.path + "/model.bin"));        /* curl would have fetched this one */
    CHECK(!fs::exists(root.path + "/kept.bin"));         /* the whole listing is refused */
    CHECK(!fs::exists(root.path + "/.hfget-manifest"));
}

/* The per-file child re-screens its name without waiting for the parent: a trailing `..` (the
 * parent's regex caught it, the child's old case did not) and curl's own path terminators must
 * fail there too, before any request or any write. */
TEST(the_child_screen_refuses_trailing_dotdot_and_curl_terminator_names) {
    NEED_ARTIFACT(hfget_beside_this_binary());
    TempDir root;
    REQUIRE(!root.path.empty());
    for (const char* name : {"model.bin#x", "model.bin?y", "sub/.."}) {
        const std::string cmd =
            "HFGET_CHILD=1 HFGET_DEST='" + root.path + "' sh '" + hfget_beside_this_binary() +
            "' \"$(printf '5\\t\\t" + std::string(name) + "')\" >/dev/null 2>&1";
        CHECK(std::system(cmd.c_str()) != 0);
    }
    CHECK(fs::is_empty(root.path));
}

/* An LFS file's oid is the hub's claim about its bytes; the download is kept only once those
 * bytes hash to it -- the size check alone would take a swapped file of the same length, and a
 * spliced resume at any length the hub likes. A mismatch deletes the file and fails the run;
 * a match lands in the manifest, which is what makes "warm" mean verified-warm after. */
const char* kBadOid =
    "#!/bin/sh\n"
    "out=''; url=''; prev=''\n"
    "for a in \"$@\"; do\n"
    "  [ \"$prev\" = -o ] && out=\"$a\"\n"
    "  case \"$a\" in https://*) url=\"$a\";; esac\n"
    "  prev=\"$a\"\n"
    "done\n"
    "case \"$url\" in *api/models*)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"big.bin\",\"size\":10,"
    "\"lfs\":{\"oid\":\"0000000000000000000000000000000000000000000000000000000000000000\"}}]';"
    " exit 0;; esac\n"
    "[ -n \"$out\" ] && printf 0123456789 > \"$out\"\n"
    "exit 0\n";

const char* kGoodOid =
    "#!/bin/sh\n"
    "out=''; url=''; prev=''\n"
    "for a in \"$@\"; do\n"
    "  [ \"$prev\" = -o ] && out=\"$a\"\n"
    "  case \"$a\" in https://*) url=\"$a\";; esac\n"
    "  prev=\"$a\"\n"
    "done\n"
    "case \"$url\" in *api/models*)\n"
    "  oid=$(printf 0123456789 | sha256sum | cut -d' ' -f1)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"big.bin\",\"size\":10,\"lfs\":{\"oid\":\"%s\"}}]'"
    " \"$oid\"; exit 0;; esac\n"
    "[ -n \"$out\" ] && printf 0123456789 > \"$out\"\n"
    "exit 0\n";

TEST(an_lfs_file_that_does_not_hash_to_its_oid_is_not_kept) {
    NEED_HFGET();
    if (!on_path("jq")) SKIP("no jq on PATH, which hfget.sh needs");
    if (!on_path("sha256sum")) SKIP("no sha256sum on PATH, which hfget.sh needs");
    TempDir root;
    REQUIRE(!root.path.empty());
    const std::string fetch = "sh '" + hfget_beside_this_binary() +
                              "' radiance-test-owner/radiance-test-model '" + root.path +
                              "' >/dev/null 2>'" + root.path + "/err'";
    {
        FakeCurl curl(kBadOid);
        CHECK(std::system(fetch.c_str()) != 0);
        CHECK(!fs::exists(root.path + "/big.bin"));        /* deleted, not left to be resumed */
        CHECK(!fs::exists(root.path + "/.hfget-manifest"));
        std::string err;
        if (FILE* e = std::fopen((root.path + "/err").c_str(), "r")) {
            char buf[256];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf - 1, e)) > 0) { buf[n] = '\0'; err += buf; }
            std::fclose(e);
        }
        CHECK(has(err, "hashes to"));
    }
    {
        FakeCurl curl(kGoodOid);
        CHECK_EQ(std::system(fetch.c_str()), 0);
        CHECK(fs::exists(root.path + "/big.bin"));
        CHECK_EQ(fs::file_size(root.path + "/big.bin"), 10u);
        CHECK(fs::exists(root.path + "/.hfget-manifest"));  /* the verified copy is certified */
    }
}

/* --model-revision names the revision the fetch reads: the pinned ref is the one listed, and
 * an unpinned fetch reads the repo's moving default branch -- which a deployment must be able
 * to refuse outright rather than restart onto. */
const char* kPinnedOnly =
    "#!/bin/sh\n"
    "out=''; url=''; prev=''\n"
    "for a in \"$@\"; do\n"
    "  [ \"$prev\" = -o ] && out=\"$a\"\n"
    "  case \"$a\" in https://*) url=\"$a\";; esac\n"
    "  prev=\"$a\"\n"
    "done\n"
    "case \"$url\" in *api/models*/tree/refs/pr/42*)\n"
    "  printf '[{\"type\":\"file\",\"path\":\"m.rad\",\"size\":10}]'; exit 0;; esac\n"
    "case \"$url\" in *api/models*) exit 22;; esac\n"
    "[ -n \"$out\" ] && printf 0123456789 > \"$out\"\n"
    "exit 0\n";

TEST(the_pinned_revision_is_the_revision_fetched) {
    NEED_HFGET();
    if (!on_path("jq")) SKIP("no jq on PATH, which hfget.sh needs");
    TempDir root;
    REQUIRE(!root.path.empty());
    FakeCurl curl(kPinnedOnly);
    const std::string repo = "radiance-test-owner/radiance-test-model";
    auto resolve = [&](const char* rev, std::string* model) {
        Config c;
        c.model = repo;
        c.download_dir = root.path;
        if (rev) c.model_revision = rev;
        bool ok = false;
        stderr_of([&] { ok = resolve_hf_model(c); });
        *model = c.model;
        return ok;
    };
    std::string model;
    CHECK(!resolve(nullptr, &model));              /* main: this hub has nothing there */
    CHECK(resolve("refs/pr/42", &model));
    CHECK_EQ(model, root.path + "/" + repo + "/m.rad");
}

/* With no --download-dir and neither XDG_CACHE_HOME nor HOME there is nowhere to fetch to, which
 * is refused by name before anything runs. */
TEST(a_repo_id_with_nowhere_to_fetch_to_is_refused) {
#if defined(_WIN32)
    /* Brilliance: Windows always has a per-user cache (the known folder, not an environment
     * variable), so a repo id always has somewhere to go: the defaults are the test above. */
    SKIP("a Windows account always has a per-user cache; see the_download_dir_defaults_to_the_xdg_cache");
#endif
    const char* x = std::getenv("XDG_CACHE_HOME");
    const char* h = std::getenv("HOME");
    const std::string sx = x ? x : "", sh = h ? h : "";
    unsetenv("XDG_CACHE_HOME");
    unsetenv("HOME");
    Config c;
    c.model = "radiance-test-owner/radiance-test-model";
    bool ok = true;
    const std::string log = stderr_of([&] { ok = resolve_hf_model(c); });
    CHECK(!ok);
    CHECK(has(log, "--download-dir"));
    CHECK_EQ(c.model, std::string("radiance-test-owner/radiance-test-model"));
    if (x) setenv("XDG_CACHE_HOME", sx.c_str(), 1);
    if (h) setenv("HOME", sh.c_str(), 1);
}

#if defined(_WIN32)
/* ================================================================== Brilliance: hfget.ps1's twins
 *
 * The cases above fake curl on PATH for hfget.sh. Windows fetches with win/libexec/hfget.ps1, which
 * lists with Invoke-WebRequest and downloads with System32's curl.exe by its full path, so the
 * stand-in here is a real hub on loopback (cpp-httplib), reached through HF_ENDPOINT: the same
 * questions, asked of the script Windows runs. Each case starts its own hub on a free port. */
#include "httplib.h"
#include <functional>
#include <thread>
#include <windows.h>
namespace {

using Handler = std::function<void(const httplib::Request&, httplib::Response&)>;

/* A hub on 127.0.0.1: `api` answers /api/models/..., `file` answers /<repo>/resolve/<rev>/<path>.
 * HF_ENDPOINT points at it for as long as it lives. */
struct StandInHub {
    httplib::Server srv;
    std::thread th;
    int port = 0;
    StandInHub(Handler api, Handler file) {
        srv.Get(R"(/api/models/.*)", api);
        srv.Get(R"(/[^/]+/[^/]+/resolve/.*)", file);
        port = srv.bind_to_any_port("127.0.0.1");
        th = std::thread([this] { srv.listen_after_bind(); });
        srv.wait_until_ready();
        _putenv_s("HF_ENDPOINT", ("http://127.0.0.1:" + std::to_string(port)).c_str());
    }
    ~StandInHub() {
        srv.stop();
        if (th.joinable()) th.join();
        _putenv_s("HF_ENDPOINT", "");
    }
};

/* A listing of the given JSON array, and a file handler sending `bytes` for every path. */
Handler listing(std::string json) {
    return [json](const httplib::Request&, httplib::Response& r) { r.set_content(json, "application/json"); };
}
Handler sends(std::string bytes) {
    return [bytes](const httplib::Request&, httplib::Response& r) { r.set_content(bytes, "application/octet-stream"); };
}
Handler fails() {
    return [](const httplib::Request&, httplib::Response& r) { r.status = 500; };
}

std::string exe_dir() {
    char b[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, b, MAX_PATH);
    return fs::path(b).parent_path().string();
}

/* hfget.ps1 where the build puts it for the engine (core/hf_model.cpp, hfget_path): libexec\brilliance
 * beside the directory of this binary's directory. */
std::string hfget_ps1() {
    return (fs::path(exe_dir()) / ".." / "libexec" / "brilliance" / "hfget.ps1").lexically_normal().string();
}

/* Runs hfget.ps1 <repo> <dest> [rev]; its exit status, with stderr in *err. */
int run_hfget(const std::string& dest, std::string* err, const char* rev = nullptr) {
    char sys[MAX_PATH] = {};
    GetSystemDirectoryA(sys, MAX_PATH);
    const std::string errf = dest + "\\..\\hfget-err.txt";
    std::string cmd = std::string("\"\"") + sys + "\\WindowsPowerShell\\v1.0\\powershell.exe\" -NoProfile -ExecutionPolicy Bypass -File \"" +
                      hfget_ps1() + "\" radiance-test-owner/radiance-test-model \"" + dest + "\"" +
                      (rev ? std::string(" ") + rev : std::string()) + " >nul 2>\"" + errf + "\"\"";
    const int rc = std::system(cmd.c_str());
    err->clear();
    if (FILE* e = std::fopen(errf.c_str(), "r")) {
        char buf[512];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf - 1, e)) > 0) { buf[n] = '\0'; *err += buf; }
        std::fclose(e);
    }
    return rc;
}

/* A port nothing listens on: bound once and released. */
int closed_port() {
    httplib::Server s;
    const int p = s.bind_to_any_port("127.0.0.1");
    s.stop();
    return p;
}

#define NEED_HFGET_PS1() \
    if (!fs::exists(hfget_ps1())) SKIP("no libexec\\brilliance\\hfget.ps1 beside this build")

}  /* namespace */

TEST(win_a_restart_serves_the_complete_copy_when_the_hub_cannot_be_reached) {
    NEED_HFGET_PS1();
    TempDir root;
    REQUIRE(!root.path.empty());
    const std::string repo = "radiance-test-owner/radiance-test-model";
    const std::string dest = root.path + "/" + repo;
    fs::create_directories(dest);
    if (FILE* f = std::fopen((dest + "/m.rad").c_str(), "w")) std::fclose(f);
    auto resolve = [&](std::string* model, std::string* log) {
        Config c;
        c.model = repo;
        c.download_dir = root.path;
        bool ok = false;
        *log = stderr_of([&] { ok = resolve_hf_model(c); });
        *model = c.model;
        return ok;
    };
    std::string model, log;
    _putenv_s("HF_ENDPOINT", ("http://127.0.0.1:" + std::to_string(closed_port())).c_str());
    CHECK(!resolve(&model, &log));                          /* no manifest: refused */
    CHECK(has(log, "exit 3"));
    if (FILE* f = std::fopen((dest + "/.hfget-manifest").c_str(), "w")) {
        std::fputs("0\tm.rad\n", f);
        std::fclose(f);
    }
    CHECK(resolve(&model, &log));
    CHECK_EQ(fs::path(model).lexically_normal().string(), fs::path(dest + "/m.rad").lexically_normal().string());
    CHECK(has(log, "serving the copy fetched before"));
    _putenv_s("HF_ENDPOINT", "");
    {
        /* A hub that lists a file and then fails to send it: a file may be partial (exit 1). curl
         * retries five times five seconds apart first. */
        StandInHub hub(listing(R"([{"type":"file","path":"new.bin","size":5}])"), fails());
        CHECK(!resolve(&model, &log));
        CHECK(has(log, "exit 1"));
    }
}

TEST(win_a_pth_that_is_not_the_weights_again_is_fetched_beside_the_safetensors) {
    NEED_HFGET_PS1();
    TempDir root;
    REQUIRE(!root.path.empty());
    StandInHub hub(listing(R"([{"type":"file","path":"model.safetensors","size":3},)"
                           R"({"type":"file","path":"codec.pth","size":3},)"
                           R"({"type":"file","path":"pytorch_model.pth","size":3},)"
                           R"({"type":"file","path":"pytorch_model.bin","size":3},)"
                           R"({"type":"file","path":"model.rad","size":3}])"),
                   sends("abc"));
    std::string err;
    CHECK_EQ(run_hfget(root.path, &err), 0);
    CHECK(fs::exists(root.path + "/model.safetensors"));
    CHECK(fs::exists(root.path + "/codec.pth"));
    CHECK(!fs::exists(root.path + "/pytorch_model.pth"));
    CHECK(!fs::exists(root.path + "/pytorch_model.bin"));
    CHECK(!fs::exists(root.path + "/model.rad"));           /* a duplicate format too (upstream 965a7ca) */
}

TEST(win_a_listing_naming_a_path_outside_the_destination_is_refused) {
    NEED_HFGET_PS1();
    /* `..`, the two spellings only Windows reads as somewhere else (a drive and a backslash), and a
     * '#' or '?', where curl would end the URL path and fetch another object under this name
     * (upstream 444b851, carried into hfget.ps1). */
    for (const char* bad : { "../escape.bin", "C:/escape.bin", "sub\\\\escape.bin", "model.bin#x",
                             "model.bin?x=1" }) {
        TempDir root;
        REQUIRE(!root.path.empty());
        StandInHub hub(listing(std::string(R"([{"type":"file","path":")") + bad +
                               R"(","size":1},{"type":"file","path":"kept.bin","size":1}])"),
                       sends("x"));
        std::string err;
        CHECK(run_hfget(root.path, &err) != 0);
        CHECK(!fs::exists(root.path + "/kept.bin"));          /* the whole listing is refused */
        CHECK(!fs::exists(root.path + "/.hfget-manifest"));
        CHECK(has(err, "does not land inside the destination"));
    }
}

TEST(win_an_lfs_file_that_does_not_hash_to_its_oid_is_not_kept) {
    NEED_HFGET_PS1();
    TempDir root;
    REQUIRE(!root.path.empty());
    std::string err;
    {
        StandInHub hub(listing(R"([{"type":"file","path":"big.bin","size":10,"lfs":{"oid":")"
                               "0000000000000000000000000000000000000000000000000000000000000000"
                               R"("}}])"),
                       sends("0123456789"));
        CHECK(run_hfget(root.path, &err) != 0);
        CHECK(!fs::exists(root.path + "/big.bin"));           /* deleted, not left to be resumed */
        CHECK(!fs::exists(root.path + "/.hfget-manifest"));
        CHECK(has(err, "hashes to"));
    }
    {
        /* sha256("0123456789") */
        StandInHub hub(listing(R"([{"type":"file","path":"big.bin","size":10,"lfs":{"oid":")"
                               "84d89877f0d4041efb6bf91a16f0248f2fd573e6af05c19f96bedb9f882f7882"
                               R"("}}])"),
                       sends("0123456789"));
        CHECK_EQ(run_hfget(root.path, &err), 0);
        CHECK(fs::exists(root.path + "/big.bin"));
        CHECK_EQ(fs::file_size(root.path + "/big.bin"), 10u);
        CHECK(fs::exists(root.path + "/.hfget-manifest"));    /* the verified copy is certified */
    }
}

TEST(win_the_pinned_revision_is_the_revision_fetched) {
    NEED_HFGET_PS1();
    TempDir root;
    REQUIRE(!root.path.empty());
    StandInHub hub(
        [](const httplib::Request& q, httplib::Response& r) {
            if (q.path.find("/tree/refs/pr/42") != std::string::npos)
                r.set_content(R"([{"type":"file","path":"m.rad","size":10}])", "application/json");
            else
                r.status = 404;                               /* main: nothing there */
        },
        sends("0123456789"));
    const std::string repo = "radiance-test-owner/radiance-test-model";
    auto resolve = [&](const char* rev, std::string* model) {
        Config c;
        c.model = repo;
        c.download_dir = root.path;
        if (rev) c.model_revision = rev;
        bool ok = false;
        stderr_of([&] { ok = resolve_hf_model(c); });
        *model = c.model;
        return ok;
    };
    std::string model;
    CHECK(!resolve(nullptr, &model));
    CHECK(resolve("refs/pr/42", &model));
    CHECK_EQ(fs::path(model).lexically_normal().string(), fs::path(root.path + "/" + repo + "/m.rad").lexically_normal().string());
}

/* The hub pages its listing (Link: <...>; rel="next"); every page is followed, or a checkpoint
 * would arrive missing the files of the later pages with every size check passing. */
TEST(win_every_page_of_the_listing_is_fetched) {
    NEED_HFGET_PS1();
    TempDir root;
    REQUIRE(!root.path.empty());
    int port = 0;
    StandInHub hub(
        [&port](const httplib::Request& q, httplib::Response& r) {
            if (q.has_param("cursor")) {
                r.set_content(R"([{"type":"file","path":"b.bin","size":1}])", "application/json");
                return;
            }
            r.set_header("Link", "<http://127.0.0.1:" + std::to_string(port) +
                                     "/api/models/radiance-test-owner/radiance-test-model/tree/main"
                                     "?recursive=true&expand=true&cursor=2>; rel=\"next\"");
            r.set_content(R"([{"type":"file","path":"a.bin","size":1}])", "application/json");
        },
        sends("z"));
    port = hub.port;
    std::string err;
    CHECK_EQ(run_hfget(root.path, &err), 0);
    CHECK(fs::exists(root.path + "/a.bin"));
    CHECK(fs::exists(root.path + "/b.bin"));
}
#endif

RAD_TEST_MAIN()
