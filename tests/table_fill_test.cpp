/* table_fill_test.cpp -- the Engine's deferred Engram fill (B-01).
 *
 * copy_table reserves and publishes a table's memory as the load's walk meets its weight, starts
 * the read, and join_table_fills waits for every table it started once the container pass has run
 * beside them. These tests drive the same reserve/start/join/release path a --ngram-placement ram
 * server drives for the tables it reads for the cards: the locked copy (a host gather) and the
 * pinned one (a card's gather) on the host backend's memory, and the shared file (ram=DIR) the
 * engines on separate cards pin in place, which only a card's backend can map.
 */
#include "rad_test.h"
#include "engine_priv.h"
#include "device/device.h"
#include "format/radfile.h"
#include "format/table_copy.h"
#include "format/table_share.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace rad {
/* The walk, copy_table and join_table_fills are private, as the other bringup tests reach them. */
struct TableFillTestAccess {
    static void hold(Engine& e, std::unique_ptr<RadFile> f) { e.file_ = std::move(f); }
    static int  copy(Engine& e, const uint8_t* src, int64_t n, int64_t foff, bool card, void** out,
                     std::string* why) {
        return e.copy_table(src, n, foff, card, out, why);
    }
    static int    join(Engine& e) { return e.join_table_fills(); }
    static size_t pending(Engine& e) { return e.pending_tables_.size(); }
    static size_t locked(Engine& e) { return e.table_copies_.size(); }
    static size_t pinned(Engine& e) { return e.card_copies_.size(); }
    static size_t shared(Engine& e) { return e.table_shares_.size(); }
};
}  // namespace rad

namespace {

/* Bytes no 16-byte shift of which can look like another: the fill's check samples them. */
std::vector<uint8_t> pattern_bytes(int64_t n, uint32_t seed) {
    std::vector<uint8_t> b((size_t)n);
    uint32_t x = seed | 1u;
    for (size_t i = 0; i < b.size(); ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        b[i] = (uint8_t)(x >> 11);
    }
    return b;
}

/* A container with three n-gram tables of different sizes, and the mapping the copies read from.
 * c comes first, so the 16-byte shift the failure case reads stays inside the container. */
struct Tables {
    std::string                   path;
    std::vector<uint8_t>          want[3];
    const uint8_t*                src[3] = {nullptr, nullptr, nullptr};
    int64_t                       off[3] = {0, 0, 0};
    int64_t                       n[3]   = {0, 0, 0};
    std::unique_ptr<rad::RadFile> file;

    int open() {
        file = std::make_unique<rad::RadFile>();
        if (file->open(path.c_str()) < 0) return -1;
        const char* names[] = {"ngram.c", "ngram.a", "ngram.b"};
        for (int i = 0; i < 3; ++i) {
            const RadFileEntry* e = file->find(names[i]);
            if (!e) return -1;
            const RadFilePlane& p = file->plane(*e, 0);
            src[i] = file->data(p);
            off[i] = (int64_t)p.offset;
            n[i] = (int64_t)p.bytes;
        }
        return 0;
    }
};

struct TempContainer {
    std::string path;
    ~TempContainer() { std::remove(path.c_str()); }
};

/* Writes the three tables; false when the writer refused. */
bool write_tables(const std::string& path, int64_t nc, int64_t na, int64_t nb, Tables* t) {
    t->path = path;
    t->n[0] = nc;
    t->n[1] = na;
    t->n[2] = nb;
    t->want[0] = pattern_bytes(nc, 0xc2b2ae35u);
    t->want[1] = pattern_bytes(na, 0x9e3779b9u);
    t->want[2] = pattern_bytes(nb, 0x85ebca6bu);
    rad::RadWriter writer;
    if (writer.begin(path.c_str()) < 0) return false;
    writer.set_arch("radtest");
    const char* names[] = {"ngram.c", "ngram.a", "ngram.b"};
    for (int i = 0; i < 3; ++i) {
        rad::RadWriter::Weight w;
        w.name = names[i];
        w.enc = rad_enc_plain(RAD_BF16);
        w.rank = 2;
        w.shape[0] = t->n[i] / 2;   /* bf16: two bytes a row */
        w.shape[1] = 1;
        const void* planes[] = {t->want[i].data()};
        if (writer.add_weight(w, planes) < 0) return false;
    }
    return writer.finish() == RAD_OK;
}

bool same_as(const void* got, const uint8_t* want, int64_t n) {
    return got && std::memcmp(got, want, (size_t)n) == 0;
}

/* Whether the shared file DIR/ngram-<id>.tab is there, whatever it holds. */
bool share_file_exists(const std::string& dir, const std::string& id) {
    return ::access((dir + "/ngram-" + id + ".tab").c_str(), F_OK) == 0;
}

}  // namespace

/* The locked copy (a host gather reads it): two tables started before either is waited on, one
 * copy for the same table a second rank asks for, publication only at the join, and a fill that
 * read the wrong bytes released and failing the load there. */
TEST(a_host_gathers_tables_read_before_the_join_and_are_published_after_it) {
    TempContainer tmp{radtest::tmp_path("table-fill-locked.rad")};
    const int64_t nc = 64ll << 10, na = 3ll << 20, nb = 1ll << 20;
    Tables t;
    REQUIRE(write_tables(tmp.path, nc, na, nb, &t));
    REQUIRE_EQ(t.open(), 0);
    const uint8_t* const* src = t.src;
    const int64_t* n = t.n;
    const int64_t* off = t.off;

    rad::Config config;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "table-fill-locked-test");
    rad::TableFillTestAccess::hold(engine, std::move(t.file));

    void* outa = nullptr;
    void* outb = nullptr;
    std::string why;
    const int rc = rad::TableFillTestAccess::copy(engine, src[1], n[1], off[1], false, &outa, &why);
    /* A locked-memory limit below the tables is the machine's setting, not a defect. */
    if (rc == RAD_E_NOMEM && why.find("mlock2") != std::string::npos) SKIP("%s", why.c_str());
    REQUIRE_EQ(rc, RAD_OK);
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[2], n[2], off[2], false, &outb, &why),
               RAD_OK);
    CHECK(outa != nullptr && outa != (const void*)src[1]);
    CHECK(outb != nullptr && outb != (const void*)src[2]);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(2));
    /* Nothing is published before its bytes are in: the join is what publishes. */
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(0));

    /* The same table for another rank is the one copy, not a second read. */
    void* again = nullptr;
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[1], n[1], off[1], false, &again, &why),
               RAD_OK);
    CHECK(again == outa);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(2));

    REQUIRE_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(2));
    /* Byte for byte, not the fill's samples: the tables are the container's bytes. */
    CHECK(same_as(outa, src[1], n[1]));
    CHECK(same_as(outb, src[2], n[2]));

    /* A fill that read the wrong bytes is not published: the join fails the load and frees it. */
    void* bad = nullptr;
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[0], n[0], off[0] + 16, false, &bad, &why),
               RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(1));
    CHECK_EQ(rad::TableFillTestAccess::join(engine), (int)RAD_E_IO);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(2));   /* the failed one is not */
    /* Joined and cleared: a second join is nothing left to do. */
    CHECK_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
}

/* The pinned copy (a card reads it over the link): the same deferral, with the copies published
 * to the engine's pinned ones only at the join. */
TEST(a_cards_tables_read_before_the_join_and_are_published_after_it) {
    TempContainer tmp{radtest::tmp_path("table-fill-pinned.rad")};
    const int64_t nc = 64ll << 10, na = 3ll << 20, nb = 1ll << 20;
    Tables t;
    REQUIRE(write_tables(tmp.path, nc, na, nb, &t));
    REQUIRE_EQ(t.open(), 0);
    const uint8_t* const* src = t.src;
    const int64_t* n = t.n;
    const int64_t* off = t.off;

    rad::Config config;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "table-fill-pinned-test");
    rad::TableFillTestAccess::hold(engine, std::move(t.file));

    void* outa = nullptr;
    void* outb = nullptr;
    std::string why;
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[1], n[1], off[1], true, &outa, &why),
               RAD_OK);
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[2], n[2], off[2], true, &outb, &why),
               RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(2));
    CHECK_EQ(rad::TableFillTestAccess::pinned(engine), size_t(0));

    REQUIRE_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::pinned(engine), size_t(2));
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(0));
    CHECK(same_as(outa, src[1], n[1]));
    CHECK(same_as(outb, src[2], n[2]));
}

/* --ngram-placement ram=DIR (the third way a table is held): the file is filled by the engine that
 * finds it unready and marked ready at the join, checked and pinned without a read by the next
 * engine on it, and given back — mapping and file both — when the fill that wrote it failed. A
 * table under kTableShareMin keeps its private pinned copy even under ram=DIR. */
TEST(a_shared_table_is_filled_once_and_pinned_by_the_next_engine) {
    /* A file is pinned where it lies (rad_dev_map_file), which only a card's backend can do: the
     * unit entry runs this binary on the host backend and skips here, the _card entry runs it. */
    if (rad::device_is_host())
        SKIP("the shared placement pins a file in place, which the host backend cannot do");

    TempContainer tmp{radtest::tmp_path("table-fill-shared.rad")};
    /* c and a are at the share minimum, so ram=DIR shares them; b is under it. c comes first, so
     * the 16-byte shift the failure case reads stays inside the container. */
    const int64_t nc = rad::kTableShareMin, na = rad::kTableShareMin, nb = 64ll << 10;
    Tables t;
    REQUIRE(write_tables(tmp.path, nc, na, nb, &t));
    REQUIRE_EQ(t.open(), 0);
    const uint8_t* const* src = t.src;
    const int64_t* n = t.n;
    const int64_t* off = t.off;

    const std::string dir = radtest::tmp_path("ngram-share");
    REQUIRE(::mkdir(dir.c_str(), 0777) == 0);   /* an empty DIR is the placement's own */
    const std::string ida = rad::table_share_id(tmp.path, off[1], n[1]);
    /* Table 0 read 16 bytes late: a wrong read is what the check has to catch, and the share's id
     * names the range that was read, so the file the fill makes carries the shifted offset. */
    const int64_t bad_off = off[0] + 16;
    const std::string idc = rad::table_share_id(tmp.path, bad_off, n[0]);
    const std::string idb = rad::table_share_id(tmp.path, off[2], n[2]);
    REQUIRE(!ida.empty() && !idc.empty() && !idb.empty() && ida != idc);

    rad::Config config;
    config.ngram_share_dir = dir;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "table-fill-shared-test");
    rad::TableFillTestAccess::hold(engine, std::move(t.file));

    void* outa = nullptr;
    void* outb = nullptr;
    std::string why;
    const int rc = rad::TableFillTestAccess::copy(engine, src[1], n[1], off[1], true, &outa, &why);
    /* A card whose locked-memory limit is under the table is the machine's setting, not a defect. */
    if (rc == RAD_E_NOMEM && why.find("locking piece") != std::string::npos) SKIP("%s", why.c_str());
    REQUIRE_EQ(rc, RAD_OK);
    CHECK(outa != nullptr && outa != (const void*)src[1]);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(1));
    CHECK_EQ(rad::TableFillTestAccess::pinned(engine), size_t(0));
    /* The file is this engine's to fill: it is not a table another engine may reuse yet. */
    CHECK(!rad::table_share_is_ready(dir, ida, n[1]));
    CHECK(share_file_exists(dir, ida));

    /* Under kTableShareMin: the private pinned copy, and no file in DIR. */
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[2], n[2], off[2], true, &outb, &why),
               RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(2));
    CHECK(!share_file_exists(dir, idb));

    REQUIRE_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::pinned(engine), size_t(2));
    CHECK_EQ(rad::TableFillTestAccess::shared(engine), size_t(1));
    CHECK(same_as(outa, src[1], n[1]));
    CHECK(same_as(outb, src[2], n[2]));
    /* The join is what marks the file ready, for the engines that come after this one. */
    CHECK(rad::table_share_is_ready(dir, ida, n[1]));

    /* The next engine on the same container and DIR checks the file's samples and pins it, with
     * nothing to read: the copy is its own the moment it is made. */
    {
        Tables next;
        next.path = tmp.path;
        REQUIRE_EQ(next.open(), 0);
        rad::Engine engine2(config);
        rad::EngineScope owner2(engine2.owner(), "table-fill-shared-test-reuse");
        rad::TableFillTestAccess::hold(engine2, std::move(next.file));
        void* again = nullptr;
        REQUIRE_EQ(rad::TableFillTestAccess::copy(engine2, next.src[1], next.n[1], next.off[1], true,
                                                 &again, &why),
                   RAD_OK);
        CHECK_EQ(rad::TableFillTestAccess::pending(engine2), size_t(0));
        CHECK_EQ(rad::TableFillTestAccess::pinned(engine2), size_t(1));
        CHECK_EQ(rad::TableFillTestAccess::shared(engine2), size_t(1));
        CHECK(again != nullptr && again != (const void*)next.src[1]);
        CHECK(same_as(again, next.src[1], next.n[1]));
    }

    /* A ready file whose table no longer matches the container (a filler killed mid-write after
     * it was marked ready, or a corrupted tmpfs) is not read: the engine lets go of its mapping
     * and its lock and fails, and the file stays for the holder that filled it. */
    {
        const int fd = ::open((dir + "/ngram-" + ida + ".tab").c_str(), O_RDWR | O_CLOEXEC);
        REQUIRE(fd >= 0);
        uint8_t wrong = 0;
        REQUIRE_EQ(::pread(fd, &wrong, 1, rad::kTableShareHeader), (ssize_t)1);
        wrong = (uint8_t)(wrong ^ 0xff);
        REQUIRE_EQ(::pwrite(fd, &wrong, 1, rad::kTableShareHeader), (ssize_t)1);
        ::close(fd);

        Tables t3;
        t3.path = tmp.path;
        REQUIRE_EQ(t3.open(), 0);
        rad::Engine engine3(config);
        rad::EngineScope owner3(engine3.owner(), "table-fill-shared-test-mismatch");
        rad::TableFillTestAccess::hold(engine3, std::move(t3.file));
        void* stale = nullptr;
        why.clear();
        CHECK_EQ(rad::TableFillTestAccess::copy(engine3, t3.src[1], t3.n[1], t3.off[1], true, &stale,
                                                &why),
                 (int)RAD_E_IO);
        CHECK(why.find("does not match") != std::string::npos);
        CHECK_EQ(rad::TableFillTestAccess::pending(engine3), size_t(0));
        CHECK_EQ(rad::TableFillTestAccess::pinned(engine3), size_t(0));
        CHECK_EQ(rad::TableFillTestAccess::shared(engine3), size_t(0));
        CHECK(share_file_exists(dir, ida));
    }

    /* A fill that read the wrong bytes releases its mapping and the file: nothing half-filled is
     * left for the engine that comes next, and the load fails. */
    void* bad = nullptr;
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[0], n[0], bad_off, true, &bad, &why),
               RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(1));
    CHECK(share_file_exists(dir, idc));
    CHECK_EQ(rad::TableFillTestAccess::join(engine), (int)RAD_E_IO);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::shared(engine), size_t(1));   /* the failed one is not */
    CHECK(!rad::table_share_is_ready(dir, idc, n[0]));
    CHECK(!share_file_exists(dir, idc));
    /* Joined and cleared: a second join is nothing left to do. */
    CHECK_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
}

RAD_TEST_MAIN()
