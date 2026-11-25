/* table_copy.h -- a container table copied into host memory (--ngram-placement ram): locked memory
 * for the host gather (made here), or pinned memory a card reads (allocated by the caller, filled
 * by table_copy_fill). Both share the read and the check.
 *
 * The locked copy uses huge pages rather than locking the mapping: the mapping costs a fault per
 * 4 KiB page and a TLB walk on nearly every scattered row, while a MADV_HUGEPAGE region covers
 * thousands of rows per page. It is locked on first touch (MLOCK_ONFAULT); filled by parallel
 * positional reads, dropping the container's pages behind so the table is not held twice; and
 * checked against the mapping before use, since a wrong offset would silently serve wrong rows.
 * A host kernel reads an address no file backs as plain memory (libavx's table_file). */
#pragma once
#include "radfile.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

namespace rad {

struct TableCopy {
    void*  base  = nullptr;
    size_t bytes = 0;       /* the mapping's length: the table rounded up to 2 MiB */
};

/* One table's read, started and waited on apart from the memory it fills, so several tables read at
 * once: the engine starts the four Engram tables' fills as it meets them and joins them beside its
 * container pass (B-01), instead of reading 59 GiB one table after the other.
 *
 * start() reads [foff, foff + n) of `file` into caller-allocated [dst, dst + n) with `readers`
 * reader threads (each its own span of 64 MiB pieces, dropping the container's pages behind it) and
 * returns at once; `file` must outlive the read. wait() joins them and then runs table_copy_check
 * against `src`. A TableFill destroyed while reading joins first, so no reader outlives the memory
 * its caller would free; move-only. */
class TableFill {
  public:
    TableFill() = default;
    ~TableFill();
    TableFill(TableFill&& o) noexcept = default;
    TableFill& operator=(TableFill&& o) noexcept;   /* joins a read still in flight */
    TableFill(const TableFill&) = delete;
    TableFill& operator=(const TableFill&) = delete;

    int  start(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff, uint8_t* dst,
               int readers = 16);
    /* Joins the readers, then checks the copy: RAD_OK, or an error with `why`. */
    int  wait(std::string* why);
    /* The read's own duration in seconds, once wait() has joined it; 0 before. */
    double seconds() const;

  private:
    struct State;
    std::shared_ptr<State> st_;    /* the readers' result and times */
    const uint8_t*         src_ = nullptr;
    uint8_t*               dst_ = nullptr;
    int64_t                n_ = 0;
    std::thread            th_;
};

/* Copy [src, src + n) of `file`'s mapping (file offset `foff`) into a new locked region. RAD_OK with
 * `out` filled, or an error with `why` and nothing held. */
int  table_copy_make(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff,
                     TableCopy* out, std::string* why);

/* table_copy_make's locked region on its own, for a fill started apart from the reservation and
 * waited on with TableFill::wait (or released with table_copy_free). Nothing is read yet. */
int  table_copy_reserve(int64_t n, TableCopy* out, std::string* why);

/* table_copy_make's read and check, into caller-allocated [dst, dst + n): pinned memory a card
 * reads in place when a device kernel gathers the table. */
int  table_copy_fill(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff,
                     uint8_t* dst, std::string* why);

/* The check table_copy_fill ends with, on its own: samples of [buf, buf + n) against [src, src + n)
 * (the first and last 4 KiB and 64 spread between). RAD_OK, or RAD_E_IO with `why`. For a copy
 * another process filled. */
int  table_copy_check(const uint8_t* src, int64_t n, const uint8_t* buf, std::string* why);

/* --ngram-placement disk=DIR: [src, src + n) read from its own file `path` (the table's bytes and
 * nothing else), mapped read-only in `out` (`bytes` = n). A missing file, or one of another size, is
 * filled from the container under `path`.part and renamed, so a file under `path` is complete
 * (`*filled` true); an existing one is checked as table_copy_check checks. The disk gather finds the
 * file behind the mapping and reads it directly. Release with munmap(out->base, out->bytes). */
int  table_file_open(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff,
                     const std::string& path, TableCopy* out, bool* filled, std::string* why);

/* Unlock and unmap; a no-op for a copy never made. */
void table_copy_free(TableCopy* c);

}  /* namespace rad */
