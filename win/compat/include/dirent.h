/* Brilliance POSIX layer: <dirent.h> over FindFirstFileExW. Names are returned in UTF-8. */
#pragma once
#include "brl_compat.h"

#define DT_UNKNOWN 0
#define DT_DIR     4
#define DT_REG     8
#define DT_LNK     10

struct dirent {
    unsigned char d_type;
    char          d_name[1024];   /* UTF-8; a 260-UTF-16-unit name is at most 780 bytes */
};

typedef struct brl_DIR DIR;

BRL_C_BEGIN
DIR*           opendir(const char* path);
struct dirent* readdir(DIR* d);
int            closedir(DIR* d);
BRL_C_END
