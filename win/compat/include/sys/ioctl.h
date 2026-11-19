/* Brilliance POSIX layer: <sys/ioctl.h> -- the terminal size and nothing else. */
#pragma once
#include "brl_compat.h"

struct winsize {
    unsigned short ws_row, ws_col, ws_xpixel, ws_ypixel;
};
#define TIOCGWINSZ 0x5413

BRL_C_BEGIN
/* Only TIOCGWINSZ, answered from the console screen buffer of the handle behind `fd`. */
int brl_ioctl_winsize(int fd, unsigned long req, struct winsize* ws);
BRL_C_END
#define ioctl brl_ioctl_winsize
