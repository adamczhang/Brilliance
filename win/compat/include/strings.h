/* Brilliance POSIX layer: <strings.h>. */
#pragma once
#include "brl_compat.h"
#include <string.h>

#ifndef strcasecmp
#define strcasecmp  _stricmp
#define strncasecmp _strnicmp
#endif
