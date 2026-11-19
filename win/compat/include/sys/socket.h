/* Brilliance POSIX layer: <sys/socket.h> -> Winsock 2. A socket is a SOCKET, not an fd: close()
 * on it is wrong (closesocket), so code that manages sockets itself still needs a Windows branch.
 * cpp-httplib, which runs the server, already has one. */
#pragma once
#include "brl_compat.h"
#include <winsock2.h>
#include <ws2tcpip.h>
