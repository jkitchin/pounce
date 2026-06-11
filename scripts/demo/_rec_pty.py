#!/usr/bin/env python3
"""Run a command inside a pty of a fixed size, pumping its IO to our stdio.

`asciinema rec` records at the size of its *own* controlling terminal, which
defaults to 80x24 when there is no real tty (e.g. under a CI runner or this
recorder). The solver's iteration table is ~90 columns wide, so at 80 it wraps
into an unreadable mess. Wrapping `asciinema` in a wider pty fixes that.

Usage: _rec_pty.py <rows> <cols> <command> [args...]
"""
import fcntl
import os
import pty
import select
import struct
import sys
import termios

rows, cols = int(sys.argv[1]), int(sys.argv[2])
cmd = sys.argv[3:]

pid, fd = pty.fork()
if pid == 0:  # child: become the target command
    os.execvp(cmd[0], cmd)

# parent: size the child's pty, then shuttle bytes until it exits
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
while True:
    try:
        readable, _, _ = select.select([fd, sys.stdin], [], [])
    except (KeyboardInterrupt, OSError):
        break
    if fd in readable:
        try:
            data = os.read(fd, 4096)
        except OSError:
            break
        if not data:
            break
        os.write(sys.stdout.fileno(), data)
    if sys.stdin in readable:
        try:
            data = os.read(sys.stdin.fileno(), 4096)
        except OSError:
            data = b""
        if data:
            os.write(fd, data)

os.close(fd)
_, status = os.waitpid(pid, 0)
sys.exit(os.waitstatus_to_exitcode(status))
