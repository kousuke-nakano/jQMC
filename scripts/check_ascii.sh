#!/usr/bin/env bash
# Reject non-ASCII bytes in Python sources.
#
# Allowed: printable ASCII (0x20..0x7E), TAB (0x09), LF (0x0A), CR (0x0D).
# Everything else is rejected.

set -e
exec python3 - "$@" <<'PYEOF'
import sys

ALLOWED = set(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0D}

rc = 0
for path in sys.argv[1:]:
    with open(path, "rb") as f:
        data = f.read()
    bad_lines = {}
    line_no = 1
    for b in data:
        if b == 0x0A:
            line_no += 1
        elif b not in ALLOWED:
            bad_lines.setdefault(line_no, set()).add(b)
    if bad_lines:
        text = data.decode("utf-8", errors="replace").splitlines()
        for n in sorted(bad_lines):
            content = text[n - 1] if n - 1 < len(text) else ""
            byte_list = ", ".join(f"0x{b:02X}" for b in sorted(bad_lines[n]))
            print(f"{path}:{n}: bytes [{byte_list}]: {content}")
        print(f"ERROR: non-ASCII detected in {path}")
        rc = 1
sys.exit(rc)
PYEOF
