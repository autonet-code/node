#!/usr/bin/env python3
"""Send one command to the mesh node in this container (mesh_node.py) and
print its JSON reply.

    python mesh_ctl.py status
    python mesh_ctl.py publish_tool '{"name": "t1"}'
"""
import json
import socket
import sys


def main() -> int:
    req = {"cmd": sys.argv[1]}
    if len(sys.argv) > 2:
        req.update(json.loads(sys.argv[2]))
    port = 9000
    with socket.create_connection(("127.0.0.1", port), timeout=120) as s:
        s.sendall(json.dumps(req).encode() + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    sys.stdout.write(buf.decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
