"""Frame length caps in the p2p reader (nodes/common/p2p.py)."""
import struct

import pytest
import trio

from nodes.common import p2p


class _Stream:
    def __init__(self, data: bytes):
        self.data, self.reads = data, 0

    async def read(self, n):
        self.reads += 1
        chunk, self.data = self.data[:n], self.data[n:]
        return chunk

    async def close(self):
        pass


def test_oversized_frames_are_refused_at_the_header():
    async def go():
        huge = _Stream(struct.pack(">I", p2p.MAX_FRAME_BYTES + 1) + b"x" * 10)
        with pytest.raises(ValueError, match="cap"):
            await p2p._read_frame_no_close(huge)
        assert huge.reads == 1          # only the header was read
        ok = _Stream(struct.pack(">I", 5) + b"hello")
        assert await p2p._read_framed(ok) == b"hello"
        big = struct.pack(">I", p2p.MAX_FRAME_BYTES + 1)
        s = _Stream(big)
        with pytest.raises(ValueError):
            await p2p._read_framed(s)
        # A blob reply may be larger than an ordinary message.
        s = _Stream(struct.pack(">I", 3) + b"abc")
        assert await p2p._read_frame_no_close(s, p2p.MAX_BLOB_FRAME_BYTES) == b"abc"
    trio.run(go)
