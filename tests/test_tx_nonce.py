"""next_nonce: concurrent sends from one key never share a nonce."""
import threading

from nodes.common import tx_nonce


class _Eth:
    def __init__(self, count):
        self.count = count

    def get_transaction_count(self, address, block="latest"):
        assert block == "pending"
        return self.count


class _W3:
    def __init__(self, count):
        self.eth = _Eth(count)


A = "0x00000000000000000000000000000000000000aa"


def test_lagging_rpc_never_repeats_a_nonce():
    w3 = _W3(48)  # the RPC never sees the sends
    got = []
    threads = [threading.Thread(target=lambda: got.append(tx_nonce.next_nonce(w3, A)))
               for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(got) == [48, 49, 50, 51, 52]


def test_chain_ahead_wins():
    w3 = _W3(5)
    assert tx_nonce.next_nonce(w3, A) == 5
    w3.eth.count = 9
    assert tx_nonce.next_nonce(w3, A) == 9


def test_failed_send_does_not_leave_a_gap_forever(monkeypatch):
    w3 = _W3(3)
    assert tx_nonce.next_nonce(w3, A) == 3   # this send fails
    monkeypatch.setattr(tx_nonce, "STALE_SECS", 0.0)
    assert tx_nonce.next_nonce(w3, A) == 3   # stale: the chain's count again


def test_reset_chain_is_followed():
    w3 = _W3(40)
    assert tx_nonce.next_nonce(w3, A) == 40
    w3.eth.count = 0                          # a fresh chain
    assert tx_nonce.next_nonce(w3, A) == 0
