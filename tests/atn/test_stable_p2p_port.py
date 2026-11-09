"""The daemon keeps one p2p port across restarts, so its address can be a
Registry seed."""
from atn.autonet_service import _stable_p2p_port


def test_first_start_picks_and_saves_a_port(tmp_path):
    port = _stable_p2p_port(tmp_path, "127.0.0.1")
    assert 0 < port < 65536
    assert (tmp_path / "p2p.port").read_text() == str(port)
    assert _stable_p2p_port(tmp_path, "127.0.0.1") == port


def test_a_saved_port_wins(tmp_path):
    (tmp_path / "p2p.port").write_text("4101")
    assert _stable_p2p_port(tmp_path) == 4101


def test_an_unreadable_file_is_replaced(tmp_path):
    (tmp_path / "p2p.port").write_text("junk")
    port = _stable_p2p_port(tmp_path, "127.0.0.1")
    assert port > 0 and (tmp_path / "p2p.port").read_text() == str(port)


def test_the_tcp_listener_keeps_its_own_port(tmp_path):
    ws = _stable_p2p_port(tmp_path, "127.0.0.1")
    tcp = _stable_p2p_port(tmp_path, "127.0.0.1", "p2p.tcp_port", avoid=ws)
    assert tcp != ws
    assert (tmp_path / "p2p.tcp_port").read_text() == str(tcp)
    assert _stable_p2p_port(tmp_path, "127.0.0.1", "p2p.tcp_port", avoid=ws) == tcp
