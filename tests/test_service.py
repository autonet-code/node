"""Tests for Epic 6 — Service and updater."""

import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from nodes.common.config import AutonetConfig, ResourceConfig, UpdateConfig, NodeConfig, PrivacyConfig
from nodes.common.updater import AutonetUpdater, UpdateInfo, UpdateSource
from nodes.service import AutonetService, ServiceState


class TestUpdateInfo(unittest.TestCase):
    """Tests for UpdateInfo."""

    def test_has_update_true(self):
        info = UpdateInfo(current_version="0.1.0", available_version="0.2.0", source="pypi")
        self.assertTrue(info.has_update)

    def test_has_update_false(self):
        info = UpdateInfo(current_version="0.1.0", available_version="0.1.0", source="pypi")
        self.assertFalse(info.has_update)


class TestAutonetUpdater(unittest.TestCase):
    """Tests for AutonetUpdater. Each test gets its own data dir: the updater
    persists last_check.json there, and the default is the real ~/.atn."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp(prefix="updater_test_")

    def _make_config(self, source="pypi", interval=3600):
        cfg = AutonetConfig()
        cfg.update = UpdateConfig(update_source=source, check_interval=interval)
        return cfg

    def test_should_check_initially(self):
        """Should check on first call."""
        cfg = self._make_config()
        u = AutonetUpdater(config=cfg, data_dir=self._tmp)
        self.assertTrue(u.should_check())

    def test_should_check_respects_interval(self):
        """Should not check again within interval."""
        cfg = self._make_config(interval=3600)
        u = AutonetUpdater(config=cfg, data_dir=self._tmp)
        u._last_check = time.time()  # Just checked
        self.assertFalse(u.should_check())

    def test_unknown_source(self):
        """Unknown source returns None."""
        cfg = self._make_config(source="floppy_disk")
        u = AutonetUpdater(config=cfg, data_dir=self._tmp)
        result = u.check_update()
        self.assertIsNone(result)

    def test_apply_no_update(self):
        """Apply returns True when no update needed."""
        cfg = self._make_config()
        u = AutonetUpdater(config=cfg, data_dir=self._tmp)
        info = UpdateInfo(current_version="0.1.0", available_version="0.1.0", source="pypi")
        self.assertTrue(u.apply_update(info))


class TestAutonetService(unittest.TestCase):
    """Tests for AutonetService."""

    def _make_config(self, max_cycles=2, cycle_delay=0.001):
        cfg = AutonetConfig()
        cfg.node = NodeConfig(max_cycles=max_cycles, cycle_delay=cycle_delay)
        cfg.update = UpdateConfig(check_interval=999999)
        cfg.resources = ResourceConfig(check_interval=999999)  # Skip resource checks in tests
        return cfg

    def test_initial_state(self):
        cfg = self._make_config()
        svc = AutonetService(config=cfg)
        self.assertEqual(svc.state, ServiceState.STOPPED)

    def test_status(self):
        cfg = self._make_config()
        svc = AutonetService(config=cfg)
        status = svc.status()
        self.assertEqual(status.state, ServiceState.STOPPED)
        self.assertEqual(status.cycles_completed, 0)

    def test_start_runs_cycles(self):
        """Service runs max_cycles then stops."""
        cfg = self._make_config(max_cycles=3, cycle_delay=0.001)
        svc = AutonetService(config=cfg)
        svc.start()
        self.assertEqual(svc.state, ServiceState.STOPPED)
        self.assertEqual(svc._cycles, 3)

    def test_stop_idempotent(self):
        """Stopping an already stopped service is safe."""
        cfg = self._make_config()
        svc = AutonetService(config=cfg)
        svc.stop()
        svc.stop()
        self.assertEqual(svc.state, ServiceState.STOPPED)

    def test_signal_shutdown(self):
        """Setting _shutdown_requested causes main loop to exit."""
        cfg = self._make_config(max_cycles=100, cycle_delay=0.001)
        svc = AutonetService(config=cfg)

        # Patch to trigger shutdown after 2 cycles
        original_run = svc._run_cycle
        call_count = [0]
        def _patched_cycle():
            call_count[0] += 1
            if call_count[0] >= 2:
                svc._shutdown_requested = True
            return original_run()

        svc._run_cycle = _patched_cycle
        svc.start()
        self.assertEqual(svc.state, ServiceState.STOPPED)
        self.assertGreaterEqual(svc._cycles, 2)

    def test_version(self):
        """Version returns a string."""
        version = AutonetService._get_version()
        self.assertIsInstance(version, str)
        self.assertRegex(version, r"\d+\.\d+\.\d+")


if __name__ == "__main__":
    unittest.main()
