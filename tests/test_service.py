"""Tests for Epic 6 — Service, resource monitor, and updater."""

import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from nodes.common.config import AutonetConfig, ResourceConfig, UpdateConfig, NodeConfig, PrivacyConfig
from nodes.common.resource_monitor import ResourceMonitor, ResourceSnapshot
from nodes.common.updater import AutonetUpdater, UpdateInfo, UpdateSource
from nodes.service import AutonetService, ServiceState


class TestResourceSnapshot(unittest.TestCase):
    """Tests for ResourceSnapshot."""

    def test_default_values(self):
        snap = ResourceSnapshot()
        self.assertEqual(snap.cpu_percent, 0.0)
        self.assertEqual(snap.memory_mb, 0.0)
        self.assertEqual(snap.gpu_percent, 0.0)


class TestResourceMonitor(unittest.TestCase):
    """Tests for ResourceMonitor."""

    def _make_config(self, max_cpu=80.0, max_mem=4096, start=0, end=24, interval=0.0):
        cfg = AutonetConfig()
        cfg.resources = ResourceConfig(
            max_cpu_percent=max_cpu,
            max_memory_mb=max_mem,
            active_hours_start=start,
            active_hours_end=end,
            check_interval=interval,
        )
        return cfg

    def test_snapshot(self):
        """Snapshot returns a ResourceSnapshot."""
        cfg = self._make_config()
        mon = ResourceMonitor(config=cfg)
        snap = mon.snapshot()
        self.assertIsInstance(snap, ResourceSnapshot)
        self.assertGreater(snap.timestamp, 0)

    def test_should_train_default(self):
        """Training allowed with default limits."""
        cfg = self._make_config(max_cpu=100.0, max_mem=99999, interval=0.0)
        mon = ResourceMonitor(config=cfg)
        # With 100% CPU limit and 99GB memory limit, should always allow
        self.assertTrue(mon.should_train())

    @patch("nodes.common.resource_monitor.ResourceMonitor.snapshot")
    def test_cpu_limit_exceeded(self, mock_snap):
        """Training paused when CPU exceeds limit."""
        mock_snap.return_value = ResourceSnapshot(cpu_percent=95.0, memory_mb=1000)
        cfg = self._make_config(max_cpu=80.0, interval=0.0)
        mon = ResourceMonitor(config=cfg)
        self.assertFalse(mon.should_train())
        self.assertIn("CPU", mon.paused_reason)

    @patch("nodes.common.resource_monitor.ResourceMonitor.snapshot")
    def test_memory_limit_exceeded(self, mock_snap):
        """Training paused when memory exceeds limit."""
        mock_snap.return_value = ResourceSnapshot(cpu_percent=50.0, memory_mb=5000)
        cfg = self._make_config(max_mem=4096, interval=0.0)
        mon = ResourceMonitor(config=cfg)
        self.assertFalse(mon.should_train())
        self.assertIn("Memory", mon.paused_reason)

    @patch("nodes.common.resource_monitor.ResourceMonitor.snapshot")
    def test_within_limits(self, mock_snap):
        """Training proceeds when within limits."""
        mock_snap.return_value = ResourceSnapshot(cpu_percent=50.0, memory_mb=2000)
        cfg = self._make_config(max_cpu=80.0, max_mem=4096, interval=0.0)
        mon = ResourceMonitor(config=cfg)
        self.assertTrue(mon.should_train())
        self.assertIsNone(mon.paused_reason)

    def test_active_hours_always(self):
        """0-24 means always active."""
        cfg = self._make_config(start=0, end=24)
        mon = ResourceMonitor(config=cfg)
        self.assertTrue(mon._within_active_hours())

    def test_active_hours_normal_range(self):
        """Normal range (e.g. 9-17)."""
        cfg = self._make_config(start=9, end=17)
        mon = ResourceMonitor(config=cfg)
        current_hour = datetime.now().hour
        result = mon._within_active_hours()
        expected = 9 <= current_hour < 17
        self.assertEqual(result, expected)


class TestUpdateInfo(unittest.TestCase):
    """Tests for UpdateInfo."""

    def test_has_update_true(self):
        info = UpdateInfo(current_version="0.1.0", available_version="0.2.0", source="git")
        self.assertTrue(info.has_update)

    def test_has_update_false(self):
        info = UpdateInfo(current_version="0.1.0", available_version="0.1.0", source="git")
        self.assertFalse(info.has_update)


class TestAutonetUpdater(unittest.TestCase):
    """Tests for AutonetUpdater."""

    def _make_config(self, source="git", interval=3600):
        cfg = AutonetConfig()
        cfg.update = UpdateConfig(update_source=source, check_interval=interval)
        return cfg

    def test_should_check_initially(self):
        """Should check on first call."""
        cfg = self._make_config()
        u = AutonetUpdater(config=cfg)
        self.assertTrue(u.should_check())

    def test_should_check_respects_interval(self):
        """Should not check again within interval."""
        cfg = self._make_config(interval=3600)
        u = AutonetUpdater(config=cfg)
        u._last_check = time.time()  # Just checked
        self.assertFalse(u.should_check())

    def test_unknown_source(self):
        """Unknown source returns None."""
        cfg = self._make_config(source="floppy_disk")
        u = AutonetUpdater(config=cfg)
        result = u.check_update()
        self.assertIsNone(result)

    def test_apply_no_update(self):
        """Apply returns True when no update needed."""
        cfg = self._make_config()
        u = AutonetUpdater(config=cfg)
        info = UpdateInfo(current_version="0.1.0", available_version="0.1.0", source="git")
        self.assertTrue(u.apply_update(info))

    @patch("subprocess.run")
    def test_git_check(self, mock_run):
        """Git check fetches and compares revisions."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # git fetch
            MagicMock(returncode=0, stdout="abc12345\n", stderr=""),  # local HEAD
            MagicMock(returncode=0, stdout="abc12345\n", stderr=""),  # remote HEAD
        ]
        cfg = self._make_config(source="git")
        u = AutonetUpdater(config=cfg)
        info = u.check_update()
        self.assertIsNotNone(info)
        self.assertFalse(info.has_update)

    @patch("subprocess.run")
    def test_git_check_update_available(self, mock_run):
        """Git check detects when remote is ahead."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git fetch
            MagicMock(returncode=0, stdout="abc12345\n"),  # local HEAD
            MagicMock(returncode=0, stdout="def67890\n"),  # remote HEAD (different)
            MagicMock(returncode=0, stdout="New feature\n"),  # changelog
        ]
        cfg = self._make_config(source="git")
        u = AutonetUpdater(config=cfg)
        info = u.check_update()
        self.assertIsNotNone(info)
        self.assertTrue(info.has_update)
        self.assertIn("New feature", info.changelog)


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
