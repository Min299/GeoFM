"""tests/test_task_scheduler.py

Tests for task scheduler.
"""
import pytest


class TestTaskScheduler:
    """Test TaskScheduler class."""

    def test_init(self):
        """Scheduler should initialize with tasks."""
        from geofm.tasks.task_scheduler import TaskScheduler

        scheduler = TaskScheduler(["flood", "burn", "lulc"])

        assert scheduler.tasks == ["flood", "burn", "lulc"]
        assert scheduler.index == 0

    def test_next_task_round_robin(self):
        """next_task should cycle through tasks."""
        from geofm.tasks.task_scheduler import TaskScheduler

        scheduler = TaskScheduler(["flood", "burn"])

        assert scheduler.next_task() == "flood"
        assert scheduler.next_task() == "burn"
        assert scheduler.next_task() == "flood"
        assert scheduler.next_task() == "burn"

    def test_reset(self):
        """reset should reset index to 0."""
        from geofm.tasks.task_scheduler import TaskScheduler

        scheduler = TaskScheduler(["flood", "burn"])
        scheduler.next_task()
        scheduler.next_task()

        scheduler.reset()

        assert scheduler.index == 0
        assert scheduler.next_task() == "flood"

    def test_current_task(self):
        """current_task should return current task."""
        from geofm.tasks.task_scheduler import TaskScheduler

        scheduler = TaskScheduler(["flood", "burn"])

        assert scheduler.current_task() == "flood"
        scheduler.next_task()
        assert scheduler.current_task() == "burn"

    def test_task_count(self):
        """task_count should return number of tasks."""
        from geofm.tasks.task_scheduler import TaskScheduler

        scheduler = TaskScheduler(["flood", "burn", "lulc"])

        assert scheduler.task_count() == 3

    def test_sample_task(self):
        """sample_task should sample based on weights."""
        from geofm.tasks.task_scheduler import TaskScheduler

        scheduler = TaskScheduler(["flood", "burn"], weights=[1.0, 0.0])

        # With 0 weight on burn, should always get flood
        for _ in range(10):
            assert scheduler.sample_task() == "flood"

    def test_get_task_weights(self):
        """get_task_weights should return weights dict."""
        from geofm.tasks.task_scheduler import TaskScheduler

        scheduler = TaskScheduler(["flood", "burn"], weights=[1.0, 2.0])

        weights = scheduler.get_task_weights()

        assert weights["flood"] == 1.0
        assert weights["burn"] == 2.0


class TestMultiTaskScheduler:
    """Test MultiTaskScheduler class."""

    def test_init_with_configs(self):
        """MultiTaskScheduler should initialize with configs."""
        from geofm.tasks.task_scheduler import MultiTaskScheduler, TaskConfig

        configs = [
            TaskConfig("flood", weight=1.0),
            TaskConfig("burn", weight=1.0),
            TaskConfig("lulc", weight=0.5),
        ]

        scheduler = MultiTaskScheduler(configs)

        assert scheduler.task_count() == 3

    def test_disabled_tasks_excluded(self):
        """Disabled tasks should be excluded."""
        from geofm.tasks.task_scheduler import MultiTaskScheduler, TaskConfig

        configs = [
            TaskConfig("flood", weight=1.0, enabled=True),
            TaskConfig("burn", weight=1.0, enabled=False),
        ]

        scheduler = MultiTaskScheduler(configs)

        # Only flood should be available
        assert scheduler.task_count() == 1
        assert scheduler.next_task() == "flood"


class TestBalanceScheduler:
    """Test BalanceScheduler class."""

    def test_balances_tasks(self):
        """BalanceScheduler should balance task samples."""
        from geofm.tasks.task_scheduler import BalanceScheduler

        scheduler = BalanceScheduler(["flood", "burn"], weights=[1.0, 1.0])

        # Sample multiple times
        for _ in range(10):
            scheduler.next_task()

        # Both tasks should have been sampled
        assert all(c > 0 for c in scheduler.sample_counts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])