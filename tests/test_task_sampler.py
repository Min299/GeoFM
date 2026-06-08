"""tests/test_task_sampler.py

Tests for task sampler.
"""
import pytest


class TestTaskSampler:
    """Test TaskSampler class."""

    def test_init(self):
        """Sampler should initialize with tasks."""
        from geofm.tasks.task_sampler import TaskSampler

        sampler = TaskSampler(["flood", "burn"])

        assert sampler.tasks == ["flood", "burn"]

    def test_sample(self):
        """sample should return a task."""
        from geofm.tasks.task_sampler import TaskSampler

        sampler = TaskSampler(["flood", "burn"])

        task = sampler.sample()

        assert task in ["flood", "burn"]

    def test_sample_with_weight(self):
        """sample_with_weight should respect weights."""
        from geofm.tasks.task_sampler import TaskSampler

        sampler = TaskSampler(["flood"], weights=[1.0])

        # With single task, should always return it
        for _ in range(10):
            assert sampler.sample_with_weight() == "flood"

    def test_sample_batch(self):
        """sample_batch should return multiple tasks."""
        from geofm.tasks.task_sampler import TaskSampler

        sampler = TaskSampler(["flood", "burn"])

        batch = sampler.sample_batch(5)

        assert len(batch) == 5
        assert all(t in ["flood", "burn"] for t in batch)

    def test_set_weights(self):
        """set_weights should update weights."""
        from geofm.tasks.task_sampler import TaskSampler

        sampler = TaskSampler(["flood", "burn"])

        sampler.set_weights([2.0, 1.0])

        assert sampler.weights == [2.0, 1.0]


class TestPrioritySampler:
    """Test PrioritySampler class."""

    def test_priorities_respected(self):
        """Higher priority tasks should be sampled more."""
        from geofm.tasks.task_sampler import PrioritySampler

        sampler = PrioritySampler(
            ["flood", "burn"],
            priorities=[1.0, 0.0],
        )

        # With 0 priority on burn, should always get flood
        for _ in range(10):
            assert sampler.sample() == "flood"


class TestAdaptiveSampler:
    """Test AdaptiveSampler class."""

    def test_update_performance(self):
        """update_performance should record history."""
        from geofm.tasks.task_sampler import AdaptiveSampler

        sampler = AdaptiveSampler(["flood", "burn"])

        sampler.update_performance("flood", 0.5)
        sampler.update_performance("flood", 0.6)

        assert len(sampler.performance_history["flood"]) == 2


class TestCurriculumSampler:
    """Test CurriculumSampler class."""

    def test_advance_level(self):
        """advance_level should increase difficulty."""
        from geofm.tasks.task_sampler import CurriculumSampler

        sampler = CurriculumSampler(
            ["easy", "hard"],
            difficulty_levels=[0, 1],
        )

        assert sampler.current_level == 0

        sampler.advance_level()

        assert sampler.current_level == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])