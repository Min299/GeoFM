"""tests/test_leaderboard.py

Tests for leaderboard.
"""
import pytest


class TestLeaderboard:
    """Test leaderboard."""

    def test_init(self):
        """Leaderboard should initialize."""
        from geofm.evaluation.leaderboard import Leaderboard

        board = Leaderboard()

        assert len(board.results) == 0
        assert board.higher_is_better is True

    def test_add_result(self):
        """add_result should add experiment."""
        from geofm.evaluation.leaderboard import Leaderboard

        board = Leaderboard()
        board.add_result("feature", 0.85)

        assert board.get_result("feature") == 0.85

    def test_add_multiple_results(self):
        """add_result should handle multiple experiments."""
        from geofm.evaluation.leaderboard import Leaderboard

        board = Leaderboard()
        board.add_result("feature", 0.85)
        board.add_result("lora", 0.82)
        board.add_result("hybrid", 0.88)

        assert len(board.results) == 3

    def test_ranking(self):
        """ranking should sort by score."""
        from geofm.evaluation.leaderboard import Leaderboard

        board = Leaderboard()
        board.add_result("feature", 0.85)
        board.add_result("lora", 0.82)
        board.add_result("hybrid", 0.88)

        ranking = board.ranking()

        assert ranking[0] == ("hybrid", 0.88)
        assert ranking[1] == ("feature", 0.85)
        assert ranking[2] == ("lora", 0.82)

    def test_ranking_lower_is_better(self):
        """ranking with lower_is_better should sort ascending."""
        from geofm.evaluation.leaderboard import Leaderboard

        board = Leaderboard(higher_is_better=False)
        board.add_result("feature", 0.85)
        board.add_result("lora", 0.82)
        board.add_result("hybrid", 0.88)

        ranking = board.ranking()

        assert ranking[0] == ("lora", 0.82)
        assert ranking[1] == ("feature", 0.85)
        assert ranking[2] == ("hybrid", 0.88)

    def test_top_k(self):
        """top_k should return top K results."""
        from geofm.evaluation.leaderboard import Leaderboard

        board = Leaderboard()
        board.add_result("a", 0.9)
        board.add_result("b", 0.7)
        board.add_result("c", 0.8)
        board.add_result("d", 0.6)

        top2 = board.top_k(2)

        assert len(top2) == 2
        assert top2[0] == ("a", 0.9)
        assert top2[1] == ("c", 0.8)

    def test_get_rank(self):
        """get_rank should return correct rank."""
        from geofm.evaluation.leaderboard import Leaderboard

        board = Leaderboard()
        board.add_result("feature", 0.85)
        board.add_result("lora", 0.82)
        board.add_result("hybrid", 0.88)

        assert board.get_rank("hybrid") == 1
        assert board.get_rank("feature") == 2
        assert board.get_rank("lora") == 3

    def test_get_rank_unknown(self):
        """get_rank should return None for unknown."""
        from geofm.evaluation.leaderboard import Leaderboard

        board = Leaderboard()
        board.add_result("feature", 0.85)

        assert board.get_rank("unknown") is None

    def test_clear(self):
        """clear should remove all results."""
        from geofm.evaluation.leaderboard import Leaderboard

        board = Leaderboard()
        board.add_result("feature", 0.85)
        board.add_result("lora", 0.82)

        board.clear()

        assert len(board.results) == 0

    def test_summary(self):
        """summary should return formatted string."""
        from geofm.evaluation.leaderboard import Leaderboard

        board = Leaderboard()
        board.add_result("feature", 0.85)
        board.add_result("lora", 0.82)

        summary = board.summary()

        assert "Leaderboard" in summary
        assert "feature" in summary
        assert "lora" in summary

    def test_to_dict(self):
        """to_dict should return dictionary."""
        from geofm.evaluation.leaderboard import Leaderboard

        board = Leaderboard()
        board.add_result("feature", 0.85)

        d = board.to_dict()

        assert d["higher_is_better"] is True
        assert "feature" in d["results"]
        assert d["results"]["feature"] == 0.85


class TestMultiMetricLeaderboard:
    """Test multi-metric leaderboard."""

    def test_init(self):
        """Multi-metric leaderboard should initialize."""
        from geofm.evaluation.leaderboard import MultiMetricLeaderboard

        board = MultiMetricLeaderboard(metrics=["dice", "iou"])

        assert "dice" in board.leaderboards
        assert "iou" in board.leaderboards

    def test_add_result(self):
        """add_result should add for all metrics."""
        from geofm.evaluation.leaderboard import MultiMetricLeaderboard

        board = MultiMetricLeaderboard(metrics=["dice", "iou"])

        scores = {"dice": 0.85, "iou": 0.75}
        board.add_result("feature", scores)

        assert board.leaderboards["dice"].get_result("feature") == 0.85
        assert board.leaderboards["iou"].get_result("feature") == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])