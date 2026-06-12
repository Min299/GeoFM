"""geofm.evaluation.leaderboard

Leaderboard for ranking experiments.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class LeaderboardEntry:
    """Single entry in the leaderboard."""

    experiment_name: str
    score: float
    rank: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Entry as dictionary
        """
        return {
            "experiment": self.experiment_name,
            "score": self.score,
            "rank": self.rank,
            **self.metadata,
        }


class Leaderboard:
    """Leaderboard for ranking experiments.

    Usage:
        board = Leaderboard()
        board.add_result("feature", 0.85)
        board.add_result("lora", 0.82)
        ranking = board.ranking()  # [("feature", 0.85), ("lora", 0.82)]
    """

    def __init__(self, higher_is_better: bool = True):
        """Initialize leaderboard.

        Args:
            higher_is_better: If True, higher scores rank higher
        """
        self.results: Dict[str, float] = {}
        self.metadata: Dict[str, Dict] = {}
        self.higher_is_better = higher_is_better
        self._ranking: Optional[List[Tuple[str, float]]] = None

    def add_result(
        self,
        name: str,
        score: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add a result to the leaderboard.

        Args:
            name: Experiment name
            score: Score value
            metadata: Optional additional metadata
        """
        self.results[name] = score
        if metadata:
            self.metadata[name] = metadata
        self._ranking = None  # Invalidate cache

    def update_result(
        self,
        name: str,
        score: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Update an existing result.

        Args:
            name: Experiment name
            score: New score value
            metadata: Optional new metadata
        """
        if name not in self.results:
            raise KeyError(f"Experiment {name} not in leaderboard")

        self.results[name] = score
        if metadata:
            self.metadata[name] = metadata
        self._ranking = None  # Invalidate cache

    def get_result(self, name: str) -> Optional[float]:
        """Get score for an experiment.

        Args:
            name: Experiment name

        Returns:
            Score or None if not found
        """
        return self.results.get(name)

    def ranking(self) -> List[Tuple[str, float]]:
        """Get ranking of experiments.

        Returns:
            List of (name, score) tuples sorted by score
        """
        if self._ranking is None:
            reverse = self.higher_is_better
            self._ranking = sorted(
                self.results.items(),
                key=lambda x: x[1],
                reverse=reverse,
            )

        return self._ranking

    def get_ranked_entries(self) -> List[LeaderboardEntry]:
        """Get leaderboard as list of entries.

        Returns:
            List of LeaderboardEntry objects
        """
        ranking = self.ranking()

        entries = []
        for rank, (name, score) in enumerate(ranking, 1):
            entry = LeaderboardEntry(
                experiment_name=name,
                score=score,
                rank=rank,
                metadata=self.metadata.get(name, {}),
            )
            entries.append(entry)

        return entries

    def top_k(self, k: int = 5) -> List[Tuple[str, float]]:
        """Get top K experiments.

        Args:
            k: Number of top experiments to return

        Returns:
            List of top K (name, score) tuples
        """
        return self.ranking()[:k]

    def get_rank(self, name: str) -> Optional[int]:
        """Get rank of an experiment.

        Args:
            name: Experiment name

        Returns:
            Rank (1-indexed) or None if not found
        """
        ranking = self.ranking()

        for rank, (exp_name, _) in enumerate(ranking, 1):
            if exp_name == name:
                return rank

        return None

    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()
        self.metadata.clear()
        self._ranking = None

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Leaderboard as dictionary
        """
        return {
            "higher_is_better": self.higher_is_better,
            "results": self.results,
            "metadata": self.metadata,
            "ranking": [
                {"rank": i + 1, "experiment": name, "score": score}
                for i, (name, score) in enumerate(self.ranking())
            ],
        }

    def summary(self) -> str:
        """Get summary string.

        Returns:
            Formatted summary
        """
        lines = []
        lines.append("Leaderboard")
        lines.append("-" * 40)

        for rank, (name, score) in enumerate(self.ranking(), 1):
            lines.append(f"  {rank}. {name}: {score:.4f}")

        return "\n".join(lines)


class MultiMetricLeaderboard:
    """Leaderboard that tracks multiple metrics."""

    def __init__(self, metrics: List[str], higher_is_better: bool = True):
        """Initialize multi-metric leaderboard.

        Args:
            metrics: List of metric names to track
            higher_is_better: If True, higher scores rank higher
        """
        self.metrics = metrics
        self.leaderboards = {
            metric: Leaderboard(higher_is_better)
            for metric in metrics
        }

    def add_result(
        self,
        name: str,
        scores: Dict[str, float],
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add results for all metrics.

        Args:
            name: Experiment name
            scores: Dictionary of metric scores
            metadata: Optional additional metadata
        """
        for metric in self.metrics:
            if metric in scores:
                self.leaderboards[metric].add_result(
                    name,
                    scores[metric],
                    metadata=metadata,
                )

    def ranking(self, metric: str) -> List[Tuple[str, float]]:
        """Get ranking for a specific metric.

        Args:
            metric: Metric name

        Returns:
            List of (name, score) tuples
        """
        if metric not in self.leaderboards:
            raise KeyError(f"Unknown metric: {metric}")
        return self.leaderboards[metric].ranking()

    def summary(self, metric: str) -> str:
        """Get summary for a metric.

        Args:
            metric: Metric name

        Returns:
            Formatted summary
        """
        if metric not in self.leaderboards:
            raise KeyError(f"Unknown metric: {metric}")
        return self.leaderboards[metric].summary()