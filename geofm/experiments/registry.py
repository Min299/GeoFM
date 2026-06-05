"""geofm.experiments.registry

Experiment registry for tracking all experiments.
"""
from typing import Dict, List, Optional
from pathlib import Path
import json

from geofm.experiments.experiment import ExperimentConfig, ExperimentResult


class ExperimentRegistry:
    """Registry for tracking experiments.

    Maintains a record of all experiments and their results.
    """

    def __init__(self, registry_path: str = "outputs/experiments_registry.json"):
        self.registry_path = Path(registry_path)
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, ExperimentResult] = {}
        self._load()

    def register(self, config: ExperimentConfig) -> None:
        """Register an experiment.

        Args:
            config: Experiment configuration
        """
        self.experiments[config.name] = config
        self._save()

    def add_result(self, result: ExperimentResult) -> None:
        """Add results for an experiment.

        Args:
            result: Experiment results
        """
        self.results[result.config.name] = result
        self._save()

    def get(self, name: str) -> Optional[ExperimentConfig]:
        """Get experiment config by name."""
        return self.experiments.get(name)

    def get_result(self, name: str) -> Optional[ExperimentResult]:
        """Get results by experiment name."""
        return self.results.get(name)

    def list_experiments(self) -> List[str]:
        """List all registered experiment names."""
        return list(self.experiments.keys())

    def list_by_task(self, task: str) -> List[str]:
        """List experiments for a specific task."""
        return [
            name for name, exp in self.experiments.items()
            if exp.task == task
        ]

    def list_by_mode(self, mode: str) -> List[str]:
        """List experiments by training mode."""
        return [
            name for name, exp in self.experiments.items()
            if exp.training_mode == mode
        ]

    def compare(
        self,
        task: str,
        modes: List[str] = None
    ) -> Dict[str, Dict]:
        """Compare experiments for a task.

        Args:
            task: Task to compare
            modes: Training modes to compare (default: ["full_ft", "lora"])

        Returns:
            Dict of mode -> metrics
        """
        if modes is None:
            modes = ["full_ft", "lora"]

        comparison = {}
        for mode in modes:
            exp_name = f"{task}_{mode}"
            result = self.get_result(exp_name)
            if result:
                comparison[mode] = result.metrics

        return comparison

    def _save(self) -> None:
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "experiments": {
                name: exp.to_dict()
                for name, exp in self.experiments.items()
            },
            "results": {
                name: res.to_dict()
                for name, res in self.results.items()
            }
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                data = json.load(f)

            self.experiments = {
                name: ExperimentConfig(**exp)
                for name, exp in data.get("experiments", {}).items()
            }

            # Results need special handling
            self.results = {}
            for name, res_data in data.get("results", {}).items():
                config = ExperimentConfig(**res_data["config"])
                self.results[name] = ExperimentResult(
                    config=config,
                    metrics=res_data["metrics"],
                    best_epoch=res_data["best_epoch"],
                    total_epochs=res_data["total_epochs"],
                    duration_seconds=res_data["duration_seconds"],
                    checkpoint_path=res_data.get("checkpoint_path"),
                )


# Global registry instance
_global_registry: Optional[ExperimentRegistry] = None


def get_registry() -> ExperimentRegistry:
    """Get the global experiment registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ExperimentRegistry()
    return _global_registry