"""geofm.builders.model_builder

Central constructor for GeoFM models from config.
"""
from __future__ import annotations

from geofm.models.backbones.backbone_factory import (
    build_backbone,
)

from geofm.models.backbones.terramind_wrapper import (
    TerraMindWrapper,
)

from geofm.models.tasks.segmentation_model import (
    SegmentationModel,
)

from geofm.models.tasks.classification_model import (
    ClassificationModel,
)


class ModelBuilder:
    """Central constructor for GeoFM models.

    Builds complete model from config:
        - Backbone (TerraMind)
        - Task head (segmentation/classification)

    Usage:
        model = ModelBuilder.build(cfg)
    """

    @staticmethod
    def build(
        cfg,
    ):
        """Build model from config.

        Args:
            cfg: Configuration object with model, task, decoder/classifier specs

        Returns:
            Assembled model ready for training
        """
        # Build backbone
        backbone = build_backbone(
            cfg.model.backbone,
            pretrained=cfg.model.pretrained,
        )

        # Wrap with TerraMind wrapper for feature extraction
        backbone = TerraMindWrapper(
            backbone=backbone,
        )

        task = cfg.task.name

        if task == "segmentation":

            decoder = (
                cfg.decoder.instance
            )

            return (
                SegmentationModel(
                    backbone,
                    decoder,
                )
            )

        if task == "classification":

            classifier = (
                cfg.classifier.instance
            )

            return (
                ClassificationModel(
                    backbone,
                    classifier,
                )
            )

        raise ValueError(
            f"Unknown task: {task}"
        )