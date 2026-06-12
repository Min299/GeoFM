"""tests/test_real_flood_pipeline.py

Tests for real flood segmentation pipeline.
"""
import pytest
import torch


class TestRealFloodPipeline:
    """Test flood segmentation forward pipeline."""

    def test_flood_decoder_init(self):
        """FloodDecoder should initialize."""
        try:
            from geofm.models.decoders.flood_decoder import FloodDecoder

            decoder = FloodDecoder()

            assert decoder is not None
            assert hasattr(decoder, 'pyramid')
            assert hasattr(decoder, 'head')
        except ImportError:
            pytest.skip("FloodDecoder not available")

    def test_flood_decoder_forward(self):
        """FloodDecoder should produce correct output shape."""
        try:
            from geofm.models.decoders.flood_decoder import FloodDecoder

            decoder = FloodDecoder()

            features = [
                torch.randn(2, 768, 64, 64),
                torch.randn(2, 768, 32, 32),
                torch.randn(2, 768, 16, 16),
                torch.randn(2, 768, 8, 8),
            ]

            out = decoder(features)

            # Output should be (B, num_classes, H_out, W_out)
            assert out.shape[0] == 2  # batch size
            assert out.shape[1] == 2  # num_classes (binary)
            assert out.dim() == 4  # 4D tensor

        except ImportError:
            pytest.skip("FloodDecoder not available")

    def test_flood_decoder_output_size(self):
        """FloodDecoder output should match input resolution."""
        try:
            from geofm.models.decoders.flood_decoder import FloodDecoder

            decoder = FloodDecoder()

            features = [
                torch.randn(2, 768, 64, 64),
                torch.randn(2, 768, 32, 32),
                torch.randn(2, 768, 16, 16),
                torch.randn(2, 768, 8, 8),
            ]

            out = decoder(features)

            # Output should be at 64x64 resolution (from first feature)
            assert out.shape[2] == 64
            assert out.shape[3] == 64

        except ImportError:
            pytest.skip("FloodDecoder not available")

    def test_pyramid_decoder(self):
        """PyramidDecoder should work."""
        try:
            from geofm.models.decoders.pyramid_decoder import PyramidDecoder

            decoder = PyramidDecoder(
                in_channels=(768, 768, 768, 768),
                decoder_channels=256,
            )

            features = [
                torch.randn(2, 768, 64, 64),
                torch.randn(2, 768, 32, 32),
                torch.randn(2, 768, 16, 16),
                torch.randn(2, 768, 8, 8),
            ]

            out = decoder(features)

            # Output should be (B, decoder_channels, H, W)
            assert out.shape[1] == 256
            assert out.dim() == 4

        except ImportError:
            pytest.skip("PyramidDecoder not available")

    def test_segmentation_head(self):
        """SegmentationHead should work."""
        try:
            from geofm.models.heads.segmentation_head import SegmentationHead

            head = SegmentationHead(
                in_channels=256,
                num_classes=2,
            )

            x = torch.randn(2, 256, 64, 64)

            out = head(x)

            assert out.shape == (2, 2, 64, 64)

        except ImportError:
            pytest.skip("SegmentationHead not available")

    def test_full_pipeline_shapes(self):
        """Full pipeline should produce correct shapes."""
        try:
            from geofm.models.decoders.flood_decoder import FloodDecoder
            from geofm.debug.feature_inspector import FeatureInspector

            decoder = FloodDecoder()

            features = [
                torch.randn(2, 768, 64, 64),
                torch.randn(2, 768, 32, 32),
                torch.randn(2, 768, 16, 16),
                torch.randn(2, 768, 8, 8),
            ]

            # Validate features
            FeatureInspector.inspect_feature_list(features)

            # Forward
            out = decoder(features)

            assert out.shape == (2, 2, 64, 64)

        except ImportError:
            pytest.skip("Pipeline not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])