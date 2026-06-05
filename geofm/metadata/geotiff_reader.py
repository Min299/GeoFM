"""geofm.metadata.geotiff_reader

Read GeoTIFF imagery and its spatial metadata via rasterio.
"""
import rasterio


class GeoTIFFReader:
    @staticmethod
    def read(path):
        with rasterio.open(path) as src:
            image = src.read()
            metadata = {
                "bounds": src.bounds,
                "crs": str(src.crs),
                "transform": src.transform,
                "height": src.height,
                "width": src.width,
            }

        return image, metadata
