"""geofm.metadata.coordinate_parser

Derive geographic coordinates (e.g. scene center) from raster bounds.
"""


class CoordinateParser:
    @staticmethod
    def center_latlon(bounds):
        lat = (bounds.top + bounds.bottom) / 2
        lon = (bounds.left + bounds.right) / 2
        return lat, lon
