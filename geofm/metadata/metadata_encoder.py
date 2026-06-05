"""geofm.metadata.metadata_encoder

Cyclic (sin/cos) encoding of latitude, longitude, and day-of-year.
Pure preprocessing -- no learnable parameters.
"""
import numpy as np


class MetadataEncoder:
    @staticmethod
    def encode(lat, lon, day):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        day_angle = 2 * np.pi * day / 365

        return {
            "lat_sin": np.sin(lat_rad),
            "lat_cos": np.cos(lat_rad),
            "lon_sin": np.sin(lon_rad),
            "lon_cos": np.cos(lon_rad),
            "day_sin": np.sin(day_angle),
            "day_cos": np.cos(day_angle),
        }
