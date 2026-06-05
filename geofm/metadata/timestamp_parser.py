"""geofm.metadata.timestamp_parser

Parse acquisition timestamps into datetimes and day-of-year.

TODO (later): support GeoTIFF tags, STAC metadata, Sentinel/Landsat filenames.
"""
from datetime import datetime


class TimestampParser:
    @staticmethod
    def parse(date_string):
        return datetime.strptime(date_string, "%Y-%m-%d")

    @staticmethod
    def day_of_year(dt):
        return dt.timetuple().tm_yday
