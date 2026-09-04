from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

city = Entity(name="city", join_keys=["city_id"])

aqi_source = FileSource(
    path="data/hourly_city_aqi.parquet",
    timestamp_field="timestamp",
)

hourly_aqi_view = FeatureView(
    name="hourly_city_aqi",
    entities=[city],
    ttl=timedelta(days=365),
    schema=[
        Field(name="aqi", dtype=Float32),
        Field(name="pm25", dtype=Float32),
        Field(name="pm10", dtype=Float32),
        Field(name="temperature", dtype=Float32),
        Field(name="humidity", dtype=Float32),
        Field(name="aqi_change_rate", dtype=Float32),
        Field(name="hour", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="month", dtype=Int64),
    ],
    source=aqi_source,
)