# pylint: disable=redefined-outer-name
from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession

from spark.session import get_spark_session


@pytest.fixture(scope="session")
def spark_session() -> Iterator[SparkSession]:
    session = get_spark_session("test_clean_taxi", master="local[1]")
    yield session
    session.stop()
