"""Shared SparkSession factory for local jobs and tests."""

from __future__ import annotations

import os
from pathlib import Path

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

_HOMEBREW_JAVA_CANDIDATES = (
    Path("/opt/homebrew/opt/openjdk@17"),
    Path("/opt/homebrew/opt/openjdk@21"),
    Path("/opt/homebrew/opt/openjdk@11"),
    Path("/opt/homebrew/opt/openjdk"),
    Path("/usr/local/opt/openjdk@17"),
    Path("/usr/local/opt/openjdk"),
)


def _java_home_from_path(root: Path) -> Path | None:
    if (root / "bin" / "java").exists():
        return root
    bundled = root / "libexec" / "openjdk.jdk" / "Contents" / "Home"
    if (bundled / "bin" / "java").exists():
        return bundled
    return None


def ensure_java_home() -> Path:
    """Point JAVA_HOME at a JDK so PySpark can start the JVM.

    Homebrew OpenJDK is keg-only, so macOS ``java`` is missing unless
    JAVA_HOME is set. Call this before SparkSession.getOrCreate().
    """
    existing = os.environ.get("JAVA_HOME")
    if existing and Path(existing, "bin", "java").exists():
        java_home = Path(existing)
    else:
        java_home = None
        for candidate in _HOMEBREW_JAVA_CANDIDATES:
            java_home = _java_home_from_path(candidate)
            if java_home is not None:
                break
        if java_home is None:
            raise RuntimeError(
                "Java is required for PySpark. Install OpenJDK 17 and retry:\n"
                "  brew install openjdk@17\n"
                "  export JAVA_HOME=\"/opt/homebrew/opt/openjdk@17\"\n"
                "  export PATH=\"$JAVA_HOME/bin:$PATH\""
            )
        os.environ["JAVA_HOME"] = str(java_home)

    java_bin = str(java_home / "bin")
    path = os.environ.get("PATH", "")
    if java_bin not in path.split(":"):
        os.environ["PATH"] = java_bin + os.pathsep + path
    return java_home


def get_spark_session(
    app_name: str,
    *,
    master: str | None = None,
    extra_config: dict[str, str] | None = None,
) -> SparkSession:
    """Build or reuse a SparkSession.

    Uses ``SparkConf`` and ``SparkSession.Builder()`` so checkers do not
    flag ``SparkSession.builder.appName`` (a JVM/classmethod proxy).

    Pass ``master`` only for local/tests. Cluster runs should omit it so
    spark-submit can set the master.
    """
    ensure_java_home()
    conf = SparkConf()
    conf.setAppName(app_name)
    if master:
        conf.setMaster(master)
    conf.set("spark.sql.session.timeZone", "UTC")
    conf.set("spark.ui.enabled", "false")
    conf.set("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
    conf.set("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
    for key, value in (extra_config or {}).items():
        conf.set(key, value)

    builder = SparkSession.Builder()
    builder.config(conf=conf)
    session = builder.getOrCreate()
    return session
