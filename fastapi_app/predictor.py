from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
# Force Spark's Py4J bridge to bind to the explicit local IPv4 address
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
SPARK_LOG_LEVEL = os.getenv("SPARK_LOG_LEVEL", "WARN").strip().upper() or "WARN"


class SpamPredictor:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.spark: SparkSession | None = None
        self.model: PipelineModel | None = None

    def load(self) -> None:
        if self.spark is None:
            python_exec = os.getenv("PYSPARK_PYTHON", sys.executable)
            driver_exec = os.getenv("PYSPARK_DRIVER_PYTHON", python_exec)
            self.spark = (
                SparkSession.builder.appName("sms-spam-fastapi")
                .master("local[*]")
                .config("spark.pyspark.python", python_exec)
                .config("spark.pyspark.driver.python", driver_exec)
                .config("spark.executorEnv.PYSPARK_PYTHON", python_exec)
                .config("spark.executorEnv.PYSPARK_DRIVER_PYTHON", driver_exec)
                .getOrCreate()
            )
            self.spark.sparkContext.setLogLevel(SPARK_LOG_LEVEL)

        if self.model is None:
            model_dir = Path(self.model_path).expanduser()
            if not model_dir.is_absolute():
                model_dir = (Path.cwd() / model_dir).resolve()

            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Model path not found: {model_dir}. "
                    "Set MODEL_PATH to a valid Spark PipelineModel directory."
                )

            self.model_path = str(model_dir)
            self.model = PipelineModel.load(self.model_path)

    def predict(self, messages: list[str]) -> list[dict[str, object]]:
        if not messages:
            return []

        self.load()
        assert self.spark is not None
        assert self.model is not None

        values_clause = ", ".join(
            f"({idx}, {self._sql_quote(msg)})" for idx, msg in enumerate(messages)
        )
        input_df = self.spark.sql(
            f"SELECT * FROM VALUES {values_clause} AS t(row_id, message)"
        )

        scored_df = self.model.transform(input_df)

        if "probability" in scored_df.columns:
            scored_df = scored_df.withColumn(
                "score_array", vector_to_array("probability")
            )
            scored_df = scored_df.withColumn("spam_score", F.col("score_array")[1])
        else:
            scored_df = scored_df.withColumn(
                "score_array", vector_to_array("rawPrediction")
            )
            scored_df = scored_df.withColumn(
                "margin",
                F.col("score_array")[1] - F.col("score_array")[0],
            )
            scored_df = scored_df.withColumn(
                "spam_score",
                1 / (1 + F.exp(-F.col("margin"))),
            )

        ordered_rows = (
            scored_df.select("row_id", "message", "prediction", "spam_score")
            .orderBy("row_id")
            .collect()
        )

        predictions: list[dict[str, object]] = []
        for row in ordered_rows:
            prediction = int(row["prediction"])
            predictions.append(
                {
                    "message": row["message"],
                    "prediction": prediction,
                    "label": "spam" if prediction == 1 else "ham",
                    "spam_score": float(row["spam_score"]),
                }
            )

        return predictions

    @staticmethod
    def _sql_quote(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace("'", "''")
        return f"'{escaped}'"
