"""Explicit MLflow publication and idempotent candidate registration."""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from retention.models.final_training import ARTIFACT, MANIFEST
from retention.models.predictor import load_predictor
from retention.models.training import _checksum
from retention.utils.config import load_config

EVALUATION = Path("reports/generated/final_evaluation/evaluation.json")
MODEL_NAME = "customer-churn-predictor"


class InferenceModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.predictor = load_predictor(context.artifacts["predictor"])

    def predict(self, context, model_input, params=None):
        return self.predictor.predict(model_input)


def register(config_path="params.yaml"):
    config = load_config(config_path)
    manifest = json.loads(MANIFEST.read_text())
    evaluation = json.loads(EVALUATION.read_text())
    if (
        manifest["readiness"] != "passed"
        or _checksum(ARTIFACT) != manifest["model_sha256"]
    ):
        raise ValueError("Final artifact failed readiness verification")
    if evaluation["fingerprints"]["model_sha256"] != manifest["model_sha256"]:
        raise ValueError("Evaluated model differs from final artifact")
    mlflow.set_tracking_uri(config["model_training"]["mlflow_tracking_uri"])
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except mlflow.exceptions.RestException:
        versions = []
    for version in versions:
        if (
            version.tags.get("model_sha256") == manifest["model_sha256"]
            and version.tags.get("configuration_sha256")
            == manifest["configuration_sha256"]
        ):
            client.set_registered_model_alias(MODEL_NAME, "candidate", version.version)
            return {
                "run_id": version.run_id,
                "version": version.version,
                "alias": "candidate",
                "reused": True,
            }
    predictor = load_predictor(ARTIFACT)
    example = pd.DataFrame(
        [
            {
                name: (
                    1
                    if name in predictor.numeric_columns
                    else predictor.categories[name][0]
                )
                for name in predictor.columns
            }
        ]
    )
    expected = predictor.predict(example)
    signature = infer_signature(example, expected)
    mlflow.set_experiment("customer-retention-final-model")
    with mlflow.start_run(run_name="frozen_final_candidate") as run:
        mlflow.set_tags(
            {
                "phase": "6B",
                "model_sha256": manifest["model_sha256"],
                "configuration_sha256": manifest["configuration_sha256"],
                "test_sha256": evaluation["fingerprints"]["test_sha256"],
            }
        )
        mlflow.log_params(
            {
                "model_family": "xgboost",
                "feature_mode": "baseline",
                "calibration": "sigmoid",
                "threshold": predictor.threshold,
                "development_sha256": manifest["development_sha256"],
            }
        )
        mlflow.log_metrics(
            {
                key: value
                for key, value in evaluation["primary"].items()
                if isinstance(value, (int, float)) and key != "batch_inference_seconds"
            }
        )
        mlflow.log_artifact(
            "configs/selected_model.yaml", artifact_path="configuration"
        )
        mlflow.log_artifact(str(MANIFEST), artifact_path="readiness")
        mlflow.log_artifacts(str(EVALUATION.parent), artifact_path="evaluation")
        mlflow.pyfunc.log_model(
            name="inference",
            python_model=InferenceModel(),
            artifacts={"predictor": str(ARTIFACT)},
            input_example=example,
            signature=signature,
            code_paths=["src"],
        )
        uri = f"runs:/{run.info.run_id}/inference"
        loaded = mlflow.pyfunc.load_model(uri)
        actual = loaded.predict(example)
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True)
        )
        version = mlflow.register_model(uri, MODEL_NAME)
        client.set_model_version_tag(
            MODEL_NAME, version.version, "model_sha256", manifest["model_sha256"]
        )
        client.set_model_version_tag(
            MODEL_NAME,
            version.version,
            "configuration_sha256",
            manifest["configuration_sha256"],
        )
        client.set_registered_model_alias(MODEL_NAME, "candidate", version.version)
        return {
            "run_id": run.info.run_id,
            "version": version.version,
            "alias": "candidate",
            "reused": False,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    print(json.dumps(register(args.config), indent=2))


if __name__ == "__main__":
    main()
