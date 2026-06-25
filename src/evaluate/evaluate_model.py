import os
import argparse
import mlflow
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bavar_bled_model')
    parser.add_argument('--run_id', type=str)
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'))

    metric_eval = 'sharpe_ratio'

    new_run = mlflow.get_run(args.run_id)
    new_metric = float(new_run.data.metrics.get(metric_eval, 0))
    version = new_run.data.tags.get("mlflow.model.version", "1")

    client = mlflow.tracking.MlflowClient()
    try:
        production_versions = client.get_latest_versions(args.model_name, stages=["Production"])
    except Exception:
        production_versions = []

    if production_versions:
        prod_run = mlflow.get_run(production_versions[0].run_id)
        prod_metric = float(prod_run.data.metrics.get(metric_eval, 0))
        
        print(f"Comparing new model ({metric_eval}={new_metric:.4f}) with production ({metric_eval}={prod_metric:.4f})...")
        if new_metric > prod_metric:
            print("New model is better! Transitioning to Production.")
            client.transition_model_version_stage(
                name=args.model_name,
                version=version,
                stage="Production"
            )
            client.transition_model_version_stage(
                name=args.model_name,
                version=production_versions[0].version,
                stage="Archived"
            )
        else:
            print("New model is not better. Transitioning to Staging.")
            client.transition_model_version_stage(
                name=args.model_name,
                version=version,
                stage="Staging"
            )
    else:
        print("No production model found. Promoting new model to Production.")
        client.transition_model_version_stage(
            name=args.model_name,
            version=version,
            stage="Production"
        )

if __name__ == '__main__':
    main()
