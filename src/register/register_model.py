import argparse
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bavar_bled_model')
    parser.add_argument('--run_id', type=str)
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'))

    run = mlflow.get_run(args.run_id)
    model_uri = f"runs:/{args.run_id}/model"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name,
        tags={
            "git_sha": run.data.tags.get("git_sha", "unknown"),
            "run_id": args.run_id,
            "experiment_name": run.info.experiment_id
        }
    )

    print(f"Registered model version: {model_version.version}")

if __name__ == '__main__':
    main()
