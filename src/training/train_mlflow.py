import os
import json
import torch
import mlflow
import mlflow.pyfunc
import argparse
import datetime
import shutil
import subprocess
from dotenv import load_dotenv

load_dotenv()

from bavar_bled.src.training.train import preprocess_data, train_model

def _get_git_metadata():
    try:
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode("utf-8").strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir).decode("utf-8").strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir).decode("utf-8").strip()
        is_dirty = len(status) > 0
        return sha, branch, is_dirty
    except Exception:
        return "unknown", "unknown", True

class BavarBledModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from bavar_bled.src.models.networks import TransformerViewGenerator, CNNRiskNetwork
        from bavar_bled.src.models.bled import BLEDEllipticalSolver
        from bavar_bled.src.agent.td3 import TD3Actor
        
        checkpoint = torch.load(context.artifacts["model_bundle"], map_location="cpu")
        
        self.transformer = TransformerViewGenerator(num_assets=29)
        self.transformer.load_state_dict(checkpoint["transformer_state_dict"])
        self.transformer.eval()
        
        self.cnn = CNNRiskNetwork(num_assets=29)
        self.cnn.load_state_dict(checkpoint["cnn_state_dict"])
        self.cnn.eval()
        
        self.actor = TD3Actor(num_assets=29)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor.eval()
        
        self.bled_solver = BLEDEllipticalSolver(num_assets=29)
        
    def predict(self, context, model_input):
        import torch
        import numpy as np
        
        if isinstance(model_input, dict):
            state_tensor = np.array(model_input["state_tensor"], dtype=np.float32)
            mu_prior = np.array(model_input["mu_prior"], dtype=np.float32)
            D_prior = np.array(model_input["D_prior"], dtype=np.float32)
        else:
            # Pandas dataframe or dictionary-like object
            try:
                state_tensor = np.array(model_input.get("state_tensor"), dtype=np.float32)
                mu_prior = np.array(model_input.get("mu_prior"), dtype=np.float32)
                D_prior = np.array(model_input.get("D_prior"), dtype=np.float32)
            except Exception:
                # If parsed as direct dictionary from json
                state_tensor = np.array(model_input["state_tensor"].iloc[0], dtype=np.float32)
                mu_prior = np.array(model_input["mu_prior"].iloc[0], dtype=np.float32)
                D_prior = np.array(model_input["D_prior"].iloc[0], dtype=np.float32)
            
        state_t = torch.FloatTensor(state_tensor)
        mu_p_t = torch.FloatTensor(mu_prior)
        D_p_t = torch.FloatTensor(D_prior)
        
        if len(state_t.shape) == 3:
            state_t = state_t.unsqueeze(0)
            mu_p_t = mu_p_t.unsqueeze(0)
            D_p_t = D_p_t.unsqueeze(0)
            
        with torch.no_grad():
            views = self.transformer(state_t)
            delta = self.cnn(state_t)
            w_star_t, _, _ = self.bled_solver(mu_p_t, D_p_t, views, delta)
            action = self.actor(w_star_t).cpu().numpy().squeeze(0)
            
        return action.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bavar_bled_model')
    args = parser.parse_args()
    
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'bavar_bled')
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Load training configs
    param_path = 'bavar_bled/parameters.json'
    if not os.path.exists(param_path):
        param_path = 'parameters.json'
        
    with open(param_path) as f:
        pars = json.load(f)
    train_args = pars.get('training', {})
    dataset_args = pars.get('dataset', {})
    features_args = pars.get('features', {})
    
    lookback = features_args.get('lookback', 15)
    asset_class = dataset_args.get('asset_class', 'equity')
    universe_name = dataset_args.get('universe_name', 'djia_29')
    
    start_date = os.environ.get('START_DATE', '2014-01-01')
    end_date = os.environ.get('END_DATE', '2024-12-31')
    
    if os.environ.get('EPISODES'):
        train_args['episodes'] = int(os.environ.get('EPISODES'))
    if os.environ.get('BATCH_SIZE'):
        train_args['batch_size'] = int(os.environ.get('BATCH_SIZE'))
        
    # Preprocess data
    states, har_matrix, splits = preprocess_data(start_date=start_date, end_date=end_date)
    
    staging_root = os.path.abspath(".code_staging")
    staging_pkg = os.path.join(staging_root, "bavar_bled")
    staging_src = os.path.join(staging_pkg, "src")
    
    try:
        # Clean any leftover staging
        if os.path.exists(staging_root):
            shutil.rmtree(staging_root)
            
        os.makedirs(staging_src)
        
        # Create empty __init__.py inside package
        with open(os.path.join(staging_pkg, "__init__.py"), "w") as f:
            f.write("")
            
        # Copy src tree
        shutil.copytree("src", staging_src, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        
        with mlflow.start_run() as run:
            sha, branch, is_dirty = _get_git_metadata()
            mlflow.set_tag("git_sha", os.environ.get("GITHUB_SHA", sha))
            mlflow.set_tag("git_branch", branch)
            mlflow.set_tag("git_dirty", str(is_dirty))
            
            mlflow.log_params(train_args)
            mlflow.log_param("lookback", lookback)
            mlflow.log_param("asset_class", asset_class)
            mlflow.log_param("universe_name", universe_name)
            mlflow.log_param("start_date", start_date)
            mlflow.log_param("end_date", end_date)
            
            # Train model
            models, metrics = train_model(splits, train_args)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Save PyTorch bundle weights
            checkpoint = {
                "transformer_state_dict": models["transformer"].state_dict(),
                "cnn_state_dict": models["cnn"].state_dict(),
                "actor_state_dict": models["agent"].actor.state_dict()
            }
            bundle_path = "model_bundle.pth"
            torch.save(checkpoint, bundle_path)
            
            # Log model wrapper with bundled code
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=BavarBledModelWrapper(),
                artifacts={"model_bundle": bundle_path},
                code_paths=[staging_pkg]
            )
            
            # Register model version and update description card
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                model_version = mlflow.register_model(
                    model_uri=model_uri,
                    name=args.model_name
                )
                mlflow.set_tag("mlflow.model.version", model_version.version)
                
                client = mlflow.tracking.MlflowClient()
                desc = (
                    f"### Model Version Card\n"
                    f"- **Trained At**: {datetime.datetime.now().isoformat()}\n"
                    f"- **Git Commit**: {sha} (Branch: {branch}, Dirty: {is_dirty})\n"
                    f"- **Hyperparameters**: batch_size={train_args.get('batch_size')}, "
                    f"episodes={train_args.get('episodes')}, num_models={train_args.get('num_models')}, "
                    f"actor_lr={train_args.get('actor_lr')}, critic_lr={train_args.get('critic_lr')}\n"
                    f"- **Data Period**: {start_date} to {end_date}\n"
                    f"- **Features**: lookback={lookback}\n"
                    f"- **Dataset**: Asset Class={asset_class}, Universe={universe_name}\n"
                    f"- **Performance (Validation Set)**:\n"
                    f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.4f}\n"
                    f"  - Total Reward: {metrics.get('total_reward', 0.0):.4f}\n"
                    f"  - Final Portfolio Value: {metrics.get('final_portfolio_value', 0.0):.1f}\n"
                )
                client.update_model_version(
                    name=args.model_name,
                    version=model_version.version,
                    description=desc
                )
            except Exception as e:
                print(f"Warning: model registration or card update failed: {e}")
                mlflow.set_tag("mlflow.model.version", "1")
                
            mlflow.set_tag("run_id", run.info.run_id)
            
            # Create performance report
            report = []
            report.append("====================================================")
            report.append("          BAVAR-BLED PERFORMANCE REPORT            ")
            report.append("====================================================")
            report.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
            report.append(f"Episodes: {train_args.get('episodes')}")
            report.append(f"Batch Size: {train_args.get('batch_size')}")
            report.append(f"Data Period: {start_date} to {end_date}")
            report.append("----------------------------------------------------")
            report.append("Performance Metrics (Validation Set):")
            for k, v in metrics.items():
                report.append(f"  {k}: {v}")
            report.append("====================================================")
            
            report_text = "\n".join(report)
            print(report_text)
            with open("performance_report.txt", "w") as rf:
                rf.write(report_text)
    finally:
        if os.path.exists(staging_root):
            shutil.rmtree(staging_root)
            
if __name__ == '__main__':
    main()
