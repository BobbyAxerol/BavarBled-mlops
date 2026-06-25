import os
import copy
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from tqdm import tqdm

from bavar_bled.src.data.pipeline import DataPipeline
from bavar_bled.src.data.features import FeatureEngineer
from bavar_bled.src.enviroment.portfolio_env import PortfolioEnv
from bavar_bled.src.models.bavar import BAVAREnsemble
from bavar_bled.src.models.bled import BLEDEllipticalSolver
from bavar_bled.src.models.networks import TransformerViewGenerator, CNNRiskNetwork
from bavar_bled.src.agent.replay_buffer import DecoupledReplayBuffer
from bavar_bled.src.agent.td3 import TD3Agent

def preprocess_data(start_date: str = "2014-01-01", end_date: str = "2024-12-31") -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Fetches stock prices, computes indicators and HAR features, and creates continuous windowed states.
    """
    print(f"Fetching raw data from {start_date} to {end_date}...")
    pipeline = DataPipeline(start_date=start_date, end_date=end_date)
    raw_df = pipeline.fetch_raw_data()
    
    print("Computing technical indicators and returns...")
    fe = FeatureEngineer()
    state_matrix = fe.compute_technical_indicators(raw_df)  # (D, 29, 12)
    
    # Feature 11 is return
    returns_matrix = state_matrix[:, :, 11]  # (D, 29)
    
    print("Extracting HAR features...")
    har_matrix = FeatureEngineer.extract_har_features(returns_matrix)  # (D, 29, 4)
    
    D, N, F = state_matrix.shape
    lookback = 15
    
    print("Constructing sliding window states of shape (D, 29, 15, 12)...")
    states = np.zeros((D, N, lookback, F))
    for t in range(lookback - 1, D):
        # Slice lookback window and transpose from (lookback, N, F) to (N, lookback, F)
        states[t] = np.transpose(state_matrix[t - (lookback - 1) : t + 1], (1, 0, 2))
        
    # Splitting into Train (60%), Val (20%), Test (20%)
    n_train = int(D * 0.60)
    n_val = int(D * 0.20)
    
    splits = {
        "train": (states[:n_train], har_matrix[:n_train]),
        "val": (states[n_train : n_train + n_val], har_matrix[n_train : n_train + n_val]),
        "test": (states[n_train + n_val :], har_matrix[n_train + n_val :])
    }
    
    return states, har_matrix, splits

def evaluate_agent(states: np.ndarray, har: np.ndarray, transformer, cnn, bled_solver, agent, train_args) -> Dict[str, float]:
    """
    Runs the agent deterministically on a dataset split and returns performance metrics.
    """
    transformer.eval()
    cnn.eval()
    agent.actor.eval()
    
    env = PortfolioEnv(states, har)
    state, har_feat = env.reset()
    bavar_ensemble = BAVAREnsemble(num_assets=29, num_models=train_args.get("num_models", 600))
    
    total_reward = 0.0
    done = False
    
    with torch.no_grad():
        while not done:
            x_t_current = np.mean(har_feat, axis=0)
            mu_prior, D_prior = bavar_ensemble.generate_priors(x_t_current)
            
            # Formulate inputs for Black-Litterman
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            mu_p_t = torch.FloatTensor(mu_prior).unsqueeze(0).to(agent.device)
            D_p_t = torch.FloatTensor(D_prior).unsqueeze(0).to(agent.device)
            
            views = transformer(state_t)
            delta = cnn(state_t)
            
            w_star_t, _, _ = bled_solver(mu_p_t, D_p_t, views, delta)
            w_star = w_star_t.cpu().numpy().flatten()
            
            # Select action deterministically
            action = agent.actor(w_star_t).cpu().numpy().flatten()
            
            next_state, next_har, reward, done = env.step(action)
            total_reward += reward
            
            # Online step BAVAR
            r_t = state[:, 14, 11]
            bavar_ensemble.step(x_t_current, r_t)
            
            state = next_state
            har_feat = next_har
            
    # Calculate Sharpe ratio of portfolio daily returns
    final_val = env.portfolio_value
    log_return = np.log(final_val / 100000.0)
    
    return {
        "total_reward": total_reward,
        "final_portfolio_value": final_val,
        "sharpe_ratio": float(log_return / (total_reward + 1e-8)) # proxy Sharpe ratio
    }

def train_model(splits: Dict[str, Tuple[np.ndarray, np.ndarray]], train_args: Dict) -> Tuple[Dict[str, torch.nn.Module], Dict[str, float]]:
    """
    Main training loop for BAVAR-BLED.
    """
    train_states, train_har = splits["train"]
    val_states, val_har = splits["val"]
    
    # Initialize networks
    transformer = TransformerViewGenerator(
        num_assets=29, 
        d_model=train_args.get("d_model", 128),
        nhead=train_args.get("nhead", 2),
        num_layers=train_args.get("num_layers", 4)
    )
    cnn = CNNRiskNetwork(
        num_assets=29,
        hidden_size=train_args.get("hidden_size", 512)
    )
    bled_solver = BLEDEllipticalSolver(
        num_assets=29,
        tau=train_args.get("tau_bl", 0.039),
        omega_sigma=train_args.get("omega_sigma", 0.052)
    )
    
    agent = TD3Agent(
        num_assets=29,
        actor_lr=train_args.get("actor_lr", 2.58e-4),
        critic_lr=train_args.get("critic_lr", 6.21e-4),
        gamma=train_args.get("gamma", 0.991),
        transformer=transformer,
        cnn=cnn,
        bled_solver=bled_solver
    )
    
    replay_buffer = DecoupledReplayBuffer(capacity=100000)
    episodes = train_args.get("episodes", 100)
    batch_size = train_args.get("batch_size", 1024)
    
    print(f"Starting training for {episodes} episodes...")
    
    best_val_sharpe = -99999.0
    best_weights = {}
    
    pbar = tqdm(range(episodes), desc="Training progress")
    for ep in pbar:
        env = PortfolioEnv(train_states, train_har)
        state, har_feat = env.reset()
        bavar_ensemble = BAVAREnsemble(num_assets=29, num_models=train_args.get("num_models", 600))
        
        transformer.train()
        cnn.train()
        agent.actor.train()
        
        ep_reward = 0.0
        done = False
        
        # Nested progress bar for steps inside the episode
        total_steps = env.T - env.current_step
        step_pbar = tqdm(total=total_steps, desc=f"Ep {ep+1} Steps", leave=False)
        
        while not done:
            x_t_current = np.mean(har_feat, axis=0)
            mu_prior, D_prior = bavar_ensemble.generate_priors(x_t_current)
            
            # Predict w_star
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            mu_p_t = torch.FloatTensor(mu_prior).unsqueeze(0).to(agent.device)
            D_p_t = torch.FloatTensor(D_prior).unsqueeze(0).to(agent.device)
            
            with torch.no_grad():
                views = transformer(state_t)
                delta = cnn(state_t)
                w_star_t, _, _ = bled_solver(mu_p_t, D_p_t, views, delta)
                w_star = w_star_t.cpu().numpy().flatten()
            
            action = agent.select_action(w_star)
            # Add action noise for TD3
            noise = np.random.normal(0, agent.policy_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
            action = action / (np.sum(np.abs(action)) + 1e-8)
            
            next_state, next_har, reward, done = env.step(action)
            ep_reward += reward
            
            # Update BAVAR
            r_t = state[:, 14, 11]
            bavar_ensemble.step(x_t_current, r_t)
            
            # Compute next step priors
            next_x_t_current = np.mean(next_har, axis=0)
            next_mu_prior, next_D_prior = bavar_ensemble.generate_priors(next_x_t_current)
            
            # Store in replay buffer
            replay_buffer.push(
                state, action, reward, next_state, done, w_star,
                mu_prior, D_prior, next_mu_prior, next_D_prior
            )
            
            # Train step
            if len(replay_buffer) >= batch_size:
                agent.train(replay_buffer, batch_size)
                
            state = next_state
            har_feat = next_har
            step_pbar.update(1)
            
        step_pbar.close()
        
        # Evaluate on validation set after each episode
        val_metrics = evaluate_agent(val_states, val_har, transformer, cnn, bled_solver, agent, train_args)
        val_sharpe = val_metrics["sharpe_ratio"]
        
        # Update progress bar description with key metrics
        pbar.set_postfix({
            "Reward": f"{ep_reward:.4f}",
            "Val Sharpe": f"{val_sharpe:.4f}",
            "Val PortVal": f"{val_metrics['final_portfolio_value']:.1f}"
        })
        
        # Save best model
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_weights = {
                "transformer_state_dict": copy.deepcopy(transformer.state_dict()),
                "cnn_state_dict": copy.deepcopy(cnn.state_dict()),
                "actor_state_dict": copy.deepcopy(agent.actor.state_dict())
            }
            
    # Load best weights
    transformer.load_state_dict(best_weights["transformer_state_dict"])
    cnn.load_state_dict(best_weights["cnn_state_dict"])
    agent.actor.load_state_dict(best_weights["actor_state_dict"])
    
    # Final evaluation on validation set
    final_metrics = evaluate_agent(val_states, val_har, transformer, cnn, bled_solver, agent, train_args)
    
    models = {
        "transformer": transformer,
        "cnn": cnn,
        "bled_solver": bled_solver,
        "agent": agent
    }
    
    return models, final_metrics
