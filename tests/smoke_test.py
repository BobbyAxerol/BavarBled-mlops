import numpy as np
import torch
import sys
import os

# Align python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bavar import BAVAREnsemble
from src.models.bled import BLEDEllipticalSolver
from src.models.networks import TransformerViewGenerator, CNNRiskNetwork
from src.enviroment.portfolio_env import PortfolioEnv
from src.agent.replay_buffer import DecoupledReplayBuffer
from src.agent.td3 import TD3Agent
from src.data.features import FeatureEngineer

def test_pipeline():
    print("Initializing dummy state and HAR features...")
    # 100 days of data for 29 assets, windowed state (29, 15, 12), HAR (29, 4)
    dummy_states = np.random.randn(100, 29, 15, 12)
    dummy_har = np.random.randn(100, 29, 4)
    # Set feature index 11 (returns) to realistic values
    dummy_states[:, :, :, 11] = np.random.normal(0, 0.01, size=(100, 29, 15))
    
    print("Setting up PortfolioEnv...")
    env = PortfolioEnv(dummy_states, dummy_har)
    state, har = env.reset()
    assert state.shape == (29, 15, 12), f"Expected state shape (29, 15, 12), got {state.shape}"
    assert har.shape == (29, 4), f"Expected har shape (29, 4), got {har.shape}"
    print("PortfolioEnv successfully reset.")

    print("Initializing BAVAREnsemble...")
    bavar_ensemble = BAVAREnsemble(num_assets=29, num_models=600)
    x_t_current = np.mean(har, axis=0)
    mu_prior, D_prior = bavar_ensemble.generate_priors(x_t_current)
    assert mu_prior.shape == (29,), f"Expected mu_prior (29,), got {mu_prior.shape}"
    assert D_prior.shape == (29, 29), f"Expected D_prior (29, 29), got {D_prior.shape}"
    print("BAVAREnsemble successfully initialized and priors generated.")

    print("Initializing Deep Learning networks...")
    transformer = TransformerViewGenerator(num_assets=29)
    cnn = CNNRiskNetwork(num_assets=29)
    bled_solver = BLEDEllipticalSolver(num_assets=29)
    
    state_t = torch.FloatTensor(state).unsqueeze(0)
    mu_p_t = torch.FloatTensor(mu_prior).unsqueeze(0)
    D_p_t = torch.FloatTensor(D_prior).unsqueeze(0)
    
    views = transformer(state_t)
    delta = cnn(state_t)
    assert views.shape == (1, 29), f"Expected views shape (1, 29), got {views.shape}"
    assert delta.shape == (1, 1), f"Expected delta shape (1, 1), got {delta.shape}"
    print("Transformer and CNN output shapes verified.")

    print("Solving Black-Litterman system...")
    w_star_t, mu_BL, D_BL = bled_solver(mu_p_t, D_p_t, views, delta)
    assert w_star_t.shape == (1, 29), f"Expected w_star_t shape (1, 29), got {w_star_t.shape}"
    w_star = w_star_t.detach().cpu().numpy().flatten()
    print("Black-Litterman system successfully solved.")

    print("Initializing TD3Agent...")
    agent = TD3Agent(
        num_assets=29,
        transformer=transformer,
        cnn=cnn,
        bled_solver=bled_solver
    )
    action = agent.select_action(w_star)
    assert action.shape == (30,), f"Expected action shape (30,), got {action.shape}"
    print("TD3Agent successfully selected action.")

    print("Running env.step...")
    next_state, next_har, reward, done = env.step(action)
    assert next_state.shape == (29, 15, 12), f"Expected next_state shape (29, 15, 12), got {next_state.shape}"
    assert next_har.shape == (29, 4), f"Expected next_har shape (29, 4), got {next_har.shape}"
    print(f"Env step executed successfully. Reward: {reward:.4f}, Done: {done}")

    print("Updating BAVAREnsemble...")
    r_t = state[:, 14, 11]
    bavar_ensemble.step(x_t_current, r_t)
    # Check that weights sum to 1
    assert np.allclose(np.sum(bavar_ensemble.weights), 1.0), f"Weights sum: {np.sum(bavar_ensemble.weights)}"
    print("BAVAREnsemble weights successfully updated online.")

    print("Verifying Replay Buffer push & sample...")
    replay_buffer = DecoupledReplayBuffer(capacity=10)
    next_x_t = np.mean(next_har, axis=0)
    next_mu, next_D = bavar_ensemble.generate_priors(next_x_t)
    
    replay_buffer.push(
        state, action, reward, next_state, done, w_star,
        mu_prior, D_prior, next_mu, next_D
    )
    assert len(replay_buffer) == 1
    
    s_b, a_b, r_b, ns_b, d_b, ws_b, mu_b, D_b, nmu_b, nD_b = replay_buffer.sample(1)
    assert s_b.shape == (1, 29, 15, 12)
    assert a_b.shape == (1, 30)
    assert r_b.shape == (1, 1)
    assert ns_b.shape == (1, 29, 15, 12)
    assert d_b.shape == (1, 1)
    assert ws_b.shape == (1, 29)
    assert mu_b.shape == (1, 29)
    assert D_b.shape == (1, 29, 29)
    assert nmu_b.shape == (1, 29)
    assert nD_b.shape == (1, 29, 29)
    print("Replay buffer sample verified.")
    
    print("Verifying TD3 training step...")
    agent.train(replay_buffer, batch_size=1)
    print("TD3Agent train step completed successfully.")

    print("SMOKE TEST SUCCESSFUL!")

if __name__ == '__main__':
    test_pipeline()
