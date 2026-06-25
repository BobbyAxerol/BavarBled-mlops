import numpy as np
import random
from collections import deque
from typing import Dict, Tuple

class DecoupledReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool, w_star: np.ndarray,
             mu_prior: np.ndarray, D_prior: np.ndarray,
             next_mu_prior: np.ndarray, next_D_prior: np.ndarray):
        self.buffer.append((state, action, reward, next_state, done, w_star, mu_prior, D_prior, next_mu_prior, next_D_prior))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, w_star, mu_prior, D_prior, next_mu_prior, next_D_prior = zip(*batch)
        
        return (np.array(state, dtype=np.float32), 
                np.array(action, dtype=np.float32), 
                np.array(reward, dtype=np.float32).reshape(-1, 1), 
                np.array(next_state, dtype=np.float32), 
                np.array(done, dtype=np.float32).reshape(-1, 1),
                np.array(w_star, dtype=np.float32),
                np.array(mu_prior, dtype=np.float32),
                np.array(D_prior, dtype=np.float32),
                np.array(next_mu_prior, dtype=np.float32),
                np.array(next_D_prior, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buffer)