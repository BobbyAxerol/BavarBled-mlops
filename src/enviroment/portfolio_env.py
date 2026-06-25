import numpy as np

class PortfolioEnv:
    def __init__(self, state_tensor: np.ndarray, har_features: np.ndarray, transaction_cost: float = 0.0025):
        """
        state_tensor: shape (T, 29, 15, 12)
        har_features: shape (T, 29, 4)
        """
        self.states = state_tensor
        self.har = har_features
        self.c = transaction_cost
        self.T = state_tensor.shape[0]
        self.reset()

    def reset(self):
        self.current_step = 22 # Offset matching minimum initialization for HAR historical bounds
        self.portfolio_value = 100000.0 # Standard base capital size matching baseline evaluation parameters
        self.w_prev = np.zeros(30)
        self.w_prev[-1] = 1.0 # 100% Cash holding strategy initialization
        return self.states[self.current_step], self.har[self.current_step]

    def step(self, action_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """
        action_weights: shape (30,) -> Verified normalized targets
        """
        # 1. Fetch current timestep actual market excess returns
        # Feature 11 is raw asset log return calculated inside engineer pipeline
        r_market = self.states[self.current_step, :, 14, 11]
        
        # Expand market returns matrix to include zero cash return reference anchor
        r_vector = np.append(r_market, 0.0) 
        
        # 2. Mathematical portfolio simulation equations
        r_p = np.sum(self.w_prev * r_vector)
        
        # Dynamic transaction friction turnover penalty calculation
        turnover = np.sum(np.abs(action_weights - self.w_prev))
        mu_t = self.c * turnover
        
        # Absolute Portfolio value geometric compounding step
        v_next = self.portfolio_value * (1.0 + r_p) * (1.0 - mu_t)
        
        # Strict log return metric computation for optimization reward convergence
        reward = float(np.log(v_next + 1e-8) - np.log(self.portfolio_value + 1e-8))
        
        self.portfolio_value = v_next
        self.w_prev = action_weights.copy()
        
        self.current_step += 1
        done = self.current_step >= self.T - 1
        
        if not done:
            next_s = self.states[self.current_step]
            next_h = self.har[self.current_step]
        else:
            next_s, next_h = np.zeros_like(self.states[0]), np.zeros_like(self.har[0])
            
        return next_s, next_h, reward, done