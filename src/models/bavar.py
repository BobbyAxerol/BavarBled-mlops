import numpy as np
from numba import njit, prange

@njit(fastmath=True, cache=True)
def _bavar_recursive_step(Phi, B_bar, nu, Lambda, weights, x, r, n, M, k):
    """
    Numba-accelerated vectorized core calculation for Bayesian updating step.
    """
    likelihoods = np.zeros(M)
    x_t = np.ascontiguousarray(x).reshape(k, 1)
    r_t = np.ascontiguousarray(r).reshape(n, 1)
    
    # 1. Compute Predictive Likelihoods
    for m in range(M):
        mu_m = B_bar[m] @ x_t
        Sigma_m = Lambda[m] / (nu[m] - n - 1)
        # Epsilon modification for strict positive-definiteness
        for i in range(n):
            Sigma_m[i, i] += 1e-6
            
        # Evaluation of Gaussian multivariate density exponent
        diff = r_t - mu_m
        sign, logdet = np.linalg.slogdet(Sigma_m)
        inv_Sigma = np.linalg.inv(Sigma_m)
        quad_form = (diff.T @ inv_Sigma @ diff)[0, 0]
        
        prob = np.exp(-0.5 * quad_form) / (np.sqrt((2 * np.pi)**n * np.exp(logdet)) + 1e-12)
        likelihoods[m] = prob

    # Update Model Probability Weights via Bayes Rule
    weights *= likelihoods
    sum_w = np.sum(weights)
    if sum_w > 1e-12:
        weights /= sum_w
    else:
        weights = np.ones(M) / M
        
    # 2. Sequential Statistics Updating Block
    for m in range(M):
        Phi_inv_old = np.linalg.inv(Phi[m])
        Phi[m] = np.linalg.inv(Phi_inv_old + x_t @ x_t.T)
        
        e_m = r_t - B_bar[m] @ x_t
        scaling_factor = 1.0 + (x_t.T @ Phi_inv_old @ x_t)[0, 0]
        
        Lambda[m] = Lambda[m] + (e_m @ e_m.T) / scaling_factor
        nu[m] += 1.0
        B_bar[m] = (B_bar[m] @ Phi_inv_old + r_t @ x_t.T) @ Phi[m]
        
    return Phi, B_bar, nu, Lambda, weights

@njit(fastmath=True, cache=True)
def _bavar_generate_priors(B_bar, Lambda, nu, weights, x_t_current, n, M, k):
    """
    Numba-accelerated computation of Bayesian Model Averaging priors.
    """
    mu_bma = np.zeros(n)
    D_bma = np.zeros((n, n))
    x_col = np.ascontiguousarray(x_t_current).reshape(k, 1)
    
    for m in range(M):
        mu_m = (B_bar[m] @ x_col).flatten()
        Sigma_m = Lambda[m] / (nu[m] - n - 1)
        
        mu_bma += weights[m] * mu_m
        D_bma += weights[m] * (Sigma_m + np.outer(mu_m, mu_m))
        
    D_bma -= np.outer(mu_bma, mu_bma)
    return mu_bma, D_bma

class BAVAREnsemble:
    def __init__(self, num_assets: int = 29, num_models: int = 600):
        self.n = num_assets
        self.M = num_models
        self.k = 4 # HAR dimension
        
        alphas = [0.01, 0.1, 1.0]
        betas = [1.0, 10.0, 100.0]
        combos = [(a, b) for a in alphas for b in betas] # 9 strategic variants
        
        self.Phi = np.zeros((self.M, self.k, self.k))
        for m in range(self.M):
            a, b = combos[m % 9]
            self.Phi[m] = np.diag(np.array([a, b, b, b]))
            
        self.B_bar = np.zeros((self.M, self.n, self.k))
        self.nu = np.ones(self.M) * (self.n + 2)
        self.Lambda = np.zeros((self.M, self.n, self.n))
        for m in range(self.M):
            self.Lambda[m] = 0.01 * np.eye(self.n)
            
        self.weights = np.ones(self.M) / self.M
 
    def step(self, x_t_minus_1: np.ndarray, r_t: np.ndarray):
        self.Phi, self.B_bar, self.nu, self.Lambda, self.weights = _bavar_recursive_step(
            self.Phi, self.B_bar, self.nu, self.Lambda, self.weights,
            x_t_minus_1, r_t, self.n, self.M, self.k
        )
 
    def generate_priors(self, x_t_current: np.ndarray):
        """
        Combines individual VAR models via Bayesian Model Averaging (BMA) to output mu and D.
        """
        return _bavar_generate_priors(
            self.B_bar, self.Lambda, self.nu, self.weights, x_t_current, self.n, self.M, self.k
        )