import torch
import torch.nn as nn

class BLEDEllipticalSolver(nn.Module):
    def __init__(self, num_assets=29, tau=0.039, omega_sigma=0.052):
        super(BLEDEllipticalSolver, self).__init__()
        self.n = num_assets
        self.tau = tau
        self.omega_sigma = omega_sigma
        self.register_buffer('P', torch.eye(num_assets)) # Identity mapping for Absolute Views

    def forward(self, mu_prior, D_prior, Q_views, delta_risk):
        """
        Calculates posterior expected returns and analytical allocation weights.
        All matrices dimensionally aligned with batch inputs.
        """
        B_size = mu_prior.size(0)
        device = mu_prior.device
        
        Omega = (self.omega_sigma ** 2) * torch.eye(self.n, device=device).unsqueeze(0).repeat(B_size, 1, 1)
        P_b = self.P.unsqueeze(0).repeat(B_size, 1, 1)
        
        # 1. Posterior Expected Returns Computation (Mu_BL)
        tau_D_inv = torch.inverse(self.tau * D_prior)
        Omega_inv = torch.inverse(Omega)
        
        info_matrix = torch.inverse(tau_D_inv + torch.bmm(torch.bmm(P_b.transpose(1, 2), Omega_inv), P_b))
        view_diff = Q_views.unsqueeze(-1) - torch.bmm(P_b, mu_prior.unsqueeze(-1))
        
        mu_BL = mu_prior + torch.bmm(torch.bmm(info_matrix, P_b.transpose(1, 2)), torch.bmm(Omega_inv, view_diff)).squeeze(-1)
        
        # 2. Posterior Dispersion Matrix Computation (D_BL)
        middle_term = torch.inverse(Omega + torch.bmm(torch.bmm(P_b, D_prior), P_b.transpose(1, 2)))
        D_BL = D_prior - torch.bmm(torch.bmm(D_prior, P_b.transpose(1, 2)), torch.bmm(middle_term, torch.bmm(P_b, D_prior)))
        
        # 3. Canonical Markowitz Weights Injection Under Tail Risks
        D_BL_inv = torch.inverse(D_BL)
        w_star = torch.bmm(D_BL_inv, mu_BL.unsqueeze(-1)).squeeze(-1)
        w_star = w_star / (delta_risk + 1e-8)
        
        return w_star, mu_BL, D_BL