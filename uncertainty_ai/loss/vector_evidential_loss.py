import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Softplus
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss
import math


class VectorRegressionEvidentialLoss(_Loss):
    def __init__(self, epsilon=1e-6) -> None:
        super().__init__()
        # regularization weight λ and numerical eps
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.d = 3
        
    def forward(self, mu, S, lognu, targets):
        """
        :mu:     (B,3)   predicted means
        :L:      (B,1) 
        :kappa:  (B,1) 
        :nu:     (B,1)
        :targets:(B,3)
        """
        B, self.d = mu.shape
        assert self.d == self.d, f"Expected d={self.d}, but got d={self.d} in mu.shape={mu.shape}"

        #–– 1) form scalar κ, ν
        # kappa = F.softplus(kappa)  # (B,1) -> (B,)
        # kappa = torch.clamp(kappa, min=self.epsilon) #+ self.epsilon     # (B,)
        
        # kappa = torch.clamp(kappa, min=1e-3)  # Prevent extremely small kappa
        nu = torch.sigmoid(lognu)*(12-self.d) + (self.d+1)   # (B,1) -> (B,)
        nu = torch.clamp(nu, min=(self.d+1) + self.epsilon)  #+ self.epsilon     # (B,)
        kappa = nu

        # Regularize Ψ for numerical stability
        I = torch.eye(self.d, device=S.device, dtype=S.dtype).view(-1, 3, 3)
        Psi_reg = S + self.epsilon * I
        # Psi_reg = S
        # Compute log|Ψ|
        sign, log_det_Psi = torch.linalg.slogdet(Psi_reg)
        log_det_Psi = torch.where(sign > 0, log_det_Psi, torch.full_like(log_det_Psi, -100))
        
        # Compute outer product (y_i - μ_0)(y_i - μ_0)^T
        delta = targets - mu  # (B, 3)
        delta_outer = torch.bmm(delta.unsqueeze(-1), delta.unsqueeze(-2))  # (B, 3, 3)
        
        # Compute Ψ + κ/(1+κ) * (y_i - μ_0)(y_i - μ_0)^T
        coeff = kappa / (1 + kappa)  # (B,)
        coeff = coeff.view(-1, 1, 1) # (B, 1, 1)
        
        Psi_plus_term = Psi_reg + coeff * delta_outer  # (B, 3, 3)
        
        # Compute log|Ψ + κ/(1+κ)(y_i-μ_0)(y_i-μ_0)^T|
        sign_plus, log_det_Psi_plus = torch.linalg.slogdet(Psi_plus_term)
        log_det_Psi_plus = torch.where(sign_plus > 0, log_det_Psi_plus, 
                                    torch.full_like(log_det_Psi_plus, -100))
        
        # Compute the NLL terms according to equation (10)
        n = self.d  # dimension
        
        # Term 1: log Γ((ν-n+1)/2) - log Γ((ν+1)/2)
        # gamma_term = torch.lgamma((nu - n + 1) / 2) - torch.lgamma((nu + 1) / 2)
        with torch.cuda.amp.autocast(enabled=False):
            gamma_term = torch.lgamma(((nu - n + 1) / 2).float()) \
                    - torch.lgamma(((nu + 1) / 2).float())
        # Term 2: (n/2) * log(π(1+κ)/κ)
        pi_term = (n / 2) * torch.log(math.pi * (1 + kappa) / kappa)
        
        # Term 3: -(ν/2) * log|Ψ|

        psi_term = -(nu / 2) * log_det_Psi.view(-1, 1)

        # Term 4: ((ν+1)/2) * log|Ψ + κ/(1+κ)(y_i-μ_0)(y_i-μ_0)^T|
        psi_plus_term = ((nu + 1) / 2) * log_det_Psi_plus.view(-1, 1)

        # Total NLL
        nll = gamma_term + pi_term + psi_term + psi_plus_term

        # Handle any NaN or inf values
        nll = torch.where(torch.isfinite(nll), nll, torch.full_like(nll, 1e6))

        
        loss = nll # + self.lam * reg_term  # (B,)
        
        return nll.mean(), kappa, nu
    

    def get_uncertainties(self, mu, S, lognu):
        """
        Extracts uncertainties from the model parameters.
        
        :mu:     (B,3)   predicted means
        :S:      (B,3,3) predicted covariance matrices
        :kappa:  (B,1) 
        :nu:     (B,1)
        
        Returns:
            Tuple of tensors containing the uncertainties.
        """
        # kappa = F.softplus(logkappa)  # (B,1) -> (B,)
        # kappa = torch.clamp(kappa, min=self.epsilon) #+ self.epsilon     # (B,)
        # kappa = torch.clamp(kappa, min=1e-3)  # Prevent extremely small kappa
        nu = torch.sigmoid(lognu)*(12-self.d) + (self.d+1)   # (B,1) -> (B,)
        nu = torch.clamp(nu, min=(self.d+1) + self.epsilon)  #+ self.epsilon     # (B,)
        kappa = nu
        # degrees of freedom: ν' = nu – d - 1
        df = torch.clamp(nu - self.d - 1.0, min=self.epsilon)                       # (B,)

        # Regularize S by adding eps*I to improve conditioning
        I = torch.eye(self.d, device=S.device, dtype=S.dtype).view(-1, 3, 3)  # (1,3,3)
        Psi = S + (self.epsilon * I)
        # Psi = S
        # ---- aleatoric and epistemic ----
        # both need broadcast of df and kappa to (B,1,1)
        df_unsq    = df.unsqueeze(-1)                 # (B,1,1)
        kappa_unsq = kappa.unsqueeze(-1)               # (B,1,1)
        # print(Psi.shape, df_unsq.shape, kappa_unsq.shape)
        
        aleatoric  = Psi / df_unsq                                   # (B,3,3)
        epistemic  = Psi / (kappa_unsq * df_unsq)                    # (B,3,3)
        return aleatoric, epistemic, kappa, nu