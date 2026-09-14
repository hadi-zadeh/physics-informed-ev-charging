"""
losses.py
Physics-Informed, Cost-and-Boundary-Aware Composite Imitation Loss.
Formulation:
  L_total = L_BC + lambda_tariff * L_tariff + lambda_boundary * L_boundary
"""
import torch
import torch.nn as nn


class CompositeImitationLoss(nn.Module):
    """
    Composite Imitation Loss:
    1. Standard Behavior Cloning loss (RMSE vs clairvoyant expert)
    2. Dynamic tariff weighting (amplifies gradient penalties during peak tariff hours)
    3. Differentiable soft boundary violation barrier (penalizes predicted physical SoC violations)
    """
    def __init__(self, lambda_tariff: float = 0.10, lambda_boundary: float = 0.05,
                 soc_min: float = 4.0, soc_max: float = 40.0,
                 eta_ch: float = 0.98, eta_dis: float = 0.98):
        super().__init__()
        self.lambda_tariff = lambda_tariff
        self.lambda_boundary = lambda_boundary
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis

    def forward(self, a_pred: torch.Tensor, a_gt: torch.Tensor,
                current_soc: torch.Tensor, price: torch.Tensor) -> dict:
        """
        Args:
            a_pred: Predicted continuous actions (batch_size, 1) in kW
            a_gt: Expert optimal actions (batch_size, 1) in kW
            current_soc: Current battery SoC (batch_size, 1) in kWh
            price: Current spot electricity price (batch_size, 1) in $/kWh
        Returns:
            Dictionary containing 'loss_total', 'loss_bc', 'loss_tariff', 'loss_boundary'
        """
        # 1. Standard Behavior Cloning Loss (RMSE)
        l_bc = torch.sqrt(torch.mean(torch.square(a_pred - a_gt)) + 1e-8)

        # 2. Tariff-Weighted Loss (Economic alignment)
        mean_p = torch.mean(price) + 1e-8
        tariff_weights = (price / mean_p).detach()
        l_tariff = torch.mean(tariff_weights * torch.square(a_pred - a_gt))

        # 3. Differentiable Boundary Feasibility Violation Penalty
        charge_mask = (a_pred >= 0).float()
        soc_next_raw = (
            charge_mask * (current_soc + self.eta_ch * a_pred) +
            (1.0 - charge_mask) * (current_soc + a_pred / self.eta_dis)
        )

        overcharge_violation = torch.relu(soc_next_raw - self.soc_max)
        overdischarge_violation = torch.relu(self.soc_min - soc_next_raw)
        l_boundary = torch.mean(torch.square(overcharge_violation) + torch.square(overdischarge_violation))

        l_total = l_bc + self.lambda_tariff * l_tariff + self.lambda_boundary * l_boundary

        return {
            'loss_total': l_total,
            'loss_bc': l_bc,
            'loss_tariff': l_tariff,
            'loss_boundary': l_boundary
        }
