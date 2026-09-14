"""
trainer.py
Training manager for Physics-Informed Imitation Learning models.
Supports Adam optimization, cosine annealing learning rate schedules, and early stopping.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .losses import CompositeImitationLoss


class Trainer:
    def __init__(self, policy: nn.Module, loss_fn: CompositeImitationLoss,
                 lr: float = 0.005, weight_decay: float = 1e-4,
                 device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.policy = policy.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_step(self, states: torch.Tensor, gt_actions: torch.Tensor,
                   current_soc: torch.Tensor, price: torch.Tensor) -> Dict[str, float]:
        self.policy.train()
        self.optimizer.zero_grad()
        a_pred = self.policy(states)
        losses = self.loss_fn(a_pred, gt_actions, current_soc, price)
        losses['loss_total'].backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self.optimizer.step()
        return {k: v.item() for k, v in losses.items()}

    def evaluate(self, states: torch.Tensor, gt_actions: torch.Tensor,
                 current_soc: torch.Tensor, price: torch.Tensor) -> Dict[str, float]:
        self.policy.eval()
        with torch.no_grad():
            a_pred = self.policy(states)
            losses = self.loss_fn(a_pred, gt_actions, current_soc, price)
        return {k: v.item() for k, v in losses.items()}
