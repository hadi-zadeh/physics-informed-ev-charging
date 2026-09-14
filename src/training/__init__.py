"""
Training and Losses Module
"""
from .losses import CompositeImitationLoss
from .trainer import Trainer

__all__ = ['CompositeImitationLoss', 'Trainer']
