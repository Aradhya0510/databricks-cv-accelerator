"""
Training Module

This module provides training functionality for computer vision models.
"""

from .trainer import UnifiedTrainer, UnifiedTrainerConfig
from .trainer_serverless import UnifiedTrainerServerless

__all__ = [
    'UnifiedTrainer',
    'UnifiedTrainerConfig', 
    'UnifiedTrainerServerless'
] 