import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ExponentialLR, OneCycleLR, CyclicLR

class SchedulerManager:
    def __init__(self):
        self.configs = {
            'CosineAnnealingWarmRestarts': {'T_0': 10, 'T_mult': 1, 'eta_min': 0.0005},
            'StepLR': {'step_size': 10, 'gamma': 0.1},
            'ExponentialLR': {'gamma': 0.95},
            'OneCycleLR': {'max_lr': 0.01, 'steps_per_epoch': 10, 'epochs': 20},
            'CyclicLR': {'base_lr': 0.001, 'max_lr': 0.01, 'step_size_up': 5, 'step_size_down': 5, 'mode': 'triangular'}
        }

    def initialize_scheduler(self, optimizer, scheduler_name):
        if scheduler_name not in self.configs:
            raise ValueError("Unsupported scheduler type")

        scheduler_params = self.configs[scheduler_name]

        if scheduler_name == 'CosineAnnealingWarmRestarts':
            return CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
        elif scheduler_name == 'StepLR':
            return StepLR(optimizer, **scheduler_params)
        elif scheduler_name == 'ExponentialLR':
            return ExponentialLR(optimizer, **scheduler_params)
        elif scheduler_name == 'OneCycleLR':
            return OneCycleLR(optimizer, **scheduler_params)
        elif scheduler_name == 'CyclicLR':
            return CyclicLR(optimizer, **scheduler_params)

