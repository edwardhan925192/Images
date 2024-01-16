# Scheduler configs
```markdown
'CosineAnnealingWarmRestarts': {'T_0': 10, 'T_mult': 1, 'eta_min': 0.0005},
'StepLR': {'step_size': 10, 'gamma': 0.1},
'ExponentialLR': {'gamma': 0.95},
'OneCycleLR': {'max_lr': 0.01, 'steps_per_epoch': 10, 'epochs': 20},
'CyclicLR': {'base_lr': 0.001, 'max_lr': 0.01, 'step_size_up': 5, 'step_size_down': 5, 'mode': 'triangular'}
```
# Sample usage 
```markdown
from images.scheduler.scheduler import SchedulerManager()

# -- example usage
scheduler_manager = SchedulerManager()

# -- optim == optimizer 
x_scheduler = scheduler_manager.initialize_scheduler(optim, 'CosineAnnealingWarmRestarts')
pred_scheduler = scheduler_manager.initialize_scheduler(optim, 'StepLR')

# -- training loop
for epoch in range(num_epochs):
    # -- training steps...

    # -- update the scheduler
    x_scheduler.step()
    pred_scheduler.step()

```
