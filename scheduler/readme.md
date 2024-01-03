# Scheduler 

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
