# check points 

# save check points usage 
```markdown
num_epochs = 5
base_directory = '/path/to/save/checkpoints'
tag = 'my_model'

for epoch in range(num_epochs):
    # Training loop (omitted for brevity)
    # ...

    # Update the learning rate scheduler
    scheduler.step()

    # Save checkpoints at defined frequency
    save_checkpoint(model, epoch, tag, base_directory, optimizer)

```

# load check points usage 
```markdown
# Path to the checkpoint file
checkpoint_path = '/path/to/checkpoint/my_model_latest_checkpoint.pth'

# Load the checkpoint
model, optimizer, scheduler, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

# Now you can resume training from the start_epoch
for epoch in range(start_epoch, num_epochs):
    # Resume training loop
    # ...

    # Update the learning rate scheduler
    scheduler.step()

    # Save checkpoints
    save_checkpoint(model, epoch, tag, base_directory, optimizer)

```
