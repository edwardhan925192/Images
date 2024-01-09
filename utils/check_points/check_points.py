import torch
import os

def load_checkpoint(r_path, model, optimizer=None, scheduler=None):

    if not os.path.isfile(r_path):
        print(f"No checkpoint found at '{r_path}'")
        return model, optimizer, scheduler, 0

    print(f"Loading checkpoint '{r_path}'")
    checkpoint = torch.load(r_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)

    if scheduler is not None:
        for _ in range(epoch):
            scheduler.step()

    return model, optimizer, scheduler, epoch

def save_checkpoint(model, epoch, tag, base_directory, optimizer=None, checkpoint_freq=1):

    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    # Save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch
    }

    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()

    # Always update the latest checkpoint
    latest_path = os.path.join(base_directory, f'{tag}_latest_checkpoint.pth')
    torch.save(save_dict, latest_path)

    # Checkpoint frequency updates
    if (epoch + 1) % checkpoint_freq == 0:
        checkpoint_path = os.path.join(base_directory, f'{tag}_checkpoint_epoch_{epoch + 1}.pth')
        torch.save(save_dict, checkpoint_path)
