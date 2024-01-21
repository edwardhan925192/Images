# -- updated check point

def save_checkpoint(model, epoch, tag, base_directory, optimizer=None, current_val_score=None, best_scores=None):
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch
    }

    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()

    if current_val_score is not None:
        if best_scores is None:
            best_scores = [(float('inf'), ''), (float('inf'), '')]

        current_filename = f'{tag}_val{current_val_score:.4f}.pth'
        save_path = os.path.join(base_directory, current_filename)

        # Insert new score and sort
        best_scores.append((current_val_score, current_filename))
        best_scores.sort(key=lambda x: x[0])

        # Keep only the top 2 scores
        best_scores = best_scores[:2]

        # Check if current model is among the top 2
        if current_val_score in [score[0] for score in best_scores]:
            torch.save(save_dict, save_path)

        # Remove files that are no longer among the top 2
        for score, filename in best_scores[2:]:
            file_path = os.path.join(base_directory, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

    return best_scores
# ========================================= new cell ================================================= #


def dataset_prep(train_df, val_df, collater, batch_sizes, shuffle, transform ):
    # -- target col
    def generate_target_columns(start, end):
      return [str(i) for i in range(start, end + 1)]
    target_columns = generate_target_columns(1, 16)
    # --

    # -- train
    train_dataset = DataFrameDataset_custom(train_df, 'img_path', target_columns, transform=transform)

    if collater:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=shuffle, collate_fn=collater)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=shuffle)

    # -- validation
    val_dataset = DataFrameDataset_custom(val_df, 'img_path', target_columns, transform=transform)

    if collater:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_sizes, shuffle=shuffle, collate_fn=collater)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_sizes, shuffle=shuffle)


    return train_dataloader, val_dataloader

# ============================================= NEW CELL ============================================= # 

import pickle
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import pandas as pd
import copy
import torchvision.ops
from images.scheduler.scheduler import SchedulerManager
from images.datasets.singlefolder_dataset import SingleFolderDataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialization(learning_rate, model_class, scheduler, sceduler_config):
  model = model_class().to(device) # -- model = JNet().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  scheduler_manager = SchedulerManager()
  scheduler_manager.configs[scheduler] = scheduler_config # -- config update
  scheduler = scheduler_manager.initialize_scheduler(optimizer, 'CosineAnnealingWarmRestarts')
  criterion = nn.CrossEntropyLoss()  # -- criterion
  return model, optimizer, scheduler

def loading_states(load_dir, model, optimizer, scheduler):
  model, optimizer, scheduler, start_epoch = load_checkpoint(load_dir, model, optimizer, scheduler)
  return model, optimizer, scheduler, start_epoch


def training_step(model, scheduler, train_dataloader, optimizer, criterion=None, accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()  # Move zero_grad to outside the batch loop

    for batch_idx, (batch_data, batch_target) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        
        outputs = model(batch_data)
        batch_target = batch_target.long()
        loss = criterion(outputs, batch_target)
        loss = loss / accumulation_steps  # Normalize our loss (if averaged)
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:  # Perform optimizer step every 'accumulation_steps' batches
            optimizer.step()
            optimizer.zero_grad()  # Zero the gradient buffers
            total_loss += loss.item()

    average_training_loss = total_loss / len(train_dataloader)
    scheduler.step()  # Update learning rate schedule
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_training_loss}")


def validation_step(model, val_dataloader, criterion = None):
  model.eval()
  total_val_loss = 0
  val_loss = 0
  with torch.no_grad():
      for batch_data, batch_target in val_dataloader:
          batch_data, batch_target = batch_data.to(device), batch_target.to(device)
          outputs = model(batch_data)

          batch_target = batch_target.long()  # Convert target labels to torch.long
          loss = criterion(outputs, batch_target)
          val_loss += loss.item()

  total_val_loss = val_loss / len(val_dataloader)
  print(f"Average Validation Loss: {total_val_loss}")

  return total_val_loss
    

def test_step(model, test_dataloader, r_path, device):
  model = model.to(device)  
  checkpoint = torch.load(r_path, map_location=device)
  
  model.load_state_dict(checkpoint['model_state_dict'])
  all_outputs = [] 
  
  for batch_data in tqdm(test_dataloader, desc="Processing Batches"):
      batch_data = batch_data.to(device)
      with torch.no_grad():  # Ensures no computational graph is constructed
        outputs = model(batch_data)
        outputs = outputs.detach()         
        all_outputs.append(mapped_output)                  
  
  dataframe = pd.DataFrame(all_outputs)

  return dataframe
    

def check_points(model, epoch, tag, base_directory, optimizer, val_score, best_scores = None):
  best_scores = save_checkpoint(model, epoch, tag, base_directory, optimizer, val_score, best_scores, checkpoint_freq=1)
  return best_scores
  
# ========================================== NEW CELL ============================================= #

# ================ INIT ================= #
scheduler_config = {'T_0': 100, 'T_mult': 1, 'eta_min': 0.0001}
scheduler = 'CosineAnnealingWarmRestarts'

# ================ LOADING STEP ================= #
load_dir = ''
load_states = True # -- bool

# ================ DATA PREP ================= #
train_df =
val_df =
collater = custom_collate_fn # or False
batch_sizes = 24
shuffle = False
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================ TRAINING STEP ================= #
num_epochs =
learning_rate = 0.001
accumulation_steps = 1

# ================ VALIDATION STEP ================= #

# ================ TEST STEP ================= #
test_df = 

# ================ CHECK POINTS ================= #
base_directory = 'directory'
tag = 'name of the model'
best_scores = None

# ===================================== EXAMPLE USAGE ======================================== #

# -- init
model, optimizer, scheduler, start_epoch_t = initialization(learning_rate, model_class, scheduler, sceduler_config, criterion)

# -- load
if load_states:
  model_t, optimizer, scheduler_t, start_epoch = loading_states(load_dir, model, optimizer, scheduler)

# -- data
train_dataloader, val_dataloader = dataset_prep(train_df, val_df, collater, batch_sizes, shuffle, transform )

for epoch in range(start_epoch, num_epochs):
  # -- training
  training_step(model, scheduler, train_dataloader, optimizer, criterion, accumulation_steps)

  # -- validation
  val_loss = validation_step(model, val_dataloader, criterion)

  # -- saving check points
  best_scores = check_points(model, epoch, tag, base_directory, optimizer, val_loss, best_scores = None)

# -- init
model, optimizer, scheduler, start_epoch_t = initialization(learning_rate, model_class, scheduler, sceduler_config, criterion)

# -- load
if load_states:
  model_t, optimizer, scheduler_t, start_epoch = loading_states(load_dir, model, optimizer, scheduler)

# -- data
train_dataloader, val_dataloader = dataset_prep(train_df, val_df, collater, batch_sizes, shuffle, transform )

for epoch in range(start_epoch, num_epochs):
  # -- training
  training_step(model, scheduler, train_dataloader, optimizer, criterion = None)

  # -- validation
  val_loss = validation_step(model, val_dataloader, criterion = None)

  # -- saving check points
  best_scores = check_points(model, epoch, tag, base_directory, optimizer, val_loss, best_scores = None)
