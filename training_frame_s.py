# -- torch vision transforms ( needed to be included )
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


def training_step(model, scheduler, train_dataloader, optimizer, criterion = None):
    model.train()
    total_loss = 0

    for batch_idx, (batch_data, batch_target) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        optimizer.zero_grad()
        outputs = model(batch_data)

        batch_target = batch_target.long()  # -- convert to long
        loss = criterion(outputs, batch_target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_training_loss = total_loss / len(train_dataloader)
    scheduler.step() # -- scheduler step
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

# ================ VALIDATION STEP ================= #

# ================ TEST STEP ================= #
test_df = 

# ================ CHECK POINTS ================= #
base_directory = 'directory'
tag = 'name of the model'


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
  training_step(model, scheduler, train_dataloader, optimizer, criterion = None)

  # -- validation
  val_loss = validation_step(model, val_dataloader, criterion = None)

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
