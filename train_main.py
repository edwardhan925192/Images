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

base_directory = '/content/drive/MyDrive/data/ijepa weights'
tag = 'jnet_base'
load_model = False

# -- load
# if load_model:
  # model, optimizer, scheduler, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

def train_model(train_df, val_df, learning_rate, num_epochs, batch_sizes,  base_directory, tag,):

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    training_loss_history = []
    validation_loss_history = []

    # -- add criterion
    criterion = nn.CrossEntropyLoss()

    # -- init (model should be seperately initialized)
    model = JNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler_manager = SchedulerManager()
    scheduler = scheduler_manager.initialize_scheduler(optimizer, 'CosineAnnealingWarmRestarts')

    # ==================== DATASET LOADING ========================== #
    # -- transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # -- target col
    def generate_target_columns(start, end):
      return [str(i) for i in range(start, end + 1)]
    target_columns = generate_target_columns(1, 16)
    # --

    train_dataset = DataFrameDataset_custom(train_df, 'img_path', target_columns, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True)

    # ========================== TRAINING BEGINS ======================== #
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (batch_data, batch_target) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)

            batch_target = batch_target.long()  # Convert target labels to torch.long
            loss = criterion(outputs, batch_target)

            # ============== LOSSES ==================== #
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ==================== check points ================== #
        scheduler.step()
        average_training_loss = total_loss / len(train_dataloader)
        save_checkpoint(model, epoch, tag, base_directory, optimizer=optimizer, checkpoint_freq=1)

        # =========================== VALIDATION BEGINS ============================== #
        model.eval()
        total_val_loss = 0

        # -- load val datasets
        val_dataset = DataFrameDataset_custom(val_df, 'img_path', target_columns, transform=transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_sizes, shuffle=True)

        val_loss = 0
        with torch.no_grad():
            for batch_data, batch_target in val_dataloader:
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                outputs = model(batch_data)

                batch_target = batch_target.long()  # Convert target labels to torch.long
                loss = criterion(outputs, batch_target)
                val_loss += loss.item()

        total_val_loss += val_loss / len(val_dataloader)

        # -- loss
        average_validation_loss = total_val_loss / len(val_df)
        training_loss_history.append(average_training_loss)
        validation_loss_history.append(average_validation_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_training_loss}, Average Validation Loss: {average_validation_loss}")

        # -- update best validation
        if average_validation_loss < best_val_loss:
            best_val_loss = average_validation_loss
            best_epoch = epoch
            print('=================Current_BEST======================')

    print(f"Best Epoch: {best_epoch + 1} with Validation Loss: {best_val_loss}")
