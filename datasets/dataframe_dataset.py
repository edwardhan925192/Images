import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class DataFrameDataset(Dataset):
    def __init__(self, dataframe, image_column, target_columns, transform=None):
        '''
        suppose image path and targets are all given in dataframe
        '''
        self.dataframe = dataframe
        self.image_column = image_column
        self.target_columns = target_columns
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # -- load the image
        img_path = self.dataframe.iloc[idx][self.image_column]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for consistency

        if self.transform:
            image = self.transform(image)

        # -- targets
        targets = self.dataframe.iloc[idx][self.target_columns]
        targets = torch.tensor(targets.values, dtype=torch.float32)

        return image, targets
