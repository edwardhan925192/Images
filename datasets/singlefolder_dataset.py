from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class SingleFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (string): Path to the folder containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image

