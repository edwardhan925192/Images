# models

# ijepa 
```markdown
from models.ijepa import VisionTransformerPredictor, VisionTransformer
import torch.nn.functional as F

# -- 0. datasets
from utils.masks.maskcollator_vit import MaskCollator
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transformations
transform = transforms.Compose([      
    transforms.ToTensor()
])

root_dir = '/content/drive/MyDrive/data/vegi picture'

# -- loading 
dataset = datasets.ImageFolder(root = root_dir, transform=transform)

# -- 1. dataloader
collator = Maskcollator()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collator)

# -- 2. models
predictor = VisionTransformerPredictor()
encoder = VisionTransformer()
```
