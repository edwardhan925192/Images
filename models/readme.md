# models

# vit main params
```markdown
img_size=[224],
patch_size=16,
in_chans=3,
embed_dim=768,
depth=12,
num_heads=12,
```

# ijepa 
```markdown
from models.ijepa import VisionTransformerPredictor, VisionTransformer
import torch.nn.functional as F
from utils.asks.mask_application import apply_masks

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
