# utils

# datasets 
```markdown
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -- define transformation
transform = transforms.Compose([      
    transforms.ToTensor()
])

root_dir = '/content/drive/rootdir'

# -- loading 
dataset = datasets.ImageFolder(root = root_dir, transform=transform)
```
