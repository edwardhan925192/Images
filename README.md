# Images
repo for images in general 

# dataset usage
```markdown
!git clone 'https://github.com/edwardhan925192/images.git'
%cd '/content/images'
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

image_dir = "path/to/image/folder"
image_type = 'jpg'
batch_size = 32

# -- transformation
transform = transforms.Compose([  
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor()
])

# -- loading 
dataset = datasets.ImageFolder(root=image_dir, transform=transform)

# -- loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

```
