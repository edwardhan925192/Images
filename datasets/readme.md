# datasets

# single folder usage
```markdown
from images.datasets import SingleFolderDataset

# -- transformation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor()
])

# -- init
dataset = SingleFolderDataset('path/to/your/single/folder', transform)

# -- loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```
