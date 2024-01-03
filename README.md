# Images
repo for images in general 

# Sample usage
```markdown
!git clone 'https://github.com/edwardhan925192/Images.git'
%cd '/content/images'
from utils.get_files import get_files_list
from torch.utils.data import DataLoader
from datasets import Image_Dataset

image_dir = "path/to/image/folder"
image_type = 'jpg'
batch_size = 32

# ================= 0. Get image files paths ================== #
image_files = get_files_list(image_dir, image_type)

# ================= 1. Convert images into tensor, preprocess, augment ==================== # 
dataset = Image_Dataset(image_files, input_size=(224, 224), augmentation_transforms=None)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# ================= Train models ====================== # 
```
