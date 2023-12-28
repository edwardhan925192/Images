# Images
this repo this with images 

# Sample usage
```markdown
from utils.get_files import get_files_list

image_dir = "path/to/image/folder"
image_type = 'png'
batch_size = 32

# ================= 0. Get image files paths ================== #
image_files = get_files_list(image_dir, image_type, mask_type)

# ================= 1. Convert images into tensor, preprocess, augment ==================== # 
dataset = CustomDataset(image_files, mask_files, input_size=(224, 224), augmentation_transforms=None)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# ================= Train models ====================== # 
```
