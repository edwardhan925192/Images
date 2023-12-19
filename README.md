# Images
this repo this with images 

# Sample usage
```markdown
image_dir = "path/to/image/folder"
mask_dir = "path/to/mask/folder"
image_type = 'png'
mask_type = 'png'
batch_size = 32

# ================= Get image files paths ================== #
image_files, mask_files = get_files_list(image_dir,mask_dir, image_type, mask_type)

# ================= Convert images into tensor, preprocess, augment ==================== # 
dataset = CustomDataset(image_files, mask_files, input_size=(256, 256), augmentation_transforms=None)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# ================= Train models ====================== # 
```
