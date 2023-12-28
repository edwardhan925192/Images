from utils.processing_images import preprocess_image
from torch.utils.data import Dataset

class Image_Dataset(Dataset):
    '''
    1. Takes list of image_files
    2. Preprocess image (convert to tensors return [C, H, W])
    3. Augment image
    '''
    def __init__(self, image_files, input_size=(224, 224), augmentation_transforms=None):
        self.image_files = image_files        
        self.input_size = input_size
        self.augmentation_transforms = augmentation_transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_path = self.image_files[idx]        

        image = preprocess_image(image_path)        

        if self.augmentation_transforms:
            image = self.augmentation_transforms(image)

        return image
