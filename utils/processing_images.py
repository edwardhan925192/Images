import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import tifffile as tiff

def preprocess_image(path):
    '''
    1. Load the image from path
    2. Add dimension if it only has one 
    3. Permute image into [C, H, W] shape
    4. Convert to tensors 
    '''
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)    

    if len(img.shape) == 2:  # Grayscale image
        img = np.tile(img[..., None], [1, 1, 3])

    img = img.astype('float32')
    mx = np.max(img)
    if mx:
        img /= mx

    img = np.transpose(img, (2, 0, 1))    
    img_ten = torch.tensor(img)
    return img_ten

def augment_image(image):
    
    image_np = image.permute(1, 2, 0).numpy()    
    
    transform = A.Compose([
        A.Resize(256,256, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.RandomCrop(height=256, width=256, always_apply=True),
        A.RandomBrightness(p=1),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

    ])

    augmented = transform(image=image_np)
    augmented_image = augmented['image']

    augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2, 0, 1)    

    return augmented_image
