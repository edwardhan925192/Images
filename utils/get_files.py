import os

def get_files_list(image_path,mask_path,image_type,mask_type):
  '''
  1. Takes path of image and mask
  2. Takes type of image and mask
  3. Returns list of image and mask files
  '''
  image_dir = "path/to/image/folder"
  mask_dir = "path/to/mask/folder"

  image_type = image_type
  mask_type = mask_type

  # Check if directories exist and contain files
  if os.path.exists(image_dir) and os.path.exists(mask_dir):
      image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(f'.{image_type}')]
      mask_files = [os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith(f'.{mask_type}')]

      if not image_files or not mask_files:
          print("No files found in the directories")
  else:
      print("One or both directories do not exist")

  return image_files, mask_files
