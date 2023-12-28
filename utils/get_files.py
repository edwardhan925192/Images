import os

def get_files_list(image_path,image_type):
  '''
  1. Takes path of image 
  2. Takes type of image ex) 'jpg', 'png' 
  3. Returns list of image 
  '''
  image_dir = image_path  
  image_type = image_type
  
  # Check if directories exist and contain files
  if os.path.exists(image_dir) and os.path.exists(mask_dir):
      image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(f'.{image_type}')]  

      if not image_files :
          print("No files found in the directories")
  else:
      print("One or both directories do not exist")

  return image_files
