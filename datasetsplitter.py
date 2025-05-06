# split images in directory into train and test sets
import os
import random

def split_dataset(root_dir, ratio=0.8):
    """
    Splits the dataset into training and testing sets.
    
    Parameters:
        root_dir (str): The root directory containing the dataset.
        ratio (float): The ratio of training to testing data.
        
    Returns:
        None
    """
    # Get all subfolders in the root directory
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    # Iterate through each subfolder
    for subfolder in subfolders:
        folder_path = os.path.join(root_dir, subfolder)
        files = os.listdir(folder_path)
        
        # Shuffle the files randomly
        random.shuffle(files)
        
        # Calculate the split index
        split_index = int(len(files) * ratio)
        
        # Create train and test directories
        train_dir = os.path.join(root_dir, 'train', subfolder)
        test_dir = os.path.join(root_dir, 'test', subfolder)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Move files to train and test directories
        for i, file in enumerate(files):
            src_path = os.path.join(folder_path, file)
            if i < split_index:
                dest_path = os.path.join(train_dir, file)
            else:
                dest_path = os.path.join(test_dir, file)
            
            # Move the file
            os.rename(src_path, dest_path)