import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import labeler
from StringIDMapper import StringIDMapper

class ImageLabel:
    def __init__(self, image_path, labels):
        self.image_path = image_path
        self.labels = labels

class CustomImageDataset(Dataset):
    def __init__(self, root_dir,device,string_id_mapper: StringIDMapper=None):
        self.transform =transforms.Compose([
            transforms.Resize(256),            # Resize the smaller edge to 256 while keeping the aspect ratio.
            transforms.CenterCrop(224),        # Crop the center 224x224 region.
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.device = device
        self.samples:list[ImageLabel] = []
        # list all subfolders
        if string_id_mapper is None:
            self.string_id_mapper = labeler.get_unique_labels(root_dir)
        else:
            self.string_id_mapper = string_id_mapper
        self.num_classes =self.string_id_mapper.__len__()
        subfolders=labeler.list_folders(root_dir)
        for subfolder in subfolders:
            labels=labeler.extract_label(subfolder)
            labelids=self.labels_to_ids(labels)
            # get folder
            folder_path = os.path.join(root_dir, subfolder)
            # list all files in the folder
            for file in os.listdir(folder_path):
                path=os.path.join(folder_path, file)
                # check if file exists
                if not os.path.exists(path):
                    print(f"File not exists: {path}")
                else:
                    y=torch.zeros(self.num_classes)
                    y[labelids] = 1.0
                    self.samples.append(ImageLabel(path,y.to(device)))
        
    def labels_to_ids(self, labels:list[str]):
        return [self.string_id_mapper.str2id(label) for label in labels]    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imglbl:ImageLabel= self.samples[idx]
        image = Image.open(imglbl.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image.to(self.device), imglbl.labels
    
    
