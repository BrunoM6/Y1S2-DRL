import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ImageBatchDataset(Dataset):
    def __init__(self, folders, data_dir, transform=None):
        self.file_paths = []
        self.transform = transform

        for folder in folders:
            if folder == "wiki":
                    label = 0
            else:
                    label = 1
            for i in range(100):
                folder_path = os.path.join(data_dir, folder, str(i).zfill(2))
                
                if os.path.exists(folder_path):
                    for f in os.listdir(folder_path):
                        self.file_paths.append((os.path.join(folder_path, f), label, folder))

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path, label, folder = self.file_paths[idx]

        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')

                if self.transform:
                    img = self.transform(img)
                
                return img, label, folder
        
        except Exception as e:
            print(f"While getting {idx} index from ImageBatchDataset, got {e}.")
            return torch.zeros((3, 512, 512)), 0, folder
