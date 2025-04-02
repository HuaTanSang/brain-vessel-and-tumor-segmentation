from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import glob

class BrainTumorDataset(Dataset):
    def __init__(self, root_folder):
        self.image_dir = os.path.join(root_folder, "image")
        self.mask_dir = os.path.join(root_folder, "mask")

        list_subfolder = ["1", "2", "3"]
        self.image_files = [] 
        self.mask_files = [] 

        for sub in list_subfolder: 
            image_paths = glob.glob(os.path.join(self.image_dir, sub, "*.jpg"))
            mask_paths = glob.glob(os.path.join(self.mask_dir, sub, "*.jpg"))

            image_paths.sort()
            mask_paths.sort()

            self.image_files.extend(image_paths)
            self.mask_files.extend(mask_paths)

        assert len(self.image_files) == len(self.mask_files), "Số lượng ảnh và mask không khớp"

        self.image_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),  
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        mask_path = self.mask_files[index]

        image = Image.open(image_path).convert("RGB") 
        image = self.image_transform(image)

        mask = Image.open(mask_path).convert("L")  
        mask = self.mask_transform(mask)
        mask = (mask > 0).float() 
        
        return {
            "image": image,
            "mask": mask
        }
