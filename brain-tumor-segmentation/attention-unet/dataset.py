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
            # Sử dụng glob để lấy tất cả các file ảnh và mask
            image_pattern = os.path.join(self.image_dir, sub, "*.jpg")  # Giả sử ảnh có đuôi .jpg
            mask_pattern = os.path.join(self.mask_dir, sub, "*.jpg")    # Giả sử mask cũng có đuôi .jpg
            
            images_file = glob.glob(image_pattern) 
            masks_file = glob.glob(mask_pattern)

            self.image_files.extend(images_file)
            self.mask_files.extend(masks_file)

        # Sắp xếp các file
        self.image_files = sorted(self.image_files)
        self.mask_files = sorted(self.mask_files)

        assert len(self.image_files) == len(self.mask_files), "Số lượng ảnh và mask không khớp"
        
        # Define transformation for image and mask 
        self.image_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        mask_path = self.mask_files[index]

        image = Image.open(image_path)
        image = self.image_transform(image)

        mask = Image.open(mask_path)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()

        return {
            "image": image,
            "mask": mask
        }
