from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *
from unet_model import UNETModelWithAttentionGate
from dataset import BrainTumorDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from shutil import copyfile

import os
import random
import torch
import torch.nn as nn

def train_model(epoch: int, model: nn.Module, dataloader: DataLoader, optim: torch.optim.Optimizer, device: torch.device):
    model.train()
    running_loss = .0
    with tqdm(desc=f'Epoch {epoch} - Training', unit='it', total=len(dataloader)) as pb:
        for it, batch in enumerate(dataloader):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            
            masked = model(image)
            loss = dice_loss(masked, mask)
            
            # Back propagation
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
            
            # Update training status
            pb.set_postfix(loss=running_loss / (it + 1))
            pb.update()
    return running_loss / len(dataloader)

def evaluate_model(epoch: int, model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_predictions = []
    all_masks = []
    
    with tqdm(desc=f'Epoch {epoch} - Evaluating', unit='it', total=len(dataloader)) as pb:
        for batch in dataloader:
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            
            with torch.no_grad():
                logits = model(image)
            
            probs = torch.sigmoid(logits)
            prediction = (probs > 0.5).long().cpu().numpy()
            mask = mask.cpu().numpy()
            
            all_predictions.extend(prediction)
            all_masks.extend(mask)
            
            pb.update()
    
    scores = compute_scores(all_predictions, all_masks)
    return scores

def save_checkpoint(dict_to_save: dict, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(dict_to_save, os.path.join(checkpoint_dir, "last_model.pth"))

def set_seed(seed=33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(folder_dir, checkpoint_dir):
    set_seed(33)
    
    # Load data
    data = BrainTumorDataset(folder_dir)
    train_indices, val_indices = train_test_split(range(len(data)), test_size=0.3, random_state=42)
    
    train_dataset = torch.utils.data.Subset(data, train_indices)
    val_dataset = torch.utils.data.Subset(data, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define model
    model = UNETModelWithAttentionGate(3, 1)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    
    epoch = 0
    allowed_patience = 5
    best_score = 0
    compared_score = "dice"
    patience = 0
    exit_train = False
    
    # Training model
    while not exit_train :
        train_loss = train_model(epoch, model, train_loader, optimizer, device)
        scores = evaluate_model(epoch, model, eval_loader, device)
        print(f"Epoch {epoch}: IOU = {scores['iou']}, Dice = {scores['dice']}, Train Loss = {train_loss:.4f}")
        
        score = scores[compared_score]
        scheduler.step(score)  # Cập nhật learning rate dựa trên Dice Score
        
        is_best_model = False
        if score > best_score:
            best_score = score
            patience = 0
            is_best_model = True
        else:
            patience += 1
        
        if patience == allowed_patience or epoch == 30:
            exit_train = True
        
        save_checkpoint({
            "epoch": epoch,
            "best_score": best_score,
            "patience": patience,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, checkpoint_dir)
        
        if is_best_model:
            copyfile(
                os.path.join(checkpoint_dir, "last_model.pth"),
                os.path.join(checkpoint_dir, "best_model.pth")
            )
        
        epoch += 1
    
if __name__ == "__main__":
    main(
        folder_dir='/home/huatansang/Documents/preparation-for-research/brain-tumor-segmentation/Brain-Tumor-Segmentation-Dataset',
        checkpoint_dir='/home/huatansang/Documents/preparation-for-research/brain-tumor-segmentation/attention-unet/checkpoint'
    )