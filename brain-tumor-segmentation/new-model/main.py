from torch.optim import AdamW
from loss import * 
from seg_models import ClothSegmentation
from dataset import BrainTumorDataset
from tqdm import tqdm 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from shutil import copyfile

import os 
import torch 
import torch.nn as nn 

def train_model(epoch: int, model: nn.Module, dataloader: DataLoader, optim: torch.optim.Optimizer, device: torch.device):
    model.train()

    running_loss = .0
    with tqdm(desc='Epoch %d - Training' % epoch, unit='it', total=len(dataloader)) as pb:
        for it, batch in enumerate(dataloader):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            criterion = DiceLoss() 
            masked = model(image)
            loss = criterion(masked, mask)

            # Back propagation
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()

            # Update training status
            pb.set_postfix(loss=running_loss / (it + 1))
            pb.update()


def evaluate_model(epoch: int, model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    model.eval()

    all_predictions = []
    all_masks = []

    with tqdm(desc='Epoch %d - Evaluating' % epoch, unit='it', total=len(dataloader)) as pb:
        for batch in dataloader:
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            
            with torch.no_grad():
                logits = model(image)

            prediction = torch.argmax(logits, dim=1).cpu().numpy()  # Chuyển thành (B, H, W)
            mask = mask.cpu().numpy()  # Ground truth cũng có shape (B, H, W)

            all_predictions.extend(prediction)
            all_masks.extend(mask)

            pb.update()

    scores = compute_scores(all_predictions, all_masks)
    return scores

def save_checkpoint(dict_to_save: dict, checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(dict_to_save, os.path.join(f"{checkpoint_dir}", "last_model.pth"))


def main(folder_dir, checkpoint_dir):

    # Load data 
    data = BrainTumorDataset(folder_dir)
    train_indices, val_indices = train_test_split(range(len(data)), test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(data, train_indices)
    val_dataset = torch.utils.data.Subset(data, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare for training 
    epoch = 0
    allowed_patience = 5
    best_score = 0
    compared_score = "dice"
    patience = 0
    exit_train = False

    # Define model 
    model = ClothSegmentation(3) 
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-3)
        
    
    # Training model 
    while True: 
        train_model(epoch, model, train_loader, optimizer, device)
        # validate
        scores = evaluate_model(epoch, model, eval_loader, device) 
        print(f"IOU: {scores['iou']}; Dice: {scores['dice']}")
        score = scores[compared_score]

        # Prepare for next epoch
        is_best_model = False 
        if score > best_score: 
            best_score = score
            patience = 0 
            is_best_model = True 
        else: 
            patience += 1 
        
        if patience == allowed_patience: 
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
        
        if exit_train: 
            break 

        epoch += 1 


if __name__ == "__main__":
    main(
        folder_dir='/home/huatansang/Documents/preparation-for-research/brain-tumor-segmentation/Brain-Tumor-Segmentation-Dataset',
        checkpoint_dir='/home/huatansang/Documents/preparation-for-research/brain-tumor-segmentation/new-model/checkpointt'
    )

