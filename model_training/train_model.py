# Bu dosya, U-Net segmentasyon modelinin eğitim sürecini ve metrik hesaplamalarını içerir.
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model.architecture import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    dice_loss,
)
import torch.nn.functional as F

# Eğitim parametreleri ve yol ayarları
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_ROOT_DIR = "../kaggle_3m"
VAL_ROOT_DIR = "../kaggle_3m"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Bir epoch boyunca modeli verilen loader ile eğitir.
    Her batch için ileri ve geri yayılım işlemlerini gerçekleştirir.
    """
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE, dtype=torch.float32)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        # İleri yayılım
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # Geri yayılım
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Anlık kayıp bilgisini göster
        loop.set_postfix(loss=loss.item())

def main():
    """
    Eğitim ve doğrulama döngüsünü başlatır, en iyi modeli kaydeder.
    Modeli oluşturur, veri yükleyicileri hazırlar ve epoch döngüsünü yönetir.
    """
    # Eğitim ve doğrulama için dönüşüm zincirleri
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    # Modeli oluştur
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    def combined_loss(pred, target):
        bce = nn.BCEWithLogitsLoss()(pred, target)
        dice = dice_loss(pred, target)
        return bce + dice
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Eğitim ve doğrulama veri yükleyicilerini oluştur
    train_loader, val_loader = get_loaders(
        TRAIN_ROOT_DIR,
        VAL_ROOT_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    # Eğitilmiş model dosyası varsa yükle
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()
    best_dice = 0
    best_miou = 0
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, combined_loss, scaler)
        # Doğrulama setinde metrikleri hesapla (Dice ve mIoU)
        model.eval()
        dice_score, miou_score = 0, 0
        num_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                # Dice skoru hesapla
                intersection = (preds * y).sum(dim=(1,2,3))
                union = preds.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3))
                dice = (2 * intersection + 1e-8) / (union + 1e-8)
                dice_score += dice.mean()
                # mIoU hesapla
                intersection = (preds * y).sum(dim=(1,2,3))
                union = ((preds + y) >= 1).sum(dim=(1,2,3))
                iou = (intersection + 1e-8) / (union + 1e-8)
                miou_score += iou.mean()
                num_batches += 1
        dice_score = dice_score / num_batches
        miou_score = miou_score / num_batches
        print(f"Epoch {epoch + 1}: Dice={dice_score:.4f}, mIoU={miou_score:.4f}")
        # En iyi modeli kaydet (Dice skoruna göre)
        if dice_score > best_dice:
            print(f"Saving new best model (Dice={dice_score:.4f})")
            best_dice = dice_score
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "best_miou": float(miou_score),
            }
            save_checkpoint(checkpoint, filename="training/best_model.pth.tar")

if __name__ == "__main__":
    main()