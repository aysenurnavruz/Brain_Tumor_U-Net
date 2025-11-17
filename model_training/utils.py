# Bu dosya, modelin kaydedilmesi/yüklenmesi, veri yükleyicilerin oluşturulması, doğruluk ve Dice skorunun hesaplanması gibi yardımcı fonksiyonları içerir.
import torch
import torchvision
import os
from dataset import BrainMRISegmentationDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Modelin mevcut durumunu (ağırlıklar, optimizer, epoch vb.) dosyaya kaydeder.
    Eğitim sırasında en iyi modeli veya ara modelleri kaydetmek için kullanılır.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """
    Kayıtlı bir model ağırlığını yükler.
    Modelin eğitimine veya tahminine devam etmek için kullanılır.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_root_dir,
    val_root_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    """
    Eğitim ve doğrulama veri yükleyicilerini oluşturur.
    train_transform ve val_transform ile veri artırma ve ön işleme uygulanır.
    """
    train_ds = BrainMRISegmentationDataset(
        root_dir=train_root_dir,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_ds = BrainMRISegmentationDataset(
        root_dir=val_root_dir,
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    """
    Verilen veri yükleyici üzerinde modelin doğruluk ve Dice skorunu hesaplar.
    Modeli değerlendirmek için kullanılır.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    """
    Modelin tahmin ettiği maskeleri ve gerçek maskeleri görsel olarak kaydeder.
    Eğitim sonrası görsel karşılaştırma ve analiz için kullanılır.
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
    model.train()


def dice_loss(preds, targets, smooth=1e-8):
    """
    Dice loss hesaplar. Tahminlere sigmoid uygular, ardından dice loss döndürür.
    Segmentasyon modellerinde örtüşme oranını optimize etmek için kullanılır.
    """
    preds = torch.sigmoid(preds)
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return 1 - dice