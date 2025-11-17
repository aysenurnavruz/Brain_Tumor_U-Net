# Bu dosya, beyin MRI segmentasyon veri setinin yüklenmesi ve işlenmesi için gerekli sınıfı içerir.
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BrainMRISegmentationDataset(Dataset):
    """
    Beyin MRI segmentasyon veri setini PyTorch Dataset formatında yükler.
    Her örnek, bir görüntü ve ona karşılık gelen maske içerir.
    Klasör yapısına veya dışarıdan verilen örnek listesine göre çalışabilir.
    """
    def __init__(self, root_dir=None, samples=None, transform=None, image_size=(256, 256)):
        self.transform = transform
        self.image_size = image_size
        if root_dir is not None:
            # Klasör yapısına göre örnekleri bul ve eşleştir
            self.samples = []
            for patient_dir in os.listdir(root_dir):
                patient_path = os.path.join(root_dir, patient_dir)
                if not os.path.isdir(patient_path):
                    continue
                files = os.listdir(patient_path)
                mask_files = [f for f in files if '_mask.tif' in f]
                for mask_file in mask_files:
                    image_file = mask_file.replace('_mask.tif', '.tif')
                    if image_file in files:
                        image_path = os.path.join(patient_path, image_file)
                        mask_path = os.path.join(patient_path, mask_file)
                        self.samples.append((image_path, mask_path))
        elif samples is not None:
            # Dışarıdan verilen örnek listesiyle başlat
            self.samples = samples
        else:
            raise ValueError("Either root_dir or samples must be provided")

    def __len__(self):
        """
        Veri setindeki toplam örnek sayısını döndürür.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Belirtilen index'teki görüntü ve maskeyi yükler, ön işler ve döndürür.
        Görüntü ve maske yeniden boyutlandırılır, normalize edilir ve istenirse transform uygulanır.
        """
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert('RGB').resize(self.image_size)
        mask = Image.open(mask_path).convert('L').resize(self.image_size)
        image = np.array(image, dtype=np.float32) / 255.0  # [0,1] aralığına normalize
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 127).astype(np.float32)  # Maskeyi binarize et
        if self.transform is not None:
            # Görüntü ve maskeye dönüşüm uygula (augmentasyon/ön işleme)
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask
