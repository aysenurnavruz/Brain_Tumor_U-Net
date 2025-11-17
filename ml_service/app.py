from flask import Flask, request, jsonify
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import io
import base64
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model')))
from model.architecture import UNET

app = Flask(__name__)

# Model dosya yolu ve cihaz seçimi
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pth.tar')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modeli yükle ve hazırla
model = UNET(in_channels=3, out_channels=1)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model.to(DEVICE)
model.eval()

def transform_image(image_bytes) -> torch.Tensor:
    """
    Yüklenen görüntüyü modele uygun boyut ve formata dönüştürür.
    Args:
        image_bytes (bytes): Yüklenecek görüntü verisi.
    Returns:
        torch.Tensor: Modelin beklediği boyut ve normalize edilmiş tensör.
    """
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image)
    return tensor.unsqueeze(0)

def overlay_mask_on_image(input_bytes, mask_array, alpha=0.5):
    """
    Segmentasyon maskesini orijinal görüntü üzerine bindirir ve PNG olarak döndürür.
    """
    input_img = Image.open(io.BytesIO(input_bytes)).convert('RGB')
    mask_img = Image.fromarray(mask_array).convert('L')
    if mask_img.size != input_img.size:
        mask_img = mask_img.resize(input_img.size, resample=Image.NEAREST)
    mask_bin = mask_img.point(lambda p: 255 if p > 127 else 0)
    color_mask = Image.new('RGBA', input_img.size, (255,0,0,0))
    color_mask_data = color_mask.load()
    mask_data = mask_bin.load()
    for y in range(mask_bin.size[1]):
        for x in range(mask_bin.size[0]):
            if mask_data[x, y] == 255:
                color_mask_data[x, y] = (255, 0, 0, int(255*alpha))
    input_img_rgba = input_img.convert('RGBA')
    overlay_img = Image.alpha_composite(input_img_rgba, color_mask)
    buf = io.BytesIO()
    overlay_img.save(buf, format='PNG')
    return buf.getvalue()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Yüklenen görüntü için segmentasyon maskesi ve overlay görseli üretir.
    Args:
        file: POST ile gelen görüntü dosyası.
    Returns:
        JSON: mask (segmentasyon sonucu) ve overlay_b64 (base64 PNG).
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    img_bytes = file.read()
    input_tensor = transform_image(img_bytes).to(DEVICE)
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor))
        mask = (output > 0.5).float().cpu().numpy()[0, 0]
        mask_img = (mask * 255).clip(0, 255).astype(np.uint8)
        overlay_bytes = overlay_mask_on_image(img_bytes, mask_img, alpha=0.5)
        overlay_b64 = base64.b64encode(overlay_bytes).decode('utf-8')
    return jsonify({'mask': mask.tolist(), 'overlay_b64': overlay_b64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
