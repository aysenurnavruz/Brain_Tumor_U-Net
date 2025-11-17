# Temel imaj olarak Python 3.9'u kullan
FROM python:3.9-slim

# Çalışma dizinini ayarla
WORKDIR /app

# requirements.txt dosyasını kopyala ve kütüphaneleri yükle
COPY ml_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model ./model

# Proje dosyalarını kopyala
COPY ml_service/. .

# Servisin çalışacağı port
EXPOSE 5001

# Uygulamayı başlatma komutu
CMD ["python", "app.py"]