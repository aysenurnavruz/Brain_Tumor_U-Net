import os
import io
import base64
import numpy as np
from PIL import Image, ImageOps
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
from pymysql.cursors import DictCursor
import mimetypes

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')

# Veritabanı bağlantı ayarları
DB_CONFIG = {
    'host': 'db', #docker container adı
    'user': 'root',
    'password': 'root',
    'database': 'init',
    'cursorclass': DictCursor #dictionary formatında sonuçları döndürür
}

def get_db_connection():
    """
    MySQL veritabanına bağlantı oluşturur.
    Her çağrıda yeni bir bağlantı döner.
    """
    return pymysql.connect(**DB_CONFIG)

def overlay_mask_on_image(input_bytes, mask_bytes, alpha=0.5):
    """
    Segmentasyon maskesini orijinal görüntü üzerine yarı saydam şekilde bindirir.
    Sadece tümör (beyaz) bölgeler kırmızı olarak vurgulanır.
    Args:
        input_bytes (bytes): Orijinal görüntü verisi.
        mask_bytes (bytes): Segmentasyon maskesi verisi.
        alpha (float): Maskenin şeffaflık oranı.
    Returns:
        bytes: Birleştirilmiş PNG görseli.
    """
    input_img = Image.open(io.BytesIO(input_bytes)).convert('RGB')
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
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

@app.route('/')
def index():
    """
    Ana sayfa. Kullanıcı giriş yapmamışsa login sayfasına yönlendirir.
    """
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session.get('user'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Kullanıcı kayıt işlemlerini yönetir.
    E-posta benzersizliğini kontrol eder ve yeni kullanıcıyı ekler.
    """
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cursor.fetchone():
                flash('Bu e-posta zaten kayıtlı!', 'danger')
                return redirect(url_for('register'))
            cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed_password))
            conn.commit()
            flash('Kayıt başarılı! Giriş yapabilirsiniz.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            conn.rollback()
            flash(f'Kayıt sırasında bir hata oluştu: {e}', 'danger')
        finally:
            cursor.close()
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Kullanıcı giriş işlemlerini yönetir.
    E-posta ve şifre doğrulaması yapar, başarılıysa oturum başlatır.
    """
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, password FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        user = dict(user) if user else None
        if user and check_password_hash(user['password'], password):
            session['user'] = user['email']
            flash('Başarıyla giriş yaptınız!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Geçersiz e-posta veya şifre.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    """
    Kullanıcı oturumunu sonlandırır ve login sayfasına yönlendirir.
    """
    session.clear()
    flash('Başarıyla çıkış yaptınız.', 'success')
    return redirect(url_for('login'))

@app.route('/segment', methods=['GET', 'POST'])
def segment():
    """
    Segmentasyon işlemini başlatır ve sonucu kullanıcıya gösterir.
    Kullanıcıdan gelen görüntüyü ML servisine gönderir, overlay görselini alır ve kullanıcıya sunar.
    """
    if 'user' not in session:
        return redirect(url_for('login'))
    input_b64 = None
    input_mime_type = None
    overlay_b64 = None
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            flash('Dosya seçilmedi!', 'danger')
            return redirect(url_for('segment'))
        file = request.files['file']
        input_bytes = file.read()
        filename = file.filename or ''
        input_mime_type, _ = mimetypes.guess_type(filename)
        if not input_mime_type:
            input_mime_type = 'image/png'
        input_b64 = base64.b64encode(input_bytes).decode('utf-8')
        ml_url = 'http://ml_service:5001/predict'
        files = {'file': (file.filename, input_bytes)}
        try:
            response = requests.post(ml_url, files=files, timeout=60)
            response.raise_for_status()
            data = response.json()
            overlay_b64 = data.get('overlay_b64', None)
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO images (user_email, image_data, mask_data, filename, mask_filename)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session['user'], input_bytes, None, file.filename, None)
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            flash(f'Segmentasyon servisi hatası: {e}', 'danger')
    return render_template('segment.html', input_b64=input_b64, input_mime_type=input_mime_type, overlay_b64=overlay_b64)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
