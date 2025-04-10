from itsdangerous import URLSafeTimedSerializer
from flask import current_app
from app.extensions import mail
import pickle
from app.models import Models
import joblib
import io
import sys
import types

# Impor modul sklearn._loss._loss sebagai skloss (ini adalah modul C-extension)
try:
    import sklearn._loss._loss as skloss
except ImportError:
    skloss = None

if skloss is not None:
    # Cek apakah atribut '__pyx_unpickle_CyHalfBinomialLoss' ada
    if not hasattr(skloss, '__pyx_unpickle_CyHalfBinomialLoss'):
        # Buat fungsi dummy sebagai fallback.
        # Perlu diperhatikan: fungsi dummy ini hanya sebagai placeholder.
        # Anda harus menyesuaikannya jika loss tersebut benar-benar diperlukan untuk prediksi.
        def dummy_unpickle_CyHalfBinomialLoss(*args, **kwargs):
            # Misalnya, kita kembalikan sebuah instance fungsi dummy atau nilai default.
            # Di sini kita hanya membuat dummy yang tidak melakukan apa-apa atau bisa
            # mengembalikan fungsi yang seharusnya, misalnya menggunakan log_loss.
            from sklearn.metrics import log_loss
            # Fungsi ini hanya sebagai contoh. Loss yang diinginkan mungkin berbeda.
            return log_loss
        setattr(skloss, '__pyx_unpickle_CyHalfBinomialLoss', dummy_unpickle_CyHalfBinomialLoss)


def send_email_async(app, msg):
    """Kirim email dalam thread terpisah menggunakan app context."""
    with app.app_context():
        mail.send(msg)

def generate_reset_token(email):
    """Buat token reset password menggunakan itsdangerous."""
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    return serializer.dumps(email, salt='password-reset-salt')

def verify_reset_token(token, expiration=3600):
    """Verifikasi token reset password."""
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=expiration)
    except Exception:
        return None
    return email

def load_active_model():
    active_model = Models.query.filter_by(is_active=True).first()
    if not active_model:
        return None

    file_obj = io.BytesIO(active_model.file)
    full_pipeline = joblib.load(file_obj)
    return full_pipeline