from itsdangerous import URLSafeTimedSerializer
from flask import current_app
from app.extensions import mail
import pickle
from app.models import Models

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
    """
    Mengambil model aktif dari database.
    Pastikan hanya ada satu model dengan is_active=True.
    """
    active_model = Models.query.filter_by(is_active=True).first()
    if not active_model:
        return None
    model = pickle.loads(active_model.file)
    return model