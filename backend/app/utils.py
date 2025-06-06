import pickle
import io
import traceback

from itsdangerous import URLSafeTimedSerializer
from flask import current_app
from app.extensions import mail
from app.models import Models
from app.wrappers import PyTorchClassifier


try:
    import sklearn._loss._loss as skloss
except ImportError:
    skloss = None

if skloss is not None:
    if not hasattr(skloss, '__pyx_unpickle_CyHalfBinomialLoss'):
        def dummy_unpickle_CyHalfBinomialLoss(*args, **kwargs):
            from sklearn.metrics import log_loss
            return log_loss
        setattr(skloss, '__pyx_unpickle_CyHalfBinomialLoss', dummy_unpickle_CyHalfBinomialLoss)

def send_email_async(app, msg):
    with app.app_context():
        mail.send(msg)

def generate_reset_token(email):
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    return serializer.dumps(email, salt='password-reset-salt')

def verify_reset_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=expiration)
    except Exception:
        return None
    return email

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "PyTorchClassifier":
            return PyTorchClassifier
        return super().find_class(module, name)

def load_active_model():
    active_model = Models.query.filter_by(is_active=True).first()
    if not active_model:
        current_app.logger.error("No active model found in the database")
        return None
    try:
        file_obj = io.BytesIO(active_model.file)
        pipeline = CustomUnpickler(file_obj).load()
        return pipeline
    except Exception as e:
        current_app.logger.error(f"[MODEL LOAD ERROR] {e}\n{traceback.format_exc()}")
        return None