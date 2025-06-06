# config.py
import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.getenv('DB_URI')
    # SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    # SQLALCHEMY_ENGINE_OPTIONS = {
    #     "pool_pre_ping": True,      # cek koneksi dulu sebelum reuse
    #     "pool_recycle": 1800,       # recycle koneksi setiap 30 menit
    #     "connect_args": {
    #         "connect_timeout": 10,
    #         "sslmode": "require"    # paksa SSL
    #     }
    # }
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    MAIL_SERVER = 'smtp.gmail.com' 
    MAIL_PORT = 587 
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')  
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD') 
