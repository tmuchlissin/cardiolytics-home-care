from flask import Flask
from app.auth import auth
from app.views import main
from app.admin.views import admin
from app.bp_monitor.views import bp_monitor
from app.cvd_predict.views import cvd_predict
from app.cardiobot.views import cardiobot
from types import SimpleNamespace
from app.cardiobot.views import initialize
from app.extensions import db, migrate,  csrf, login_manager, mail
from app.models import User, PatientData, Models, Document
from config import Config
import os

def create_app():
    app = Flask(__name__)
    
    app.config.from_object(Config)
    app.config['WTF_CSRF_ENABLED'] = False
    
    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    login_manager.init_app(app)  
    mail.init_app(app)
    login_manager.login_view = 'auth.login' 

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(user_id)
    
    app.register_blueprint(auth)
    app.register_blueprint(main)
    app.register_blueprint(admin)
    app.register_blueprint(bp_monitor)
    app.register_blueprint(cvd_predict)
    app.register_blueprint(cardiobot)

    with app.app_context():
        initialize(SimpleNamespace(app=app))
    
    return app
