from flask import Flask
from app.views import main
from app.bp_monitor.views import bp_monitor
from app.cvd_predict.views import cvd_predict
from app.cardiobot.views import cardiobot
from app.extensions import db, migrate,  csrf
from config import Config
import os

def create_app():
    app = Flask(__name__)
    
    app.config.from_object(Config)
    
    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    
    app.register_blueprint(main)
    app.register_blueprint(bp_monitor)
    app.register_blueprint(cvd_predict)
    app.register_blueprint(cardiobot)

    return app
