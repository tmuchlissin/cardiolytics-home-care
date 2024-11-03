from flask_sqlalchemy import SQLAlchemy
from app.extensions import db

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)

class PatientData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    gender = db.Column(db.String(1), nullable=False)  # Assuming 0: Male, 1: Female
    ap_hi = db.Column(db.Integer, nullable=False)
    ap_lo = db.Column(db.Integer, nullable=False)
    cholesterol = db.Column(db.String(1), nullable=False)  # Normal, Above Normal, Well Above Normal
    gluc = db.Column(db.String(1), nullable=False)  # Normal, Above Normal, Well Above Normal
    smoke = db.Column(db.Boolean, nullable=False)
    alco = db.Column(db.Boolean, nullable=False)
    active = db.Column(db.Boolean, nullable=False)
    cardio = db.Column(db.String(1), nullable=False)  # Healthy or Cardio