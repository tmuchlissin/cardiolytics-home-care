from flask_sqlalchemy import SQLAlchemy
from app.extensions import db
from datetime import datetime
import enum
from flask_login import UserMixin
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy import LargeBinary

class UserRole(enum.Enum):
    user = 'user'
    admin = 'admin'

class User(db.Model, UserMixin):
    id = db.Column(db.String(10), primary_key=True)
    full_name = db.Column(db.String(100), index=True, nullable=False)
    user_name = db.Column(db.String(64), index=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    phone_number = db.Column(db.String(20), index=True, unique=True)
    role = db.Column(db.Enum(UserRole), default=UserRole.user)
    approved = db.Column(db.Boolean, default=None)
    device_id = db.Column(db.String(36), db.ForeignKey('device.id', ondelete="SET NULL"), unique=True, nullable=True) 
    created_at = db.Column(db.DateTime, index=True, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, index=True, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    device = db.relationship('Device', back_populates='user') 

    def get_id(self):
        return str(self.id)


class Device(db.Model):
    id = db.Column(db.String(36), primary_key=True)  
    model = db.Column(db.String(50), nullable=False) 
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)  
    updated_at = db.Column(db.DateTime, index=True, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())
     
    user = db.relationship('User', back_populates='device', uselist=False)  
    blood_pressure_records = db.relationship('BloodPressureRecord', back_populates='device', lazy=True)  

    def __repr__(self):
        return f"<Device {self.id}, Model: {self.model}, Registered At: {self.registered_at}>"

class BloodPressureRecord(db.Model):
    id = db.Column(db.String(36), primary_key=True) 
    device_id = db.Column(db.String(36), db.ForeignKey('device.id'), nullable=False)  
    systolic = db.Column(db.Integer, nullable=False)  
    diastolic = db.Column(db.Integer, nullable=False)  
    pulse_rate = db.Column(db.Integer, nullable=False) 
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)  

    device = db.relationship('Device', back_populates='blood_pressure_records') 

    def __repr__(self):
        return f"<BloodPressureRecord {self.id}, Device: {self.device_id}, Systolic: {self.systolic}, Diastolic: {self.diastolic}, Pulse: {self.pulse_rate}, Timestamp: {self.timestamp}>"

    
class Models(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), index=True, nullable=False)
    filename = db.Column(db.String(120))
    file = db.Column(LargeBinary, nullable=False)  
    is_active = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.String(64), db.ForeignKey('user.id'), nullable=False)  
    user = db.relationship('User', backref=db.backref('models', lazy=True))
    created_at = db.Column(db.DateTime, index=True, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, index=True, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    @classmethod
    def get_admin_models(cls):
        """Mengambil data model yang hanya dapat diakses oleh admin"""
        return cls.query.join(User).filter(User.role == UserRole.admin).all()
    
class PatientData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(10), db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref='patient_data')
    age = db.Column(db.Integer, nullable=False)  
    height = db.Column(db.Integer, nullable=False)  
    weight = db.Column(db.Integer, nullable=False)  
    gender = db.Column(db.Boolean, nullable=False) 
    systolic = db.Column(db.Integer, nullable=False) 
    diastolic = db.Column(db.Integer, nullable=False)
    bmi =  db.Column(db.Float, nullable=False)
    map =  db.Column(db.Float, nullable=False)
    pulse_pressure =  db.Column(db.Float, nullable=False)
    cholesterol = db.Column(db.Integer, nullable=False) 
    gluc = db.Column(db.Integer, nullable=False) 
    smoke = db.Column(db.Boolean, nullable=False)  
    alco = db.Column(db.Boolean, nullable=False) 
    active = db.Column(db.Boolean, nullable=False)  
    cardio = db.Column(db.Boolean, nullable=False) 
    submitted_at = db.Column(db.DateTime, index=True, default=db.func.current_timestamp())
    
    def __repr__(self):
        return f"<PatientData(id={self.id}, age={self.age}, gender={self.gender}, cardio={self.cardio})>"
    
class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title_file = db.Column(db.String(255), nullable=False)
    file_data = db.Column(db.LargeBinary)  
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Document {self.title_file}>'
