from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import IntegerField, FloatField, SelectField, SubmitField, StringField, PasswordField, EmailField, FileField
from wtforms.validators import DataRequired, Optional, Email, EqualTo, Regexp, Length

class LoginForm(FlaskForm):
    patient_id = StringField(
        'Patient ID',
        validators=[
            DataRequired(message="Patient ID is required."),
            Length(min=10, max=10, message="Patient ID must be exactly 10 digits.")
        ]
    )
    password = PasswordField(
        'Password',
        validators=[DataRequired(message="Password is required.")]
    )
    submit = SubmitField('Login')

    
class RegistrationForm(FlaskForm):
    patient_id = StringField('Patient ID', validators=[
        DataRequired(message="Patient ID is required."),
        Regexp(r'^\d{10}$', message="Patient ID must be exactly 10 digits."),
        Length(min=10, max=10)
    ])
    
    full_name = StringField('Full Name', validators=[
        DataRequired(message="Full name is required."),
        Length(max=100, message="Full name must be at most 100 characters long.")
    ])
    
    nickname = StringField('Nickname', validators=[
        DataRequired(message="Nickname is required."),
        Length(max=64, message="Nickname must be at most 64 characters long.")
    ])
    
    email = StringField('Email', validators=[
        DataRequired(message="Email is required."),
        Email(message="Please enter a valid email address."),
        Length(max=100, message="Email must be at most 100 characters long.")
    ])
    
    phone_number = StringField('Phone Number', validators=[
        DataRequired(message="Phone Number is required."),
        Length(min=10, max=13, message="Phone number must start with 0 and be between 10-13 digits.")
    ])
    
    password = PasswordField('Password', validators=[
        DataRequired(message="Password is required."),
        Length(min=8, max=20, message="Password must be between 8 and 20 characters."),
        Regexp(
            r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])',
            message="Password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character."
        )
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(message="Please confirm your password."),
        EqualTo('password', message="Passwords must match.")
    ])
    submit = SubmitField('Register')


class EditUserForm(FlaskForm):
    full_name = StringField("Full Name", validators=[DataRequired()])
    user_name = StringField("Username", validators=[DataRequired()])
    email = EmailField("Email", validators=[DataRequired(), Email()])
    phone_number = StringField("Phone Number")
    role = SelectField("Role", choices=[("admin", "Admin"), ("user", "User")])
    approved = SelectField(
        "Approval Status", 
        choices=[
            ("approved", "Approved"),
            ("pending", "Pending"),
            ("rejected", "Rejected")
        ]
    )
    device_id = SelectField("Device ID", choices=[], coerce=str)
    submit = SubmitField("Update")

    
class PatientDataForm(FlaskForm):
    age = IntegerField('Age', validators=[DataRequired()])
    height = IntegerField('Height (cm)', validators=[DataRequired()])
    weight = FloatField('Weight (kg)', validators=[DataRequired()])
    gender = SelectField('Gender', 
                         choices=[('', 'None'), ('0', 'Female'), ('1', 'Male')],
                         validators=[DataRequired()])
    systolic = IntegerField('Systolic BP (mm/Hg)', validators=[DataRequired()])
    diastolic = IntegerField('Diastolic BP (mm/Hg)', validators=[DataRequired()])
    cholesterol = SelectField('Cholesterol', 
                              choices=[('', 'None'), ('0', 'Normal'), ('1', 'Above normal'), ('2', 'Well above normal')],
                              validators=[DataRequired()])
    gluc = SelectField('Glucose', 
                       choices=[('', 'None'), ('0', 'Normal'), ('1', 'Above normal'), ('2', 'Well above normal')],
                       validators=[DataRequired()])
    smoke = SelectField('Smoking', 
                        choices=[('', 'None'), ('0', 'No'), ('1', 'Yes')],
                        validators=[DataRequired()])
    alco = SelectField('Alcohol intake', 
                       choices=[('', 'None'), ('0', 'No'), ('1', 'Yes')],
                       validators=[DataRequired()])
    active = SelectField('Physical activity', 
                         choices=[('', 'None'), ('0', 'No'), ('1', 'Yes')],
                         validators=[DataRequired()])
    submit = SubmitField('Predict CVD')
    
class ModelForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    filename = FileField('Filename', validators=[
        FileAllowed(['pkl', 'joblib', 'h5'], 'Invalid file format!')
    ])
    submit = SubmitField('Add Model')

class DeviceForm(FlaskForm):
    device_id = StringField(
        'Device ID',
        validators=[DataRequired(message="Device ID harus diisi"), Length(max=36)]
    )
    model = StringField(
        'Model',
        validators=[DataRequired(message="Model harus diisi"), Length(max=50)]
    )
    submit = SubmitField('Upload Device')