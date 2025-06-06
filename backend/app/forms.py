from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import IntegerField, FloatField, SelectField, SubmitField, StringField, PasswordField, EmailField, FileField, DateField
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
    submit = SubmitField('Sign in')

class RegistrationForm(FlaskForm):
    patient_id = StringField('Patient ID', validators=[
        DataRequired(message="Patient ID is required."),
        Regexp(r'^\d{10}$', message="Patient ID must be exactly 10 digits."),
        Length(min=10, max=10)
    ])
    
    full_name = StringField('Full Name', validators=[
        DataRequired(message="Full name is required."),
        Length(max=100, message="Full name must be at most 100 characters.")
    ])
    
    nickname = StringField('Nickname', validators=[
        DataRequired(message="Nickname is required."),
        Length(max=64, message="Nickname must be at most 64 characters.")
    ])
    
    email = StringField('Email', validators=[
        DataRequired(message="Email is required."),
        Email(message="Enter a valid email address."),
        Length(max=100, message="Email must be at most 100 characters.")
    ])
    
    phone_number = StringField('Phone Number', validators=[
        DataRequired(message="Phone number is required."),
        Length(min=10, max=13, message="Phone number must start with 0 and be 10-13 digits.")
    ])
    
    password = PasswordField('Password', validators=[
        DataRequired(message="Password is required."),
        Length(min=8, max=20, message="Password must be between 8 and 20 characters."),
        Regexp(
            r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])',
            message="Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character."
        )
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(message="Confirm password is required."),
        EqualTo('password', message="Passwords must match.")
    ])
    submit = SubmitField('Register')

class PatientProfileForm(FlaskForm):
    date_of_birth = DateField(
        'Date of Birth',
        validators=[DataRequired(message="Date of birth is required.")],
        format='%Y-%m-%d'
    )
    gender = SelectField(
        'Gender',
        choices=[('', 'Select Gender'), ('Male', 'Male'), ('Female', 'Female')],
        validators=[DataRequired(message="Gender is required.")]
    )
    national_id = StringField(
        'National ID Number',
        validators=[
            DataRequired(message="National ID number is required."),
            Length(min=1, max=20, message="National ID number must be between 1 and 20 characters.")
        ]
    )
    emergency_contact = StringField(
        'Emergency Contact',
        validators=[
            DataRequired(message="Emergency contact is required."),
            Length(min=1, max=20, message="Emergency contact must be between 1 and 20 characters."),
            Regexp(r'^\+?\d{10,15}$', message="Enter a valid phone number (e.g., +6281234567890).")
        ]
    )
    address = StringField(
        'Address',
        validators=[Length(max=255, message="Address must be at most 255 characters.")]
    )
    city = StringField(
        'City',
        validators=[Length(max=100, message="City must be at most 100 characters.")]
    )
    postal_code = StringField(
        'Postal Code',
        validators=[Length(max=10, message="Postal code must be at most 10 characters.")]
    )
    occupation = StringField(
        'Occupation',
        validators=[Length(max=100, message="Occupation must be at most 100 characters.")]
    )
    
class EditUserForm(FlaskForm):
    full_name = StringField("Full Name", validators=[DataRequired()])
    user_name = StringField("Nickname", validators=[DataRequired()])
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
    height = IntegerField('Height (cm)', validators=[DataRequired()])
    weight = FloatField('Weight (kg)', validators=[DataRequired()])
    systolic = IntegerField('Systolic BP (mmHg)', validators=[DataRequired()])
    diastolic = IntegerField('Diastolic BP (mmHg)', validators=[DataRequired()])
    cholesterol = SelectField('Cholesterol', 
                              choices=[('', 'None'), ('0', 'Normal (≤ 200 mg/dL)'), ('1', 'Above normal (200–239 mg/dL)'), ('2', 'Well above normal (≥ 126 mg/dL)')],
                              validators=[DataRequired()])
    gluc = SelectField('Glucose (fasting glucose)', 
                       choices=[('', 'None'), ('0', 'Normal (≤ 100 mg/dL)'), ('1', 'Above normal (100–125 mg/dL)'), ('2', 'Well above normal (≥ 126 mg/dL)')],
                       validators=[DataRequired()])
    smoke = SelectField('Smoking', 
                        choices=[('', 'None'), ('0', 'No (Non-smoker)'), ('1', 'Yes (Active smoker)')],
                        validators=[DataRequired()])
    alco = SelectField('Alcohol Consumption', 
                       choices=[('', 'None'), ('0', 'No (0 glasses/day)'), ('1', 'Yes (≥ 1 glass/day)')],
                       validators=[DataRequired()])
    active = SelectField('Physical Activity', 
                         choices=[('', 'None'), ('0', 'No (Not routine)'), ('1', 'Yes (Routine ≥ 3 times/week)')],
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
        validators=[DataRequired(message="Device ID is required"), Length(max=36)]
    )
    model = StringField(
        'Model',
        validators=[DataRequired(message="Model is required"), Length(max=50)]
    )
    submit = SubmitField('Upload Device')