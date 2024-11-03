from flask_wtf import FlaskForm
from wtforms import IntegerField, FloatField, SelectField, SubmitField, StringField
from wtforms.validators import DataRequired

class PatientDataForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    age = IntegerField('Age', validators=[DataRequired()])
    height = IntegerField('Height (cm)', validators=[DataRequired()])
    weight = FloatField('Weight (kg)', validators=[DataRequired()])
    gender = SelectField('Gender', choices=[('1', 'Male'), ('2', 'Female')], validators=[DataRequired()])
    ap_hi = IntegerField('Systolic BP (mm/Hg)', validators=[DataRequired()])
    ap_lo = IntegerField('Diastolic BP (mm/Hg)', validators=[DataRequired()])
    cholesterol = SelectField('Cholesterol', choices=[('1', 'Normal'), ('2', 'Above normal'), ('3', 'Well above normal')], validators=[DataRequired()])
    gluc = SelectField('Glucose', choices=[('1', 'Normal'), ('2', 'Above normal'), ('3', 'Well above normal')], validators=[DataRequired()])
    
    # Dropdown fields for Yes/No options
    smoke = SelectField('Smoking', choices=[('0', 'No'), ('1', 'Yes')], validators=[DataRequired()])
    alco = SelectField('Alcohol intake', choices=[('0', 'No'), ('1', 'Yes')], validators=[DataRequired()])
    active = SelectField('Physical activity', choices=[('0', 'No'), ('1', 'Yes')], validators=[DataRequired()])

    cardio = SelectField('Cardiovascular Disease', choices=[('0', 'Healthy'), ('1', 'CVD')], validators=[DataRequired()])
    submit = SubmitField('Predict CVD')