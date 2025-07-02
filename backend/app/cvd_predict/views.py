import pandas as pd
import numpy as np
import traceback

from flask import (
    Blueprint, render_template, redirect, 
    url_for, flash, request, send_file, current_app, jsonify
)

from flask_login import current_user, login_required

from markupsafe import Markup
from sqlalchemy.orm import joinedload

from app.forms import PatientDataForm
from app.models import db, PatientData, PatientProfile, User, BloodPressureRecord
#from app.utils import load_active_model, download_patient_pdf, calculate_age
from app.utils import download_patient_pdf, calculate_age

cvd_predict = Blueprint('cvd_predict', __name__)

@cvd_predict.route('/user/cvd-predict', methods=['GET', 'POST'])
@login_required
def menu():
    active_tab = request.args.get('tab', 'predict')  
    form = PatientDataForm()
    
    patients = PatientData.query.filter_by(user_id=current_user.id)\
                                .options(joinedload(PatientData.user).joinedload(User.profile))\
                                .order_by(PatientData.submitted_at.desc()).all()
    patients_with_age = []
    for patient in patients:
        age = calculate_age(patient.user.profile.date_of_birth if patient.user and patient.user.profile else None)
        patients_with_age.append((patient, age))
    
    return render_template(
        'main/cvd_predict.html',
        navbar_title='CVD Predict',
        form=form,
        patients_with_age=patients_with_age,
        active_tab=active_tab
    )

# @cvd_predict.route('/user/cvd-predict/form', methods=['GET', 'POST'])
# @login_required
# def upload_menu():
#     form = PatientDataForm()
    
#     profile = PatientProfile.query.filter_by(user_id=current_user.id).first()
#     if not profile:
#         flash('⚠️ Please complete your profile before submitting data.', 'warning')
#         return redirect(url_for('cvd_predict.menu', tab='predict'))

#     age = calculate_age(profile.date_of_birth)
#     if age is None:
#         flash('⚠️ Invalid date of birth or not set in your profile.', 'warning')
#         return redirect(url_for('cvd_predict.menu', tab='predict'))
    
#     gender = profile.gender.lower()
#     encoded_gender = 1 if gender == 'male' else 0  

#     if request.method == 'GET':
#         device = current_user.device
#         if device:
#             last_bp = BloodPressureRecord.query \
#                         .filter_by(device_id=device.id) \
#                         .order_by(BloodPressureRecord.timestamp.desc()) \
#                         .first()
#             if last_bp:
#                 form.systolic.data  = last_bp.systolic
#                 form.diastolic.data = last_bp.diastolic
                
#     if form.validate_on_submit():
#         height = form.height.data
#         weight = form.weight.data
#         systolic = form.systolic.data
#         diastolic = form.diastolic.data

#         encoded_chol   = int(form.cholesterol.data)
#         encoded_gluc   = int(form.gluc.data)
#         encoded_smoke  = int(form.smoke.data)
#         encoded_alco   = int(form.alco.data)
#         encoded_active = int(form.active.data)
        
#         bmi = round(weight / ((height / 100) ** 2), 2)
#         map_value = round((systolic + 2 * diastolic) / 3, 2)
#         pulse_pressure = systolic - diastolic

#         features_arr = np.array([[
#             age, 
#             height,
#             weight, 
#             systolic, 
#             diastolic, 
#             bmi, 
#             map_value, 
#             pulse_pressure, 
#             encoded_gender,
#             encoded_chol, 
#             encoded_gluc, 
#             encoded_smoke, 
#             encoded_alco, 
#             encoded_active
#         ]])
        
#         feature_columns = [
#             'age', 'height', 'weight', 'systolic', 'diastolic',
#             'bmi', 'map', 'pulse_pressure',
#             'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'
#         ]

#         features_df = pd.DataFrame(features_arr, columns=feature_columns)
#         print("Input Features DataFrame:")
#         print(features_df.head())
        
#         full_pipeline = load_active_model()
#         if not full_pipeline:
#             flash('⚠️ Model not available. Please contact the admin.', 'warning')
#             return redirect(url_for('cvd_predict.menu', tab='predict'))
        
#         y_pred = full_pipeline.predict(features_df)[0]
#         y_proba = full_pipeline.predict_proba(features_df)
#         print("Predicted Value:", y_pred)
#         print("Predicted Probabilities:", y_proba)
        
#         cardio_result = bool(y_pred)
        
#         history_url = url_for('cvd_predict.menu', tab='history')
#         if cardio_result:
#             message = Markup(
#                 f'⚠️ Your condition is at risk of cardiovascular disease. '
#                 f'<a href="{history_url}"><u>See details</u></a>'
#             )
#             flash(message, 'warning')
#         else:
#             message = Markup(
#                 f'✅ Your condition is healthy. '
#                 f'<a href="{history_url}"><u>See details</u></a>'
#             )
#             flash(message, 'success')
        
#         new_patient = PatientData(
#             user_id=current_user.id,
#             height=height,
#             weight=weight,
#             systolic=systolic,
#             diastolic=diastolic,
#             bmi=bmi,  
#             map=map_value,
#             pulse_pressure=pulse_pressure, 
#             cholesterol=encoded_chol,
#             gluc=encoded_gluc,
#             smoke=bool(encoded_smoke),
#             alco=bool(encoded_alco),
#             active=bool(encoded_active),
#             cardio=cardio_result
#         )
        
#         db.session.add(new_patient)
#         db.session.commit()
#         return redirect(url_for('cvd_predict.menu', tab='predict'))
    
        
#     patients = PatientData.query.filter_by(user_id=current_user.id)\
#                                 .options(joinedload(PatientData.user).joinedload(User.profile))\
#                                 .order_by(PatientData.submitted_at.desc()).all()
#     patients_with_age = [(patient, calculate_age(patient.user.profile.date_of_birth if patient.user and patient.user.profile else None)) for patient in patients]
    
#     return render_template(
#         'main/cvd_predict.html',
#         form=form,
#         navbar_title='CVD Predict',
#         active_tab='predict',
#         patients_with_age=patients_with_age
#     )

# @cvd_predict.route('/api/cvd-predict', methods=['POST'])
# def api_cvd_predict():
#     data = request.get_json()
#     if not data:
#         return jsonify({'error': 'Invalid JSON'}), 400

#     try:

#         age = float(data.get('age'))
#         height = float(data.get('height'))
#         weight = float(data.get('weight'))
#         systolic = float(data.get('systolic'))
#         diastolic = float(data.get('diastolic'))

#         gender = int(data.get('gender')) 
#         cholesterol = int(data.get('cholesterol'))
#         gluc = int(data.get('gluc'))
#         smoke = int(data.get('smoke'))
#         alco = int(data.get('alco'))
#         active = int(data.get('active'))

#         bmi = round(weight / ((height / 100) ** 2), 2)
#         map_value = round((systolic + 2 * diastolic) / 3, 2)
#         pulse_pressure = systolic - diastolic

#         features_df = pd.DataFrame([{
#             'age': age,
#             'height': height,
#             'weight': weight,
#             'systolic': systolic,
#             'diastolic': diastolic,
#             'bmi': bmi,
#             'map': map_value,
#             'pulse_pressure': pulse_pressure,
#             'gender': gender,
#             'cholesterol': cholesterol,
#             'gluc': gluc,
#             'smoke': smoke,
#             'alco': alco,
#             'active': active
#         }])

#         pipeline = load_active_model()
#         if pipeline is None:
#             current_app.logger.error("Failed to load model")
#             return jsonify({'error': 'Model not available'}), 500

#         pred = int(pipeline.predict(features_df)[0])
#         prob = round(pipeline.predict_proba(features_df)[0][1], 4)

#         result = {
#             'prediction': pred,
#             'label': 'Cardio Risk' if pred else 'Healthy',
#             'probability': prob
#         }

#         return jsonify(result)

#     except Exception as e:
#         current_app.logger.error(f"[API PREDICT ERROR] {e}\n{traceback.format_exc()}")
#         return jsonify({'error': str(e)}), 500

    
@cvd_predict.route('/user/cvd-predict/history')
@login_required
def history_menu():
    patients = PatientData.query.filter_by(user_id=current_user.id)\
                                .options(joinedload(PatientData.user).joinedload(User.profile))\
                                .order_by(PatientData.submitted_at.desc()).all()
    
    patients_with_age = [(patient, calculate_age(patient.user.profile.date_of_birth if patient.user and patient.user.profile else None)) for patient in patients]
    
    form = PatientDataForm()
    return render_template(
        'main/cvd_predict.html',
        patients_with_age=patients_with_age,
        form=form,
        navbar_title='CVD History',
        active_tab='history'
    )

@cvd_predict.route('/download_patient_pdf/<int:patient_id>', methods=['GET'])
@login_required
def download_patient_pdf_route(patient_id):
    patient = PatientData.query.get_or_404(patient_id)
    if patient.user_id != current_user.id:
        flash('⚠️ You are not authorized to access this patient data.', 'danger')
        return redirect(url_for('cvd_predict.menu', tab='history'))
    return download_patient_pdf(patient_id)