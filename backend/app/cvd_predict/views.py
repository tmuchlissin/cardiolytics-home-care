import pandas as pd
import numpy as np
import os
import traceback
from datetime import datetime

from flask import (
    Blueprint, render_template, redirect, 
    url_for, flash, request, send_file, current_app
)

from flask_login import current_user, login_required
from markupsafe import Markup
from app.forms import PatientDataForm
from app.models import db, PatientData, PatientProfile, User, BloodPressureRecord
from app.utils import load_active_model
from flask import send_file
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.units import inch
from sqlalchemy.orm import joinedload
from flask import jsonify

cvd_predict = Blueprint('cvd_predict', __name__)

def calculate_age(date_of_birth):
    if date_of_birth is None:
        return None
    today = datetime.today()
    age = today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
    return age if age >= 0 else None  

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

@cvd_predict.route('/user/cvd-predict/form', methods=['GET', 'POST'])
@login_required
def upload_menu():
    form = PatientDataForm()
    
    profile = PatientProfile.query.filter_by(user_id=current_user.id).first()
    if not profile:
        flash('⚠️ Please complete your profile before submitting data.', 'warning')
        return redirect(url_for('cvd_predict.menu', tab='predict'))

    age = calculate_age(profile.date_of_birth)
    if age is None:
        flash('⚠️ Invalid date of birth or not set in your profile.', 'warning')
        return redirect(url_for('cvd_predict.menu', tab='predict'))
    
    gender = profile.gender.lower()
    encoded_gender = 1 if gender == 'male' else 0  

    if request.method == 'GET':
        device = current_user.device
        if device:
            last_bp = BloodPressureRecord.query \
                        .filter_by(device_id=device.id) \
                        .order_by(BloodPressureRecord.timestamp.desc()) \
                        .first()
            if last_bp:
                form.systolic.data  = last_bp.systolic
                form.diastolic.data = last_bp.diastolic
                
    if form.validate_on_submit():
        height = form.height.data
        weight = form.weight.data
        systolic = form.systolic.data
        diastolic = form.diastolic.data

        encoded_chol   = int(form.cholesterol.data)
        encoded_gluc   = int(form.gluc.data)
        encoded_smoke  = int(form.smoke.data)
        encoded_alco   = int(form.alco.data)
        encoded_active = int(form.active.data)
        
        bmi = round(weight / ((height / 100) ** 2), 2)
        map_value = round((systolic + 2 * diastolic) / 3, 2)
        pulse_pressure = systolic - diastolic

        features_arr = np.array([[
            age, 
            height,
            weight, 
            systolic, 
            diastolic, 
            bmi, 
            map_value, 
            pulse_pressure, 
            encoded_gender,
            encoded_chol, 
            encoded_gluc, 
            encoded_smoke, 
            encoded_alco, 
            encoded_active
        ]])
        
        feature_columns = [
            'age', 'height', 'weight', 'systolic', 'diastolic',
            'bmi', 'map', 'pulse_pressure',
            'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]

        features_df = pd.DataFrame(features_arr, columns=feature_columns)
        print("Input Features DataFrame:")
        print(features_df.head())
        
        full_pipeline = load_active_model()
        if not full_pipeline:
            flash('⚠️ Model not available. Please contact the admin.', 'warning')
            return redirect(url_for('cvd_predict.menu', tab='predict'))
        
        y_pred = full_pipeline.predict(features_df)[0]
        y_proba = full_pipeline.predict_proba(features_df)
        print("Predicted Value:", y_pred)
        print("Predicted Probabilities:", y_proba)
        
        cardio_result = bool(y_pred)
        
        history_url = url_for('cvd_predict.menu', tab='history')
        if cardio_result:
            message = Markup(
                f'⚠️ Your condition is at risk of cardiovascular disease. '
                f'<a href="{history_url}"><u>See details</u></a>'
            )
            flash(message, 'warning')
        else:
            message = Markup(
                f'✅ Your condition is healthy. '
                f'<a href="{history_url}"><u>See details</u></a>'
            )
            flash(message, 'success')
        
        new_patient = PatientData(
            user_id=current_user.id,
            height=height,
            weight=weight,
            systolic=systolic,
            diastolic=diastolic,
            bmi=bmi,  
            map=map_value,
            pulse_pressure=pulse_pressure, 
            cholesterol=encoded_chol,
            gluc=encoded_gluc,
            smoke=bool(encoded_smoke),
            alco=bool(encoded_alco),
            active=bool(encoded_active),
            cardio=cardio_result
        )
        
        db.session.add(new_patient)
        db.session.commit()
        return redirect(url_for('cvd_predict.menu', tab='predict'))
    
        
    patients = PatientData.query.filter_by(user_id=current_user.id)\
                                .options(joinedload(PatientData.user).joinedload(User.profile))\
                                .order_by(PatientData.submitted_at.desc()).all()
    patients_with_age = [(patient, calculate_age(patient.user.profile.date_of_birth if patient.user and patient.user.profile else None)) for patient in patients]
    
    return render_template(
        'main/cvd_predict.html',
        form=form,
        navbar_title='CVD Predict',
        active_tab='predict',
        patients_with_age=patients_with_age
    )

@cvd_predict.route('/api/cvd-predict', methods=['POST'])
def api_cvd_predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    try:

        age = float(data.get('age'))
        height = float(data.get('height'))
        weight = float(data.get('weight'))
        systolic = float(data.get('systolic'))
        diastolic = float(data.get('diastolic'))

        gender = int(data.get('gender')) 
        cholesterol = int(data.get('cholesterol'))
        gluc = int(data.get('gluc'))
        smoke = int(data.get('smoke'))
        alco = int(data.get('alco'))
        active = int(data.get('active'))

        bmi = round(weight / ((height / 100) ** 2), 2)
        map_value = round((systolic + 2 * diastolic) / 3, 2)
        pulse_pressure = systolic - diastolic

        features_df = pd.DataFrame([{
            'age': age,
            'height': height,
            'weight': weight,
            'systolic': systolic,
            'diastolic': diastolic,
            'bmi': bmi,
            'map': map_value,
            'pulse_pressure': pulse_pressure,
            'gender': gender,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active
        }])

        pipeline = load_active_model()
        if pipeline is None:
            current_app.logger.error("Failed to load model")
            return jsonify({'error': 'Model not available'}), 500

        pred = int(pipeline.predict(features_df)[0])
        prob = round(pipeline.predict_proba(features_df)[0][1], 4)

        result = {
            'prediction': pred,
            'label': 'Cardio Risk' if pred else 'Healthy',
            'probability': prob
        }

        return jsonify(result)

    except Exception as e:
        current_app.logger.error(f"[API PREDICT ERROR] {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

    
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

def generate_observations(patient):
    observations = f"Observations for {patient.user.full_name} (ID: {patient.user_id}), "

    if patient.systolic >= 160 or patient.diastolic >= 100:
        observations += "The patient shows very high blood pressure, indicating Stage 2 Hypertension. "
    elif patient.systolic >= 140 or patient.diastolic >= 90:
        observations += "The patient shows signs of Stage 1 Hypertension. "
    elif patient.systolic >= 120 or patient.diastolic >= 80:
        observations += "There is a mild increase in blood pressure (prehypertension). "
    else:
        observations += "The patient's vital signs are stable. "

    if patient.cholesterol == 2:
        observations += "Cholesterol levels are well above normal. "
    elif patient.cholesterol == 1:
        observations += "Cholesterol levels are above normal. "
    else:
        observations += "Cholesterol levels are within the normal range. "

    if patient.gluc == 2:
        observations += "Glucose levels are well above normal. "
    elif patient.gluc == 1:
        observations += "Glucose levels are above normal. "
    else:
        observations += "Glucose levels are within the normal range. "

    if patient.smoke:
        observations += "The patient is a smoker. "
    else:
        observations += "The patient is not a smoker. "

    if patient.alco:
        observations += "Alcohol consumption is recorded. "
    else:
        observations += "No alcohol consumption reported. "

    if patient.active:
        observations += "The patient is physically active. "
    else:
        observations += "The patient has low physical activity. "

    if patient.cardio:
        observations += "Overall, the patient is at risk of cardiovascular disease."
    else:
        observations += "Overall, cardiovascular risk is low. "

    observations += ("Cardiovascular risk should be monitored regularly. It is recommended to maintain a healthy diet, "
                    "engage in regular exercise, and consult a cardiologist every 6 months.")

    return observations

@cvd_predict.route('/download_patient_pdf/<int:patient_id>', methods=['GET'])
@login_required
def download_patient_pdf(patient_id):
    patient = PatientData.query.options(joinedload(PatientData.user)).get_or_404(patient_id)
    
    age = calculate_age(patient.user.profile.date_of_birth if patient.user and patient.user.profile else None)
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=25,
        leftMargin=25,
        topMargin=20,
        bottomMargin=20
    )
    doc.allowSplitting = False 

    styles = getSampleStyleSheet()
    styles['Normal'].alignment = TA_JUSTIFY
    styles['Title'].alignment = TA_CENTER
    styles['Heading2'].alignment = TA_LEFT

    # Define a style for table cells to ensure consistent rendering
    table_cell_style = styles['Normal'].clone('TableCell')
    table_cell_style.fontSize = 10
    table_cell_style.leading = 12
    table_cell_style.alignment = TA_LEFT

    # Define a bold style for specific labels
    bold_cell_style = styles['Normal'].clone('BoldCell')
    bold_cell_style.fontName = 'Helvetica-Bold'
    bold_cell_style.fontSize = 10
    bold_cell_style.leading = 12
    bold_cell_style.alignment = TA_LEFT
    
    elements = []

    # ---------------------------------------------------------
    # 1. HEADER (LOGO + ADDRESS) 
    # ---------------------------------------------------------
    logo_path = os.path.join(current_app.root_path, 'static', 'img', 'cardiolytics.png')
    try:
        logo = Image(logo_path, width=145, height=40)
    except Exception:
        logo = Paragraph("<b>Cardiolytics</b>", styles['Title'])

    header_data = [
        [
            logo,
            Paragraph(
                """
                <para align="RIGHT">
                <b>Cardiolytics Home Care</b><br/>
                Surabaya State Electronics Polytechnic, Jl. Raya ITS, Keputih, Sukolilo District, Surabaya, East Java 60111<br/>
                Phone: (021) 555-1234
                </para>
                """,
                styles['Normal']
            )
        ]
    ]
    header_table = Table(header_data, colWidths=[4*inch, 3.8*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 10))

    title_paragraph = Paragraph("<b>Medical Report</b>", styles['Title'])
    elements.append(title_paragraph)
    elements.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # 2. PATIENT INFORMATION
    # ---------------------------------------------------------
    patient_info_data = [
        ["Patient ID", Paragraph(patient.user_id, table_cell_style)],
        ["Full Name", Paragraph(patient.user.full_name, table_cell_style)],
        ["Email", Paragraph(patient.user.email, table_cell_style)],
        ["Phone Number", Paragraph(patient.user.phone_number, table_cell_style)],
    ]
    patient_info_table = Table(patient_info_data, colWidths=[250, 300])
    patient_info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
    ]))
    elements.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
    elements.append(patient_info_table)
    elements.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # 3. MEDICAL DATA 
    # ---------------------------------------------------------
    def categorize_level_cholesterol(value):
        categories = {0: "Normal (≤ 200 mg/dL)", 1: "Above normal (200–239 mg/dL)", 2: "Well above normal (≥ 126 mg/dL)"}
        return categories.get(value, "Unknown")
    
    def categorize_level_glucose(value):
        categories = {0: "Normal (≤ 100 mg/dL)", 1: "Above normal (100–125 mg/dL)", 2: "Well above normal (≥ 126 mg/dL)"}
        return categories.get(value, "Unknown")

    bmi_value = round(patient.weight / ((patient.height/100)**2), 2)
    pp_value = patient.systolic - patient.diastolic
    map_value = round((patient.systolic + 2*patient.diastolic) / 3, 2)

    medical_data_data = [
        ["Age", Paragraph(f"{age if age is not None else '-'} years", table_cell_style)],
        ["Gender", Paragraph(patient.user.profile.gender if patient.user and patient.user.profile else '-', table_cell_style)],
        ["Height", Paragraph(f"{patient.height} (cm)", table_cell_style)],
        ["Weight", Paragraph(f"{patient.weight:.1f} (kg)", table_cell_style)],
        ["BMI", Paragraph(f"{bmi_value:.2f} (kg/m²)", table_cell_style)],
        ["Blood Pressure", Paragraph(f"{patient.systolic}/{patient.diastolic} (mmHg)", table_cell_style)],
        ["Pulse Pressure", Paragraph(f"{pp_value} (mmHg)", table_cell_style)],
        ["Mean Arterial Pressure", Paragraph(f"{map_value:.2f} (mmHg)", table_cell_style)],
        ["Cholesterol", Paragraph(categorize_level_cholesterol(patient.cholesterol), table_cell_style)],
        ["Glucose (fasting glucose)", Paragraph(categorize_level_glucose(patient.gluc), table_cell_style)],
        ["Smoking", Paragraph("Yes (Active Smoker)" if patient.smoke else "No (Non-Smoker)", table_cell_style)],
        ["Alcohol", Paragraph("Yes (≥ 1 glass/day)" if patient.alco else "No (0 glasses/day)", table_cell_style)],
        ["Physical Activity", Paragraph("Yes (Regular ≥ 3 times/week)" if patient.active else "No (Not Regular)", table_cell_style)],
        [Paragraph("Cardiovascular Risk", bold_cell_style), Paragraph("<b>At Risk (Diagnosed with CVD Risk)</b>" if patient.cardio else "<b>Healthy (Diagnosed without CVD)</b>", table_cell_style)],
    ]

    medical_data_table = Table(medical_data_data, colWidths=[250, 300])
    medical_data_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(Paragraph("<b>Medical Data</b>", styles['Heading2']))
    elements.append(medical_data_table)
    elements.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 4. OBSERVATIONS / RECOMMENDATIONS, FOOTER & SIGNATURE 
    # ---------------------------------------------------------
    observations_style = styles['Normal'].clone('observations_small')
    observations_style.fontSize = 10
    observations_style.leading = 12

    auto_observations = generate_observations(patient)
    observations_text = f"""
    <b>Observations:</b><br/>
    {auto_observations}
    """
    observations_para = Paragraph(observations_text, observations_style)

    footer_text = Paragraph(f'''<b>Disclaimer:</b><br/>
                            This medical report was generated automatically. For official use, please consult your healthcare provider.
                            ''', styles['Normal']
    )

    signature_label = Paragraph(
        """<para align="center"><b>Physician</b></para>""",
        styles['Normal']
    )

    signature_path = os.path.join(current_app.root_path, 'static', 'img', 'signature.png')
    try:
        signature_img = Image(signature_path, width=100, height=30)
    except Exception:
        signature_img = Paragraph("<b>Signature Not Found</b>", styles['Normal'])

    signature_table_inner = Table([
        [signature_label],
        [signature_img]
    ], colWidths=[250])  

    signature_table_inner.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0, colors.white),
    ]))

    footer_table_data = [
        [observations_para],
        [footer_text, signature_table_inner]
    ]
    footer_table = Table(footer_table_data, colWidths=[250, 300])  
    footer_table.setStyle(TableStyle([
        ('SPAN', (0, 0), (-1, 0)),  
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))

    report_generated = Paragraph(
        f"Report Generated: {patient.submitted_at.strftime('%Y-%m-%d %H:%M')}<br/><br/>")
    
    elements.append(Spacer(1, 12))
    elements.append(footer_table)
    elements.append(Spacer(1, 12))  
    elements.append(report_generated)
    
    # ---------------------------------------------------------
    # REBUILD DOCUMENT & RETURN PDF 
    # ---------------------------------------------------------
    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"{patient.user_id}_record.pdf"
    )