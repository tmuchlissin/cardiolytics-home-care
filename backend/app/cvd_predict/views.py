from flask import Blueprint, render_template, redirect, url_for, flash, request, send_file, make_response,  current_app
from flask_login import current_user, login_required
from markupsafe import Markup
from app.forms import PatientDataForm
from app.models import db, PatientData
from app.utils import load_active_model
import pandas as pd
import numpy as np
import os
from flask import send_file
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.units import inch
from sqlalchemy.orm import joinedload
from reportlab.platypus import KeepInFrame
import io
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

cvd_predict = Blueprint('cvd_predict', __name__)

@cvd_predict.route('/user/cvd-predict', methods=['GET', 'POST'])
@login_required
def menu():
    active_tab = request.args.get('tab', 'predict')  
    form = PatientDataForm()
    patients = PatientData.query.filter_by(user_id=current_user.id)\
                                .order_by(PatientData.submitted_at.desc()).all()
    return render_template(
        'cvd_predict.html',
        navbar_title='CVD Predict',
        form=form,
        patients=patients,
        active_tab=active_tab
    )
@cvd_predict.route('/user/cvd-predict/form', methods=['GET', 'POST'])
@login_required
def upload_menu():
    form = PatientDataForm()
    if form.validate_on_submit():

        age = form.age.data
        height = form.height.data
        weight = form.weight.data
        systolic = form.systolic.data
        diastolic = form.diastolic.data

        encoded_gender = int(form.gender.data)
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
            flash('⚠️ Model is not available. Please contact the admin.', 'warning')
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
                f'<a href="{history_url}"><u>See more details</u></a>'
            )
            flash(message, 'warning')
        else:
            message = Markup(
                f'✅ Your condition is healthy. '
                f'<a href="{history_url}"><u>See more details</u></a>'
            )
            flash(message, 'success')
        
        new_patient = PatientData(
            user_id=current_user.id,
            age=age,
            gender=bool(encoded_gender),
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
    
    return render_template(
        'cvd_predict.html',
        form=form,
        navbar_title='CVD Predict',
        active_tab='predict'
    )

    
@cvd_predict.route('/user/cvd-predict/history')
@login_required
def history_menu():
    patients = PatientData.query.filter_by(user_id=current_user.id)\
                                .order_by(PatientData.submitted_at.desc()).all()
    form = PatientDataForm()
    return render_template(
        'cvd_predict.html',
        patients=patients,
        form=form,
        navbar_title='CVD History',
        active_tab='history'
    )

def generate_observations(patient):
    """
    Menghasilkan kalimat observasi dinamis berdasarkan data medis pasien
    dan menyertakan nama lengkap serta patient_id di awal.
    """
    # Awal kalimat dengan identitas pasien
    observations = f"Observations for {patient.user.full_name} (ID : {patient.user_id}), "

    # Tekanan darah
    if patient.systolic >= 160 or patient.diastolic >= 100:
        observations += "The patient exhibits significantly high blood pressure, indicating Stage 2 Hypertension. "
    elif patient.systolic >= 140 or patient.diastolic >= 90:
        observations += "The patient shows signs of Stage 1 Hypertension. "
    elif patient.systolic >= 120 or patient.diastolic >= 80:
        observations += "There is a mild elevation in blood pressure (prehypertension). "
    else:
        observations += "Patient shows stable vital signs. "

    # Kolesterol
    if patient.cholesterol == 2:
        observations += "Cholesterol levels are well above normal. "
    elif patient.cholesterol == 1:
        observations += "Cholesterol levels are above normal. "
    else:
        observations += "Cholesterol levels are within normal range. "

    # Glukosa
    if patient.gluc == 2:
        observations += "Glucose levels are well above normal. "
    elif patient.gluc == 1:
        observations += "Glucose levels are above normal. "
    else:
        observations += "Glucose levels are within normal range. "

    # Kebiasaan merokok
    if patient.smoke:
        observations += "Patient is a smoker. "
    else:
        observations += "Patient does not smoke. "

    # Konsumsi alkohol
    if patient.alco:
        observations += "Alcohol consumption is noted. "
    else:
        observations += "No alcohol consumption is reported. "

    # Aktivitas fisik
    if patient.active:
        observations += "Patient is physically active. "
    else:
        observations += "Patient has low physical activity. "

    # Risiko kardiovaskular
    if patient.cardio:
        observations += "Overall, the patient is at risk of cardiovascular disease."
    else:
        observations += "Overall, cardiovascular risk is low. "

    # Rekomendasi umum
    observations += ("Cardio risk is monitored regularly. Recommended to maintain a healthy diet, "
                    "engage in regular exercise, and follow up with the cardiologist every 6 months.")

    return observations

@cvd_predict.route('/download_patient_pdf/<int:patient_id>', methods=['GET'])
@login_required
def download_patient_pdf(patient_id):
    patient = PatientData.query.options(joinedload(PatientData.user)).get_or_404(patient_id)
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=30,
        leftMargin=30,
        topMargin=20,
        bottomMargin=20
    )
    doc.allowSplitting = False  # Tetap pakai False agar semua konten di satu halaman

    styles = getSampleStyleSheet()
    styles['Normal'].alignment = TA_JUSTIFY
    styles['Title'].alignment = TA_CENTER
    styles['Heading2'].alignment = TA_LEFT

    elements = []

    # ---------------------------------------------------------
    # 1. HEADER (LOGO + ALAMAT) dalam satu baris
    # ---------------------------------------------------------
    logo_path = os.path.join(current_app.root_path, 'static', 'img', 'cardiolytics.png')
    try:
        logo = Image(logo_path, width=150, height=35)
    except Exception:
        logo = Paragraph("<b>Cardiolytics</b>", styles['Title'])

    header_data = [
        [
            logo,
            Paragraph(
                """
                <para align="RIGHT">
                <b>Cardiolytics Home Care</b><br/>
                Politeknik Elektronika Negeri Surabaya, Jl. Raya ITS, Keputih, Kec. Sukolilo, Surabaya, Jawa Timur 60111<br/>
                Telp: (021) 555-1234
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
    elements.append(Spacer(1, 4))

    title_paragraph = Paragraph("<b>Medical Report</b>", styles['Title'])
    elements.append(title_paragraph)
    elements.append(Spacer(1, 4))

    # ---------------------------------------------------------
    # 2. INFORMASI PASIEN
    # ---------------------------------------------------------
    patient_info_data = [
        ["Patient ID", patient.user_id],
        ["Full Name", patient.user.full_name],
        ["Email", patient.user.email],
        ["Phone Number", patient.user.phone_number],
    ]
    patient_info_table = Table(patient_info_data, colWidths=[250, 290])
    patient_info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ]))
    elements.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
    elements.append(patient_info_table)
    elements.append(Spacer(1, 4))

    # ---------------------------------------------------------
    # 3. DATA MEDIS 
    # ---------------------------------------------------------
    def categorize_level(value):
        categories = {0: "Normal", 1: "Above Normal", 2: "Well Above Normal"}
        return categories.get(value, "Unknown")

    bmi_value = round(patient.weight / ((patient.height/100)**2), 2)
    pp_value  = patient.systolic - patient.diastolic
    map_value = round((patient.systolic + 2*patient.diastolic) / 3, 2)

    medical_data_data = [
        ["Age",                          str(patient.age)],
        ["Gender",                       "Male" if patient.gender else "Female"],
        ["Height (cm)",                  str(patient.height)],
        ["Weight (kg)",                  str(int(patient.weight))],
        ["BMI (kg/m²)",                  f"{bmi_value:.2f}"],
        ["Blood Pressure (mmHg)",        f"{patient.systolic}/{patient.diastolic}"],
        ["Pulse Pressure (mmHg)",        str(pp_value)],
        ["Mean Arterial Pressure (mmHg)", f"{map_value:.2f}"],
        ["Cholesterol",                  categorize_level(patient.cholesterol)],
        ["Glucose",                      categorize_level(patient.gluc)],
        ["Smoke",                        "Yes" if patient.smoke else "No"],
        ["Alcohol",                      "Yes" if patient.alco else "No"],
        ["Physical Activity",            "Yes" if patient.active else "No"],
        ["Cardio Risk",                  "At Risk" if patient.cardio else "Healthy"],
    ]

    medical_data_table = Table(medical_data_data, colWidths=[200,240])
    medical_data_table.setStyle(TableStyle([
        ('ALIGN',       (0, 0), (-1, -1), 'LEFT'),
        ('GRID',        (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 6),
    ]))
    elements.append(Paragraph("<b>Medical Data</b>", styles['Heading2']))
    elements.append(medical_data_table)
    elements.append(Spacer(1, 12))

   # ---------------------------------------------------------
    # 4. OBSERVATIONS / RECOMMENDATIONS, FOOTER & SIGNATURE 
    # ---------------------------------------------------------
    observations_style = styles['Normal'].clone('observations_small')
    observations_style.fontSize = 10
    observations_style.leading = 12

    auto_observations = generate_observations(patient)
    observations_text = f"""
    <b>Observations :</b><br/>
    {auto_observations}
    """
    observations_para = Paragraph(observations_text, observations_style)

    footer_text = Paragraph(f'''<b>Disclaimer :</b><br/>
                            This medical record is generated automatically. For official use, please consult your healthcare provider.
                            ''',styles['Normal']
    )

    signature_label = Paragraph(
        """<para align="center"><b>Physician</b></para>""",
        styles['Normal']
    )

    signature_path = os.path.join(current_app.root_path, 'static', 'img', 'signature.png')
    try:
        signature_img = Image(signature_path, width=120, height=50)
    except Exception:
        signature_img = Paragraph("<b>No Signature Found</b>", styles['Normal'])

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
    footer_table = Table(footer_table_data, colWidths=[250, 290])  
    footer_table.setStyle(TableStyle([
        ('SPAN', (0, 0), (-1, 0)),  
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    report_generated = Paragraph(
        f"Report Generated : {patient.submitted_at.strftime('%Y-%m-%d %H:%M')}<br/><br/>")
    
    elements.append(Spacer(1, 12))
    elements.append(footer_table)
    elements.append(Spacer(1, 12))  
    elements.append(report_generated)

    # ---------------------------------------------------------
    # Bangun dokumen & kembalikan file PDF
    # ---------------------------------------------------------
    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"{patient.user_id}_record.pdf"
    )
