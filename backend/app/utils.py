import pickle
import io
import traceback
import os

from io import BytesIO
from datetime import datetime
from sqlalchemy.orm import joinedload
from itsdangerous import URLSafeTimedSerializer

from flask import current_app, send_file

from app.extensions import mail
from app.models import Models, PatientData
from app.wrappers import PyTorchClassifier

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.units import inch


###########################################################
##################### FORGOT PASSWORD #####################
###########################################################

def send_email_async(app, msg):
    with app.app_context():
        mail.send(msg)

def generate_reset_token(email):
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    return serializer.dumps(email, salt='password-reset-salt')

def verify_reset_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=expiration)
    except Exception:
        return None
    return email

###########################################################
##################### CVD PREDICT #########################
###########################################################

try:
    import sklearn._loss._loss as skloss
except ImportError:
    skloss = None

if skloss is not None:
    if not hasattr(skloss, '__pyx_unpickle_CyHalfBinomialLoss'):
        def dummy_unpickle_CyHalfBinomialLoss(*args, **kwargs):
            from sklearn.metrics import log_loss
            return log_loss
        setattr(skloss, '__pyx_unpickle_CyHalfBinomialLoss', dummy_unpickle_CyHalfBinomialLoss)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "PyTorchClassifier":
            return PyTorchClassifier
        return super().find_class(module, name)

def load_active_model():
    active_model = Models.query.filter_by(is_active=True).first()
    if not active_model:
        current_app.logger.error("No active model found in the database")
        return None
    try:
        file_obj = io.BytesIO(active_model.file)
        pipeline = CustomUnpickler(file_obj).load()
        return pipeline
    except Exception as e:
        current_app.logger.error(f"[MODEL LOAD ERROR] {e}\n{traceback.format_exc()}")
        return None
    
###########################################################
##################### MEDICAL REPORT ######################
###########################################################

def calculate_age(date_of_birth):
    if date_of_birth is None:
        return None
    today = datetime.today()
    age = today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
    return age if age >= 0 else None

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

    table_cell_style = styles['Normal'].clone('TableCell')
    table_cell_style.fontSize = 10
    table_cell_style.leading = 12
    table_cell_style.alignment = TA_LEFT

    bold_cell_style = styles['Normal'].clone('BoldCell')
    bold_cell_style.fontName = 'Helvetica-Bold'
    bold_cell_style.fontSize = 10
    bold_cell_style.leading = 12
    bold_cell_style.alignment = TA_LEFT
    
    elements = []

    # Header (Logo + Address)
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
    elements.append(Spacer(1, 10))

    title_paragraph = Paragraph("<b>Medical Report</b>", styles['Title'])
    elements.append(title_paragraph)
    elements.append(Spacer(1, 6))

    # Patient Information
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

    # Medical Data
    def categorize_level_cholesterol(value):
        categories = {0: "Normal (≤ 200 mg/dL)", 1: "Above normal (200–239 mg/dL)", 2: "Well above normal (≥ 240 mg/dL)"}
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

    # Observations, Footer, and Signature
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
    
    # Build Document and Return PDF
    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"{patient.user_id}_record.pdf"
    )
