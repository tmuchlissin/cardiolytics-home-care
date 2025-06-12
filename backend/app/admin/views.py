from functools import wraps
from flask import Blueprint, render_template, redirect, url_for, flash, abort, request, current_app, send_file
from flask_login import logout_user, current_user, login_required
from app.models import UserRole, User, Models, Device, PatientData, Document
from app.extensions import db
from app.forms import EditUserForm, ModelForm, DeviceForm
from sqlalchemy import or_
from werkzeug.utils import secure_filename
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.units import inch
from sqlalchemy.orm import joinedload
import os
from datetime import datetime
from sqlalchemy import case
    
admin = Blueprint('admin', __name__, url_prefix='/admin')

def calculate_age(date_of_birth):
    if date_of_birth is None:
        return None
    today = datetime.today()
    age = today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
    return age if age >= 0 else None  

###########################################################
######################### AUTH ############################
###########################################################

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != UserRole.admin:
            abort(403) 
        return f(*args, **kwargs)
    return decorated_function

@admin.route('/logout')
@login_required
def logout():
    logout_user() 
    flash("✅ You have been logged out.", "success")
    return redirect(url_for('auth.login'))

###########################################################
####################### USER APPROVAL #####################
###########################################################

@admin.route('/user-approval')
@login_required
@admin_required
def user_approval():
    pending_users = User.query.filter(
        or_(User.approved == None, User.approved == False)
    ).all()
    return render_template('admin/user_approval.html', navbar_title="User Approval", pending_users=pending_users)

@admin.route('/approve-user/<user_id>', methods=['POST'])
@login_required
def approve_user(user_id):
    user = User.query.get(user_id)
    if user:
        user.approved = True
        db.session.commit()
        print(f"User {user.user_name} has been approved.", "success")
    else:
        print("❕User not found.", "danger")
    return redirect(url_for('admin.user_approval'))

###########################################################
##################### MANAGE USER #########################
###########################################################
@admin.route('/user-management')
@login_required
@admin_required
def user_management():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search_query = request.args.get('search', '')
    
    query = User.query.filter(User.approved == True)
    
    if search_query:
        search_pattern = f"%{search_query}%"
        query = query.filter(
            (User.id.ilike(search_pattern)) |
            (User.full_name.ilike(search_pattern)) |
            (User.user_name.ilike(search_pattern)) |
            (User.email.ilike(search_pattern)) |
            (User.phone_number.ilike(search_pattern)) |
            (User.role.ilike(search_pattern)) |
            (User.approved.ilike(search_pattern))
        )

    query = query.order_by(
        case((User.role == 'admin', 1), else_=2).asc(), 
        User.device_id.isnot(None).desc(),  
        User.created_at.asc()
    )
    
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    users = pagination.items

    devices = Device.query.order_by(Device.id).all()
    assigned_device_ids = {str(u.device_id) for u in User.query.filter(User.device_id.isnot(None)).all()}
    available_devices = [d for d in devices if str(d.id) not in assigned_device_ids]

    device_choices = [('none', 'None')] + [(str(d.id), f"{d.id}") for d in available_devices]

    form = EditUserForm()
    form.device_id.choices = device_choices

    return render_template(
        'admin/user_management.html',
        navbar_title="Manage Users",
        users=users,
        devices=devices,
        form=form, 
        pagination=pagination,
        entries_per_page=per_page,
        search_query=search_query  
    )

@admin.route("/user-management/edit-user/<string:user_id>", methods=["GET", "POST"])
@login_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search_query = request.args.get('search', '')
    
    query = User.query.filter(User.approved == True)
    if search_query:
        search_pattern = f"%{search_query}%"
        query = query.filter(
            (User.id.ilike(search_pattern)) |
            (User.full_name.ilike(search_pattern)) |
            (User.user_name.ilike(search_pattern)) |
            (User.email.ilike(search_pattern)) |
            (User.phone_number.ilike(search_pattern)) |
            (User.role.ilike(search_pattern)) |
            (User.approved.ilike(search_pattern))
        )
    
    devices = Device.query.order_by(Device.id).all()
    device_choices = [('none', 'None')] + [(d.id, f"{d.id}") for d in devices]

    if request.method == "POST":
        full_name    = request.form.get("full_name")
        user_name    = request.form.get("user_name")
        email        = request.form.get("email")
        phone_number = request.form.get("phone_number")
        role         = request.form.get("role")
        approved     = request.form.get("approved")
        device_id    = request.form.get("device_id")

        existing_email = User.query.filter(User.email == email, User.id != user.id).first()
        if existing_email:
            flash("⚠️ Email already exists!", "warning")
            return redirect(url_for("admin.user_management", page=page, per_page=per_page, search=search_query))
        
        existing_phone = User.query.filter(User.phone_number == phone_number, User.id != user.id).first()
        if existing_phone:
            flash("⚠️ Phone number already exists!", "warning")
            return redirect(url_for("admin.user_management", page=page, per_page=per_page, search=search_query))
        
        user.full_name    = full_name
        user.user_name    = user_name
        user.email        = email
        user.phone_number = phone_number
        user.role         = role
        
        if approved == "approved":
            user.approved = True
        elif approved == "rejected":
            user.approved = False
        else:
            user.approved = None
        
        if device_id in ("none", ""):
            user.device_id = None
        else:
            user.device_id = device_id
        
        db.session.commit()
        flash("✅ User successfully updated!", "success")
        return redirect(url_for("admin.user_management", page=page, per_page=per_page, search=search_query))
    
    return render_template(
        "admin/user_management.html",
        user=user,
        devices=devices,
        device_choices=device_choices,
        page=page, 
        per_page=per_page, 
        search=search_query
    )

@admin.route("/user-management/delete-user/<string:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):
    user = User.query.get(user_id)
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search_query = request.args.get('search', '')
    
    if user:
        db.session.delete(user)
        db.session.commit()
        flash("✅ User successfully deleted!", "success")
    
    return redirect(url_for("admin.user_management", page=page, per_page=per_page, search=search_query))

###########################################################
####################### PATIENT DATA ######################
###########################################################

@admin.route('/patient_data')
@login_required
def patient_data():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search_query = request.args.get('search', '')
    
    query = PatientData.query.options(joinedload(PatientData.user).joinedload(User.profile))\
                            .order_by(PatientData.submitted_at.desc())
    
    if search_query:
        query = query.filter(PatientData.user_id.contains(search_query))
    
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    patient_data_list = pagination.items
    

    patient_data_with_details = [
    {
        'patient': patient,
        'age': calculate_age(patient.user.profile.date_of_birth if patient.user and patient.user.profile else None),
        'gender': True if patient.user.profile.gender == "Male" else False if patient.user.profile.gender == "Female" else None
    }
    for patient in patient_data_list
    ]

    # for data in patient_data_with_details:
    #     print(f"Patient ID: {data['patient'].user_id}, Gender: {data['gender']}")
    
    return render_template(
        'admin/patient_data.html', 
        navbar_title="Patient Data", 
        patient_data=patient_data_with_details,
        pagination=pagination,
        entries_per_page=per_page,
        search_query=search_query
    )

# @admin.route('/admin/patient-data/delete-data/<int:patient_id>', methods=['POST'])
# @login_required
# def delete_patient(patient_id):
#     patient = PatientData.query.get_or_404(patient_id)
#     try:
#         db.session.delete(patient)
#         db.session.commit()
#         flash('Patient data deleted successfully.', 'success')
#     except Exception as e:
#         db.session.rollback()
#         flash(f'Error deleting patient data: {str(e)}', 'danger')
#     
#     return redirect(url_for(
#         'admin.patient_data'
#     ))

def generate_observations(patient):
    observations = f"Observations for {patient.user.full_name} (ID: {patient.user_id}), "

    if patient.systolic >= 160 or patient.diastolic >= 100:
        observations += "Patient shows very high blood pressure, indicative of Stage 2 Hypertension. "
    elif patient.systolic >= 140 or patient.diastolic >= 90:
        observations += "Patient shows signs of Stage 1 Hypertension. "
    elif patient.systolic >= 120 or patient.diastolic >= 80:
        observations += "Slight elevation in blood pressure (pre-hypertension). "
    else:
        observations += "Patient's vital signs are stable. "

    if patient.cholesterol == 2:
        observations += "Cholesterol level is well above normal. "
    elif patient.cholesterol == 1:
        observations += "Cholesterol level is above normal. "
    else:
        observations += "Cholesterol level is within normal range. "

    if patient.gluc == 2:
        observations += "Glucose level is well above normal. "
    elif patient.gluc == 1:
        observations += "Glucose level is above normal. "
    else:
        observations += "Glucose level is within normal range. "

    if patient.smoke:
        observations += "Patient is a smoker. "
    else:
        observations += "Patient is not a smoker. "

    if patient.alco:
        observations += "Alcohol consumption recorded. "
    else:
        observations += "No alcohol consumption reported. "

    if patient.active:
        observations += "Patient is physically active. "
    else:
        observations += "Patient has low physical activity. "

    if patient.cardio:
        observations += "Overall, patient is at risk of cardiovascular disease."
    else:
        observations += "Overall, low cardiovascular risk. "

    observations += ("Cardiovascular risk is monitored regularly. It is recommended to maintain a healthy diet, "
                    "engage in regular exercise, and consult with a cardiologist every 6 months.")

    return observations

@admin.route('/download_patient_pdf/<int:patient_id>', methods=['GET'])
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
        [Paragraph("Cardiovascular Risk", bold_cell_style), Paragraph("<b>At Risk (Diagnosed at Risk of CVD)</b>" if patient.cardio else "<b>Healthy (Diagnosed without CVD)</b>", table_cell_style)],
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
                            This medical report is generated automatically. For official use, please consult with your healthcare provider.
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

###########################################################
######################### SETTINGS ########################
###########################################################
@admin.route('/settings')
@login_required
@admin_required
def settings():
    active_tab      = request.args.get('tab', 'models')
    page            = request.args.get('page', 1, type=int)
    search_query    = request.args.get('search', '').strip()

    model           = Models.query.order_by(Models.created_at.desc()).first()

    device_q        = Device.query
    if search_query and active_tab == 'devices':
        device_q = device_q.filter(
            or_(
                Device.id.ilike(f"%{search_query}%"),
                Device.model.ilike(f"%{search_query}%")
            )
        )
    device_pagination = device_q \
        .order_by(Device.registered_at.desc()) \
        .paginate(page=page, per_page=5, error_out=False)
    devices         = device_pagination.items

    doc_q           = Document.query
    if search_query and active_tab == 'docs':
        doc_q = doc_q.filter(
            Document.title_file.ilike(f"%{search_query}%")
        )
    documents_pagination = doc_q \
        .order_by(Document.created_at.desc()) \
        .paginate(page=page, per_page=5, error_out=False)
    documents       = documents_pagination.items

    form            = ModelForm()

    return render_template(
        'admin/admin_settings.html',
        active_tab=active_tab,
        search_query=search_query,
        model=model,

        devices=devices,
        device_pagination=device_pagination,

        documents=documents,
        documents_pagination=documents_pagination,

        form=form,
        navbar_title="Settings",
    )

#------------------------- Model --------------------------#

@admin.route('/delete_model/<int:model_id>', methods=['POST'])
@login_required
@admin_required
def delete_model(model_id):
    model = Models.query.get_or_404(model_id)
    db.session.delete(model)
    db.session.commit()
    flash('✅ Model successfully deleted!', 'success')
    return redirect(url_for('admin.settings'))

ALLOWED_EXTENSIONS = {'pkl', 'joblib', 'h5'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@admin.route('/upload_model', methods=['GET', 'POST'])
@login_required
def upload_model():
    if current_user.role != UserRole.admin:
        flash('❌ You are not authorized to upload a model.', 'danger')
        return redirect(url_for('index'))

    existing_model = Models.query.first()
    if existing_model:
        flash('⚠️ A model already exists. Please delete it before uploading a new one.', 'warning')
        return redirect(url_for('admin.settings'))

    form = ModelForm()
    if form.validate_on_submit():
        file_data = form.filename.data  
        if file_data and allowed_file(file_data.filename):
            model_binary = file_data.read()
            new_filename = secure_filename(file_data.filename)
            new_model = Models(
                name=form.name.data,
                filename=new_filename,
                file=model_binary,
                is_active=True,  
                user_id=current_user.id
            )
            db.session.add(new_model)
            db.session.commit()
            flash('✅ Model successfully uploaded!', 'success')
            return redirect(url_for('admin.settings'))
        else:
            flash('No file selected or file type not allowed.', 'warning')
            return redirect(url_for('admin.settings', tab='models'))
    
    return render_template('admin/admin_settings.html', tab='models', model=None, form=form, navbar_title="Settings")

#------------------------ Device --------------------------#

@admin.route('/upload_device', methods=['GET', 'POST'])
@login_required
@admin_required
def upload_device():
    form = DeviceForm()  
    if form.validate_on_submit():
        device_id = form.device_id.data
        model = form.model.data
        
        if Device.query.filter_by(id=device_id).first():
            flash('⚠️ Device with this ID is already registered!', 'warning')
            return redirect(url_for('admin.settings', tab='devices'))
        
        new_device = Device(id=device_id, model=model)
        db.session.add(new_device)
        db.session.commit()
        flash('✅ Device successfully added!', 'success')
        return redirect(url_for('admin.settings', tab='devices'))
        
    return render_template('admin/admin_settings.html', tab='devices', form=form, navbar_title="Settings")

@admin.route('/delete_device/<string:device_id>', methods=['POST'])
@login_required
@admin_required
def delete_device(device_id):
    device = Device.query.get_or_404(device_id)
    db.session.delete(device)
    db.session.commit()
    flash('✅ Device successfully deleted!', 'success')
    return redirect(url_for('admin.settings', tab='devices'))

@admin.route('/edit_device', methods=['POST'])
@login_required
@admin_required
def edit_device():
    device_id = request.form.get('device_id')
    dev = Device.query.get_or_404(device_id)

    new_device_id = request.form.get('edit_device_id')
    new_model = request.form.get('edit_model')

    if Device.query.filter(Device.id == new_device_id, Device.id != dev.id).first():
        flash("⚠️ Device ID is already in use!", "danger")
        return redirect(url_for('admin.settings', tab='devices'))

    dev.id = new_device_id
    dev.model = new_model
    db.session.commit()

    flash("✅ Device successfully updated!", "success")
    return redirect(url_for('admin.settings', tab='devices'))
