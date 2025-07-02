from functools import wraps
from datetime import datetime
from werkzeug.utils import secure_filename

from flask import Blueprint, render_template, redirect, url_for, flash, abort, request
from flask_login import logout_user, current_user, login_required

from app.models import UserRole, User, Models, Device, PatientData, Document, BloodPressureRecord
from app.extensions import db
from app.forms import EditUserForm, ModelForm, DeviceForm
from app.utils import download_patient_pdf, calculate_age

from sqlalchemy import or_
from sqlalchemy.orm import joinedload
from sqlalchemy import case

admin = Blueprint('admin', __name__, url_prefix='/admin')

def get_bp_status(systolic, diastolic):
    if systolic < 90 or diastolic < 60:
        return 'Hypotension'
    elif systolic <= 120 and diastolic <= 80:
        return 'Normal'
    elif (systolic <= 139 and diastolic <= 89) or (systolic > 120 and diastolic <= 80):
        return 'Prehypertension'
    elif systolic <= 159 or diastolic <= 99:
        return 'Stage 1 Hypertension'
    else:
        return 'Stage 2 Hypertension'

def get_hr_status(pulse_rate):
    if pulse_rate < 60:
        return 'Bradycardia'
    elif 60 <= pulse_rate <= 100:
        return 'Normal'
    else:
        return 'Tachycardia'

def serialize_bp_record(record):
    return {
        'id': record.id,
        'device_id': record.device_id,
        'systolic': record.systolic,
        'diastolic': record.diastolic,
        'pulse_rate': record.pulse_rate,
        'timestamp': record.timestamp.strftime('%d-%m-%Y, %H:%M:%S WIB') if record.timestamp else None,
    }

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
        if user.profile:
            db.session.delete(user.profile)

        if user.patient_data:
            for pdata in user.patient_data:
                db.session.delete(pdata)

        if user.documents:
            for doc in user.documents:
                db.session.delete(doc)

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
    tab = request.args.get('tab', 'bp-monitor')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search_query = request.args.get('search', '').strip()
    user_id = request.args.get('user_id', '').strip()

    users_with_devices = User.query.filter(User.device_id != None).all()

    data = None
    status = 'N/A'
    hrStatus = 'N/A'
    serialized_records = [] 

    if tab == 'bp-monitor' and user_id:
        bp_records = BloodPressureRecord.query\
            .join(Device, BloodPressureRecord.device_id == Device.id)\
            .join(User, Device.id == User.device_id)\
            .filter(User.id == user_id)\
            .order_by(BloodPressureRecord.timestamp.desc())\
            .limit(30)\
            .all()

        data = bp_records[0] if bp_records else None
        if data:
            status = get_bp_status(data.systolic, data.diastolic)
            hrStatus = get_hr_status(data.pulse_rate)
        else:
            status = 'N/A'
            hrStatus = 'N/A'

        serialized_records = [serialize_bp_record(r) for r in bp_records]


    query = PatientData.query.options(joinedload(PatientData.user).joinedload(User.profile))\
                            .order_by(PatientData.submitted_at.desc())
    
    if search_query and tab == 'cvd-predict':
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

    return render_template(
        'admin/patient_data.html', 
        navbar_title="Patient Data", 
        active_tab=tab,
        users_with_devices=users_with_devices,
        user_id=user_id,
        data=data,
        records=serialized_records, 
        status=status,
        hrStatus=hrStatus,
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

@admin.route('/download_patient_pdf/<int:patient_id>', methods=['GET'])
@login_required
@admin_required
def download_patient_pdf_route(patient_id):
    return download_patient_pdf(patient_id)

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
